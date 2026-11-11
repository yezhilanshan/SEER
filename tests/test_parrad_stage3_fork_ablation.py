from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_parrad_controller import _controller, _mask, _tokens


class _LinearEventFormer:
    def stream_update(self, previous, radar, maturity, radar_mask, prompt_cache):
        del radar_mask, prompt_cache
        step = radar.mean(dim=(1, 2))[:, None, None]
        gate = torch.as_tensor(maturity, dtype=previous.dtype, device=previous.device)
        if gate.ndim == 0:
            gate = gate.expand(previous.shape[0])
        gate = gate[:, None, None]
        return SimpleNamespace(event_tokens=previous + step * gate)


class _BatchSensitiveEventFormer(_LinearEventFormer):
    def stream_update(self, previous, radar, maturity, radar_mask, prompt_cache):
        result = super().stream_update(previous, radar, maturity, radar_mask, prompt_cache)
        return SimpleNamespace(event_tokens=result.event_tokens + previous.shape[0] * 0.01)


class _GateSensitiveEventFormer:
    def stream_update(self, previous, radar, maturity, radar_mask, prompt_cache):
        del radar_mask, prompt_cache
        gate = torch.full((*previous.shape[:2], 1), maturity, dtype=previous.dtype)
        return SimpleNamespace(event_tokens=previous + radar.mean() * gate, gates=gate)


def test_layer_probe_separates_event_state_from_projected_key_and_records_actual_gate() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    branches = diagnostic.ForkTrajectories(
        _GateSensitiveEventFormer(), lambda events: events[:, 0, :1],
        torch.tensor([[[0.0, 0.0]]]), torch.tensor([[[0.0, 2.0]]]), None,
    )

    branches.step(torch.ones((1, 4, 2)), torch.ones((1, 4), dtype=torch.bool), progressive_maturity=0.25)

    assert branches.measure_gate("pp")["mean"] == pytest.approx(0.25)
    assert branches.measure_gate("pf")["mean"] == pytest.approx(1.0)
    gate_pair = branches.pair_geometry("pp", "pf")
    assert gate_pair["event_l2"] == pytest.approx(2**0.5 * 0.75)
    assert gate_pair["key_l2"] == pytest.approx(0.75)
    assert gate_pair["gate_abs_mean"] == pytest.approx(0.75)
    init_pair = branches.pair_geometry("pp", "ap")
    assert init_pair["event_l2"] == pytest.approx(2.0)
    assert init_pair["key_l2"] == pytest.approx(0.0)


def test_four_arms_change_one_factor_at_a_time_and_measure_own_drift() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    base = torch.tensor([[[0.0]]])
    active = torch.tensor([[[2.0]]])
    projection = lambda states: states[:, 0, :]
    branches = diagnostic.ForkTrajectories(_LinearEventFormer(), projection, base, active, None)

    branches.step(torch.ones((1, 4, 1)), torch.ones((1, 4), dtype=torch.bool), progressive_maturity=0.25)

    assert branches.events[:, 0, 0].tolist() == pytest.approx([0.25, 1.0, 2.25, 3.0])
    assert branches.keys[:, 0].tolist() == pytest.approx([0.25, 1.0, 2.25, 3.0])
    assert branches.measure("pp")["delta_event"] == pytest.approx(0.25)
    assert branches.measure("ap")["delta_event"] == pytest.approx(0.25)
    assert branches.measure("af")["delta_key"] == pytest.approx(1.0)


def test_four_arms_reuse_same_input_across_updates() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    branches = diagnostic.ForkTrajectories(
        _LinearEventFormer(), lambda states: states[:, 0, :],
        torch.tensor([[[0.0]]]), torch.tensor([[[2.0]]]), None,
    )
    radar = torch.ones((1, 4, 1))
    mask = torch.ones((1, 4), dtype=torch.bool)
    branches.step(radar, mask, progressive_maturity=0.25)
    branches.step(radar, mask, progressive_maturity=0.5)

    assert branches.events[:, 0, 0].tolist() == pytest.approx([0.75, 2.0, 2.75, 4.0])
    assert branches.measure("pp")["delta_event"] == pytest.approx(0.75)
    assert branches.measure("ap")["delta_event"] == pytest.approx(0.75)


def test_reproduction_gate_rejects_wrong_maturity_or_state() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    branches = diagnostic.ForkTrajectories(
        _LinearEventFormer(), lambda states: states[:, 0, :],
        torch.tensor([[[0.0]]]), torch.tensor([[[2.0]]]), None,
    )
    branches.step(torch.ones((1, 4, 1)), torch.ones((1, 4), dtype=torch.bool), progressive_maturity=0.25)
    observed = branches.reproduction_errors(
        candidate_events=torch.tensor([[[0.25]]]), candidate_key=torch.tensor([[0.25]]),
        shadow_events=torch.tensor([[[3.0]]]), shadow_key=torch.tensor([[3.0]]),
        tolerance=1e-6,
    )
    assert observed == {"candidate_event": 0.0, "candidate_key": 0.0,
                        "shadow_event": 0.0, "shadow_key": 0.0}
    with pytest.raises(RuntimeError, match="reproduction"):
        branches.reproduction_errors(
            candidate_events=torch.tensor([[[0.5]]]), candidate_key=torch.tensor([[0.25]]),
            shadow_events=torch.tensor([[[3.0]]]), shadow_key=torch.tensor([[3.0]]),
            tolerance=1e-6,
        )


def test_each_arm_replays_batch_one_to_match_half_precision_controller() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    branches = diagnostic.ForkTrajectories(
        _BatchSensitiveEventFormer(), lambda states: states[:, 0, :],
        torch.tensor([[[0.0]]]), torch.tensor([[[2.0]]]), None,
    )
    branches.step(torch.ones((1, 4, 1)), torch.ones((1, 4), dtype=torch.bool), progressive_maturity=0.25)

    errors = branches.reproduction_errors(
        candidate_events=torch.tensor([[[0.26]]]), candidate_key=torch.tensor([[0.26]]),
        shadow_events=torch.tensor([[[3.01]]]), shadow_key=torch.tensor([[3.01]]),
        tolerance=1e-6,
    )
    assert all(value < 1e-6 for value in errors.values())


def test_ablation_rejects_test_records_before_toggling_controller() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    with pytest.raises(ValueError, match="validation"):
        diagnostic.audit_fork_ablation(
            None, (SimpleNamespace(split="test"),), None, None, None, horizon_updates=12,
        )


def test_replay_reproduces_existing_fork_and_records_all_four_arms(tmp_path) -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    forced = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    values = [1.0, 10.0, 2.0, 2.0, 2.0, 2.0]
    times = [index * 0.18 for index in range(len(values))]
    record = SimpleNamespace(
        radar_tokens=np.stack([np.full((4, 4), value, dtype=np.float32) for value in values]),
        radar_mask=np.ones((len(values), 4), dtype=np.bool_),
        timestamps=np.asarray(times, dtype=np.float64).reshape(-1, 1),
        frame_labels=np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64).reshape(-1, 1),
        oracle_run_id=np.asarray([0, 0, 1, 1, 1, 1], dtype=np.int64),
        oracle_keys=np.tile(np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(values), 1)),
    )
    item = SimpleNamespace(split="validation", key="one", metadata={"sensor_id": "14xx"}, record=record)
    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    natural = _controller(reject_timeout_s=0.18)
    physical = []
    for index, (value, timestamp) in enumerate(zip(values, times, strict=True)):
        output = natural.step_tokens(_tokens(value), _mask(), timestamp)
        physical.append(dict(recording_key="one", update_index=index, decision=output.decision,
                             proposal_probability=output.proposal_probability))
    assert [row["decision"] for row in physical] == ["none", "proposal", "reject", "none", "none", "none"]
    (baseline_dir / "upstream-summary.json").write_text(json.dumps({
        "source_split": "validation", "test_evaluation_allowed": False,
    }), encoding="utf-8")
    (baseline_dir / "physical-updates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in physical), encoding="utf-8",
    )
    (baseline_dir / "semantic-candidates.jsonl").write_text(json.dumps({
        "candidate_id": "one:1", "update_index": 2,
        "online_probability": 0.0000453978687, "online_cosine_probability": 0.0,
    }) + "\n", encoding="utf-8")
    thresholds = SimpleNamespace(
        proposal=0.8, verify_confirm=0.8, verify_reject=0.2,
        candidate_timeout_updates=1, proposal_persistence=1,
    )
    old_rows, old_integrity = forced.audit_forced_horizon(
        SimpleNamespace(controller=_controller(reject_timeout_s=0.18)),
        (item,), thresholds, baseline_dir, horizon_updates=4,
    )
    forced_dir = tmp_path / "forced"
    forced_dir.mkdir()
    (forced_dir / "forced-summary.json").write_text(json.dumps({
        "source_split": "validation", "test_evaluation_allowed": False,
        "diagnostic_only": True, "forced_horizon_updates": 4,
        "integrity": old_integrity,
    }), encoding="utf-8")
    (forced_dir / "forced-candidates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in old_rows), encoding="utf-8",
    )

    rows, integrity = diagnostic.audit_fork_ablation(
        SimpleNamespace(controller=_controller(reject_timeout_s=0.18)),
        (item,), thresholds, baseline_dir, forced_dir, horizon_updates=4,
    )

    assert [row["age_updates"] for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["clean_proposal_target"] is True
    assert set(rows[0]["arms"]) == {"pp", "pf", "ap", "af"}
    assert all(row["arms"]["pp"]["delta_event"] > 0 for row in rows)
    assert set(rows[0]["gate_stats"]) == {"pp", "pf", "ap", "af"}
    assert rows[0]["pair_geometry"]["pp_pf"]["event_l2"] >= 0
    assert rows[0]["pair_geometry"]["pp_ap"]["key_l2"] >= 0
    assert rows[0]["radar_change"]["l2"] == pytest.approx(36.0)
    assert rows[0]["physical_proposal_probability"] == pytest.approx(physical[1]["proposal_probability"])
    assert integrity["baseline_decisions_checked"] == 6
    assert integrity["baseline_forced_rows_checked"] == 4
    assert all(value == 0 for value in integrity["max_abs_reproduction_error"].values())


def test_summary_compares_the_same_full_horizon_candidates_and_sessions() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    rows = []
    for candidate, recording, sensor, target, pp, ap in (
        ("yes", "subject__set000__14xx", "14xx", True, 0.1, 0.9),
        ("no", "subject__set000__68xx", "68xx", False, 0.9, 0.1),
        ("short", "subject__set001__14xx", "14xx", True, 0.8, 0.8),
    ):
        for age in ((1,) if candidate == "short" else (1, 2)):
            scores = {feature: 0.5 for feature in diagnostic.SCORE_FIELDS}
            scores.update(a_to_pp=pp, a_to_ap=ap)
            rows.append(dict(
                candidate_id=candidate, recording_key=recording, sensor_id=sensor,
                session_key=recording.rsplit("__", 1)[0], proposal_time=float(len(rows)),
                age_updates=age, clean_proposal_target=target,
                arms={name: {"state_norm": 2.0, "key_norm": 1.0,
                             "delta_event": pp, "delta_key": ap} for name in diagnostic.ARM_NAMES},
                **scores,
            ))

    result = diagnostic.summarize_fork_rows(rows, horizon_updates=1, update_period_s=0.18)
    age = result["matched_full_horizon"]["by_age_updates"]["2"]

    assert result["matched_full_horizon"]["candidate_count"] == 2
    assert age["clean_count"] == 2
    assert age["scores"]["a_to_pp"]["auroc"] == 0.0
    assert age["scores"]["a_to_ap"]["auroc"] == 1.0
    assert age["by_session"]["subject__set000"]["a_to_ap"]["count"] == 2
    assert result["same_recording_hard_pairs"]["by_age_updates"]["2"]["pair_count"] == 0


def test_layer_summary_keeps_full_cohort_and_does_not_confuse_event_with_key_auc() -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    rows = []
    for candidate, sensor, target, event_distance, key_distance, horizon in (
        ("yes", "14xx", True, 0.8, 0.1, 2),
        ("no", "68xx", False, 0.2, 0.9, 2),
        ("short", "14xx", True, 0.95, 0.95, 1),
    ):
        for age in range(1, horizon + 1):
            geometry = {name: {"event_l2": event_distance, "event_cosine": event_distance,
                               "key_l2": key_distance, "key_cosine": key_distance,
                               "gate_abs_mean": 0.25} for name in
                        ("pp_pf", "ap_af", "pp_ap", "pf_af", "pp_af")}
            rows.append({
                "candidate_id": candidate, "session_key": "one", "sensor_id": sensor,
                "age_updates": age, "clean_proposal_target": target,
                "pair_geometry": geometry,
                "gate_stats": {name: {"mean": 0.5} for name in diagnostic.ARM_NAMES},
                "radar_change": {"l2": event_distance, "cosine": key_distance},
                "physical_proposal_probability": event_distance,
            })

    result = diagnostic.summarize_fork_layers(rows, horizon_updates=1, update_period_s=0.18)
    second = result["by_age_updates"]["2"]

    assert result["source_split"] == "validation"
    assert result["test_evaluation_allowed"] is False
    assert result["matched_full_horizon"] == 2
    assert second["clean_count"] == 2
    assert second["scores"]["pp_pf_event_l2"]["auroc"] == 1.0
    assert second["scores"]["pp_pf_key_l2"]["auroc"] == 0.0
    assert second["by_session"]["one"]["pp_pf_event_l2"]["count"] == 2
    assert second["by_sensor"]["14xx"]["pp_pf_event_l2"]["count"] == 1


def test_output_writer_preserves_validation_only_and_refuses_overwrite(tmp_path) -> None:
    diagnostic = importlib.import_module("continuous_mmwave.parrad_stage3_fork_ablation")
    output = tmp_path / "result"
    with pytest.raises(ValueError, match="validation"):
        diagnostic.write_fork_outputs([{"candidate_id": "a"}], {
            "source_split": "test", "test_evaluation_allowed": False,
        }, output)
    diagnostic.write_fork_outputs([{"candidate_id": "a"}], {
        "source_split": "validation", "test_evaluation_allowed": False,
        "diagnostic_only": True,
    }, output)
    assert json.loads((output / "fork-rows.jsonl").read_text(encoding="utf-8"))["candidate_id"] == "a"
    assert json.loads((output / "fork-summary.json").read_text(encoding="utf-8"))["diagnostic_only"] is True
    with pytest.raises(FileExistsError):
        diagnostic.write_fork_outputs([], {
            "source_split": "validation", "test_evaluation_allowed": False,
            "diagnostic_only": True,
        }, output)
