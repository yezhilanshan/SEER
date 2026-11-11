from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test_parrad_controller import _controller, _mask, _tokens


def test_shadow_features_use_proposal_residual_and_causal_slope() -> None:
    probe = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    active = torch.tensor([[1.0, 0.0]])
    initial_candidate = torch.tensor([[1.0, 0.0]])
    initial_shadow = torch.tensor([[0.0, 1.0]])
    candidate = torch.tensor([[0.0, 1.0]])
    shadow = torch.tensor([[1.0, 0.0]])

    initial = probe.candidate_shadow_features(
        active, initial_candidate, initial_shadow,
        initial_candidate, initial_shadow, previous=None, delta_s=None,
    )
    current = probe.candidate_shadow_features(
        active, candidate, shadow,
        initial_candidate, initial_shadow, previous=initial, delta_s=0.18,
    )

    assert initial["candidate_shadow_residual_change"] == pytest.approx(0.0)
    assert current["d_active_candidate"] == pytest.approx(0.5)
    assert current["d_active_shadow"] == pytest.approx(0.0)
    assert current["d_candidate_shadow"] == pytest.approx(0.5)
    assert current["candidate_vs_active_drift"] == pytest.approx(0.5)
    assert current["candidate_shadow_residual_change"] == pytest.approx(2**0.5 / 2)
    assert current["slope_d_active_candidate"] == pytest.approx(0.5 / 0.18)
    assert current["slope_d_active_shadow"] == pytest.approx(-0.5 / 0.18)
    assert initial["slope_d_active_candidate"] is None


def test_clean_label_excludes_later_and_multiple_boundaries() -> None:
    probe = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    label = probe.clean_proposal_label
    assert label([1.3], proposal_time=1.0, horizon_s=2.16, tolerance_s=0.72) is True
    assert label([], proposal_time=1.0, horizon_s=2.16, tolerance_s=0.72) is False
    assert label([2.0], proposal_time=1.0, horizon_s=2.16, tolerance_s=0.72) is None
    assert label([1.3, 2.0], proposal_time=1.0, horizon_s=2.16, tolerance_s=0.72) is None


@pytest.mark.parametrize(
    ("difference_logit", "verify_logit", "expected_natural_decision"),
    [(10.0, 10.0, "confirm"), (-10.0, -10.0, "reject")],
)
def test_forced_branch_survives_without_changing_natural_controller(
    difference_logit, verify_logit, expected_natural_decision
) -> None:
    probe = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    source = _controller(
        difference_logit=difference_logit,
        verify_logit=verify_logit,
        reject_timeout_s=0.18,
    )
    source.step_tokens(_tokens(1), _mask(), 0.0)
    assert source.step_tokens(_tokens(10), _mask(), 0.18).decision == "proposal"

    forced = probe.fork_candidate_controller(source, horizon_updates=12)
    first_forced = forced.step_tokens(_tokens(2), _mask(), 0.36)
    second_forced = forced.step_tokens(_tokens(2), _mask(), 0.54)

    assert first_forced.decision == "none"
    assert second_forced.decision == "none"
    assert forced.candidate is not None
    assert source.decision_trace == ["none", "proposal"]
    natural = source.step_tokens(_tokens(2), _mask(), 0.36)
    assert natural.decision == expected_natural_decision
    assert natural.difference_probability == pytest.approx(first_forced.difference_probability)
    assert natural.verify_probability == pytest.approx(first_forced.verify_probability)


def test_forced_probe_rejects_test_record_before_touching_runtime() -> None:
    probe = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    with pytest.raises(ValueError, match="validation"):
        probe.audit_forced_horizon(
            None, (SimpleNamespace(split="test"),), None, None, horizon_updates=12
        )


def test_forced_replay_extends_one_real_proposal_past_natural_reject(tmp_path) -> None:
    probe = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    values = [1.0, 10.0, 2.0, 2.0, 2.0, 2.0]
    times = [0.18 * index for index in range(len(values))]
    record = SimpleNamespace(
        radar_tokens=np.stack([np.full((4, 4), value, dtype=np.float32) for value in values]),
        radar_mask=np.ones((len(values), 4), dtype=np.bool_),
        timestamps=np.array(times, dtype=np.float64).reshape(-1, 1),
        frame_labels=np.array([0, 0, 1, 1, 1, 1], dtype=np.int64).reshape(-1, 1),
        oracle_run_id=np.array([0, 0, 1, 1, 1, 1], dtype=np.int64),
        oracle_keys=np.tile(np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), (len(values), 1)),
    )
    item = SimpleNamespace(split="validation", key="one", metadata={"sensor_id": "14xx"}, record=record)
    baseline = _controller(reject_timeout_s=0.18)
    expected_physical = []
    for index, (value, timestamp) in enumerate(zip(values, times, strict=True)):
        output = baseline.step_tokens(_tokens(value), _mask(), timestamp)
        expected_physical.append(dict(
            recording_key="one", update_index=index, decision=output.decision,
            proposal_probability=output.proposal_probability,
        ))
    assert [row["decision"] for row in expected_physical] == [
        "none", "proposal", "reject", "none", "none", "none"
    ]
    (tmp_path / "upstream-summary.json").write_text(json.dumps({
        "source_split": "validation", "test_evaluation_allowed": False,
    }), encoding="utf-8")
    (tmp_path / "physical-updates.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in expected_physical), encoding="utf-8"
    )
    (tmp_path / "semantic-candidates.jsonl").write_text(json.dumps({
        "candidate_id": "one:1", "update_index": 2,
        "online_probability": 0.0000453978687, "online_cosine_probability": 0.0,
    }) + "\n", encoding="utf-8")
    thresholds = SimpleNamespace(
        proposal=0.8, verify_confirm=0.8, verify_reject=0.2,
        candidate_timeout_updates=1, proposal_persistence=1,
    )

    rows, integrity = probe.audit_forced_horizon(
        SimpleNamespace(controller=_controller(reject_timeout_s=0.18)),
        (item,), thresholds, tmp_path, horizon_updates=4,
    )

    assert len(rows) == 4
    assert [row["update_index"] for row in rows] == [2, 3, 4, 5]
    assert rows[-1]["age_s"] == pytest.approx(0.72)
    assert all(row["candidate_id"] == "one:1" for row in rows)
    assert all(row["proposal_near_boundary_0p72s"] for row in rows)
    assert all(row["clean_proposal_target"] is True for row in rows)
    assert all(row["d_active_candidate"] == pytest.approx(row["online_cosine_probability"]) for row in rows)
    assert all(row["d_active_shadow"] is not None for row in rows)
    assert all(row["candidate_shadow_residual_change"] is not None for row in rows)
    assert all(row["slope_d_candidate_shadow"] is not None for row in rows)
    assert integrity["baseline_decisions_checked"] == len(values)
    assert integrity["baseline_semantic_scores_checked"] == 1
    assert integrity["forced_candidates"] == 1


def test_forced_summary_uses_same_candidates_and_exposes_natural_survival() -> None:
    probe = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    rows = [
        dict(candidate_id="a", sensor_id="14xx", age_updates=2, proposal_near_boundary_0p72s=True,
             current_run_switch=True, online_cosine_probability=0.2, online_head_probability=0.4,
             naturally_observed=True, online_reference_to_oracle_probability=0.1,
             online_candidate_to_oracle_probability=0.2),
        dict(candidate_id="b", sensor_id="14xx", age_updates=2, proposal_near_boundary_0p72s=False,
             current_run_switch=False, online_cosine_probability=0.8, online_head_probability=0.6,
             naturally_observed=True, online_reference_to_oracle_probability=0.2,
             online_candidate_to_oracle_probability=0.3),
        dict(candidate_id="a", sensor_id="14xx", age_updates=3, proposal_near_boundary_0p72s=True,
             current_run_switch=True, online_cosine_probability=0.9, online_head_probability=0.7,
             naturally_observed=False, online_reference_to_oracle_probability=0.1,
             online_candidate_to_oracle_probability=0.1),
        dict(candidate_id="b", sensor_id="14xx", age_updates=3, proposal_near_boundary_0p72s=False,
             current_run_switch=False, online_cosine_probability=0.1, online_head_probability=0.3,
             naturally_observed=True, online_reference_to_oracle_probability=0.2,
             online_candidate_to_oracle_probability=0.2),
    ]

    result = probe.summarize_forced_rows(rows, horizon_updates=2, update_period_s=0.18)

    assert result["by_age_updates"]["2"]["proposal_near_boundary"]["cosine"]["auroc"] == 0.0
    assert result["by_age_updates"]["3"]["proposal_near_boundary"]["cosine"]["auroc"] == 1.0
    assert result["by_age_updates"]["3"]["naturally_observed_candidates"] == 1
    assert result["matched_full_horizon"]["candidate_count"] == 2
    assert result["matched_full_horizon"]["by_age_updates"]["3"]["proposal_near_boundary"]["cosine"]["auroc"] == 1.0


def test_probe_rejects_baseline_from_a_different_checkpoint_or_cache() -> None:
    probe = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    baseline = {
        "source_split": "validation", "test_evaluation_allowed": False,
        "checkpoint_sha256": "checkpoint-a", "threshold_artifact_sha256": "threshold-a",
        "cache_fingerprint": "cache-a",
    }
    probe.validate_baseline_context(
        baseline, checkpoint_sha256="checkpoint-a",
        threshold_sha256="threshold-a", cache_fingerprint="cache-a",
    )
    with pytest.raises(ValueError, match="checkpoint"):
        probe.validate_baseline_context(
            baseline, checkpoint_sha256="checkpoint-b",
            threshold_sha256="threshold-a", cache_fingerprint="cache-a",
        )
    with pytest.raises(ValueError, match="cache"):
        probe.validate_baseline_context(
            baseline, checkpoint_sha256="checkpoint-a",
            threshold_sha256="threshold-a", cache_fingerprint="cache-b",
        )


def test_shadow_summary_reports_matched_clean_auc_and_recording_counts() -> None:
    probe = importlib.import_module("continuous_mmwave.parrad_stage3_forced_horizon_probe")
    rows = []
    for candidate_id, recording_key, sensor, clean, score in (
        ("a", "rec-1", "14xx", True, 0.9),
        ("b", "rec-1", "14xx", False, 0.1),
        ("c", "rec-2", "68xx", None, 0.95),
    ):
        for age in (2, 3):
            rows.append(dict(
                candidate_id=candidate_id, recording_key=recording_key, sensor_id=sensor,
                age_updates=age, clean_proposal_target=clean,
                proposal_near_boundary_0p72s=(clean is True), current_run_switch=False,
                online_cosine_probability=score, online_head_probability=score,
                d_active_candidate=score, d_active_shadow=0.2, d_candidate_shadow=score,
                candidate_vs_active_drift=score - 0.2, candidate_shadow_residual_change=score,
                **{f"slope_{name}": score for name in probe.SHADOW_FEATURES},
                naturally_observed=True, online_reference_to_oracle_probability=0.1,
                online_candidate_to_oracle_probability=0.1,
            ))

    result = probe.summarize_forced_rows(rows, horizon_updates=2, update_period_s=0.18)
    age = result["matched_full_horizon"]["by_age_updates"]["3"]
    assert age["clean_proposal"]["d_active_candidate"]["auroc"] == 1.0
    assert age["clean_proposal"]["d_active_candidate"]["count"] == 2
    assert age["clean_proposal"]["candidate_shadow_residual_change"]["auroc"] == 1.0
    assert age["clean_proposal"]["by_recording"]["rec-1"]["count"] == 2
    assert age["clean_proposal"]["by_recording"]["rec-2"]["count"] == 0
    assert age["clean_proposal"]["by_sensor"]["68xx"]["count"] == 0
