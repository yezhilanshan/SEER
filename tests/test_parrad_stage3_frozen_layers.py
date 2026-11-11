"""Frozen layer instrumentation must observe, never change, Event Former updates."""

from __future__ import annotations

import pytest
import torch
import numpy as np

from continuous_mmwave.config import ModelConfig
from continuous_mmwave.models.event_former import Qwen2Adapter, StreamingSemanticEventFormer, TinyLLMAdapter


def _config() -> ModelConfig:
    return ModelConfig(
        radar_dim=16, llm_dim=32, llm_layers=4, llm_heads=4,
        cross_attn_dim=8, cross_attn_heads=2, semantic_dim=8,
        dropout=0.0, max_commit_radar_tokens=16,
    )


@torch.inference_mode()
def _assert_trace(former: StreamingSemanticEventFormer, cache: object | None) -> None:
    former.eval()
    previous = former.initial_state(1).clone()
    radar = torch.randn(1, 4, 16)
    captured: list[dict[str, torch.Tensor]] = []
    with torch.inference_mode():
        ordinary = former.stream_update(previous, radar, 0.5, prompt_cache=cache)
        traced = former.stream_update(previous, radar, 0.5, prompt_cache=cache, trace=captured.append)
    assert len(captured) == 1
    values = captured[0]
    assert set(values) == {"H4", "R", "E_hat", "alpha", "E", "E_prev"}
    runtime_previous = previous.to(former.adapter.compute_dtype)
    torch.testing.assert_close(values["E_prev"], runtime_previous)
    expected_r = values["H4"].to(former.output_projection.weight.dtype) - runtime_previous.to(former.output_projection.weight.dtype)
    torch.testing.assert_close(values["R"], expected_r)
    with torch.inference_mode():
        torch.testing.assert_close(values["E_hat"], former.output_projection(values["R"]))
    expected_e = torch.lerp(runtime_previous.to(values["E_hat"].dtype), values["E_hat"], values["alpha"])
    torch.testing.assert_close(values["E"], expected_e.to(traced.event_tokens.dtype))
    torch.testing.assert_close(values["E"], traced.event_tokens)
    torch.testing.assert_close(ordinary.event_tokens, traced.event_tokens)
    torch.testing.assert_close(ordinary.gates, traced.gates)


def test_frozen_trace_observes_exact_uncached_update() -> None:
    torch.manual_seed(41)
    config = _config()
    former = StreamingSemanticEventFormer(TinyLLMAdapter(64, 32, 4, 4), config, torch.tensor([1, 2, 3]))
    _assert_trace(former, None)


def test_frozen_trace_observes_exact_cached_update() -> None:
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    config = transformers.Qwen2Config(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        head_dim=8, max_position_embeddings=64,
    )
    adapter = Qwen2Adapter(transformers.Qwen2ForCausalLM(config).to(torch.bfloat16), 4, 4)
    former = StreamingSemanticEventFormer(adapter, _config(), torch.tensor([1, 2, 3]))
    _assert_trace(former, former.build_prompt_cache(1))


@pytest.mark.parametrize(
    ("labels", "reference", "current", "expected"),
    [
        ([[1, 1], [1, 1], [1, 1]], 0, 2, False),
        ([[1, 1], [1, 1], [2, 2]], 0, 2, True),
        ([[1, 1], [2, 2], [1, 1]], 0, 2, None),
        ([[1, 1], [-100, -100], [2, 2]], 0, 2, None),
        ([[1, 1], [1, 2], [2, 2]], 0, 2, None),
        ([[-100, -100], [2, 2]], 0, 1, None),
    ],
)
def test_clean_update_label_rejects_ambiguous_runs(labels, reference, current, expected) -> None:
    from continuous_mmwave.parrad_stage3_frozen_layers import clean_update_change_label

    assert clean_update_change_label(np.asarray(labels), reference, current) is expected


def test_reference_binds_actual_frozen_active_state_not_prior_update() -> None:
    from continuous_mmwave.parrad_stage3_frozen_layers import bind_active_reference

    layers = {name: torch.full((1, 2), 2.0) for name in ("F", "H4", "R", "E_hat", "E", "z")}
    bound = bind_active_reference(
        candidate_id="rec:10", proposal_update=10, active_ref_update=7,
        active_ref_state_id="active:7", trace_state_id="active:7", layers=layers,
        frozen_event=layers["E"].clone(), frozen_key=layers["z"].clone(),
    )
    assert bound.active_ref_update == 7
    assert bound.active_ref_state_id == "active:7"
    with pytest.raises(ValueError, match="state ID"):
        bind_active_reference(
            candidate_id="rec:10", proposal_update=10, active_ref_update=7,
            active_ref_state_id="active:7", trace_state_id="active:9", layers=layers,
            frozen_event=layers["E"], frozen_key=layers["z"],
        )
    with pytest.raises(ValueError, match="frozen E"):
        bind_active_reference(
            candidate_id="rec:10", proposal_update=10, active_ref_update=7,
            active_ref_state_id="active:7", trace_state_id="active:7", layers=layers,
            frozen_event=layers["E"] + 0.01, frozen_key=layers["z"],
        )


def test_trace_context_observes_controller_style_calls_and_restores_method() -> None:
    from continuous_mmwave.parrad_stage3_frozen_layers import capture_stream_updates

    config = _config()
    former = StreamingSemanticEventFormer(TinyLLMAdapter(64, 32, 4, 4), config, torch.tensor([1, 2, 3]))
    former.eval()
    original = former.stream_update.__func__
    with torch.inference_mode(), capture_stream_updates(former) as calls:
        output = former.stream_update(former.initial_state(1), torch.randn(1, 4, 16), 0.5, None, None)
        assert len(calls) == 1
        torch.testing.assert_close(calls[0]["E"], output.event_tokens)
    assert former.stream_update.__func__ is original


def _tiny_train_recording(split="train"):
    from types import SimpleNamespace
    from test_parrad_controller import _controller
    from continuous_mmwave.parrad_stage3_data import Stage3CacheItem, Stage3CacheRecord

    controller = _controller(reject_timeout_s=0.18)
    controller.event_former = StreamingSemanticEventFormer(
        TinyLLMAdapter(64, 32, 4, 4), _config(), torch.tensor([1, 2, 3])
    ).eval()
    values = np.asarray([1, 10, 2, 1, 10, 2, 1], dtype=np.float32)
    record = Stage3CacheRecord(
        radar_tokens=np.broadcast_to(values[:, None, None], (7, 4, 16)).copy(),
        radar_mask=np.ones((7, 4), dtype=bool),
        frame_labels=np.asarray([[1, 1], [1, 1], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]),
        timestamps=np.arange(14).reshape(7, 2) * 0.09,
        oracle_keys=np.zeros((7, 32)), oracle_maturity=np.zeros(7), oracle_run_id=np.zeros(7),
    )
    return SimpleNamespace(controller=controller), Stage3CacheItem(
        split, "subject_02_set000_sensor", {"subject_id": "subject_02", "session_id": "set000", "sensor_id": "sensor"}, record
    )


def test_extraction_profiles_keep_sparse_default_and_continuous_ages():
    from continuous_mmwave.parrad_stage3_frozen_layers import extraction_ages
    assert extraction_ages("sparse") == (1, 4, 7, 13)
    assert extraction_ages("continuous") == tuple(range(1, 14))
    with pytest.raises(ValueError):
        extraction_ages("unknown")


@pytest.mark.parametrize("decision", ["confirm", "reject"])
def test_continuous_replay_retains_terminal_step_without_changing_controller(decision):
    from continuous_mmwave.parrad_stage3_frozen_layers import extract_recording, extraction_ages
    from test_parrad_controller import _ConstantDifference, _ConstantVerify
    results = []
    for profile in ("sparse", "continuous"):
        torch.manual_seed(731)
        runtime, item = _tiny_train_recording()
        if decision == "confirm":
            runtime.controller.semantic_difference = _ConstantDifference(10.0)
            runtime.controller.verify = _ConstantVerify(10.0)
        results.append(extract_recording(runtime, item, ages=extraction_ages(profile)))
    sparse, continuous = results
    assert sparse["candidates"] == continuous["candidates"]
    assert sparse["integrity"] == continuous["integrity"]
    index = {(r["candidate_id"], r["cohort"], r["age_updates"]): i for i, r in enumerate(continuous["rows"])}
    for i, row in enumerate(sparse["rows"]):
        j = index[row["candidate_id"], row["cohort"], row["age_updates"]]
        assert row == continuous["rows"][j]
        for name, array in sparse["arrays"].items():
            np.testing.assert_array_equal(array[i], continuous["arrays"][name][j])
    for candidate in continuous["candidates"]:
        if candidate["terminal_age"] is None:
            continue
        rows = [r for r in continuous["rows"] if r["cohort"] == "natural" and r["candidate_id"] == candidate["candidate_id"]]
        assert rows[-1]["age_updates"] == candidate["terminal_age"]
        assert rows[-1]["decision"] == decision


def test_train_replay_separates_forced_and_natural_and_maps_rejected_active() -> None:
    from continuous_mmwave.parrad_stage3_frozen_layers import extract_recording

    runtime, item = _tiny_train_recording()
    result = extract_recording(runtime, item, ages=(1, 2, 3))
    assert result["integrity"]["max_event_error"] == 0.0
    assert result["integrity"]["max_key_error"] == 0.0
    assert len(result["candidates"]) == 2
    first, second = result["candidates"]
    assert first["active_ref_update"] == 0
    assert second["active_ref_update"] == 3
    assert first["terminal_decision"] == "reject"
    assert {row["cohort"] for row in result["rows"]} == {"natural", "forced"}
    assert len({(row["candidate_id"], row["cohort"], row["age_updates"]) for row in result["rows"]}) == len(result["rows"])
    for layer in ("F", "H4", "R", "E_hat", "E", "z"):
        assert result["arrays"][f"current_{layer}"].shape[0] == len(result["rows"])


def test_replay_refuses_validation_and_existing_output(tmp_path) -> None:
    from continuous_mmwave.parrad_stage3_frozen_layers import extract_recording, write_recording

    runtime, item = _tiny_train_recording("validation")
    with pytest.raises(ValueError, match="train"):
        extract_recording(runtime, item)
    runtime, item = _tiny_train_recording()
    result = extract_recording(runtime, item, ages=(1, 2, 3))
    output = tmp_path / "recording"
    write_recording(result, output)
    before = (output / "rows.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_recording(result, output)
    assert (output / "rows.json").read_bytes() == before


def test_forced_candidates_can_overlap_without_changing_natural_decisions() -> None:
    from dataclasses import replace
    from continuous_mmwave.parrad_stage3_frozen_layers import extract_recording

    runtime, item = _tiny_train_recording()
    record = item.record
    doubled = replace(record, radar_tokens=np.concatenate([record.radar_tokens] * 2),
                      radar_mask=np.concatenate([record.radar_mask] * 2),
                      frame_labels=np.concatenate([record.frame_labels] * 2),
                      timestamps=np.arange(28).reshape(14, 2) * 0.09)
    result = extract_recording(runtime, replace(item, record=doubled), ages=(1, 4, 7))
    assert result["integrity"]["natural_updates"] == 14
    assert result["integrity"]["natural_decisions"]["proposal"] == 4
    assert result["integrity"]["natural_decisions"]["reject"] == 4
    assert any(row["cohort"] == "forced" and row["age_updates"] == 7 for row in result["rows"])


def test_gate_and_previous_state_roundtrip(tmp_path) -> None:
    from continuous_mmwave.parrad_stage3_frozen_layers import extract_recording, write_recording

    runtime, item = _tiny_train_recording()
    result = extract_recording(runtime, item, ages=(1, 2, 3))
    write_recording(result, tmp_path / "recording")
    with np.load(tmp_path / "recording" / "layers.npz") as arrays:
        for side in ("current", "reference"):
            alpha = arrays[f"{side}_alpha"]
            previous = arrays[f"{side}_E_prev"]
            delta = arrays[f"{side}_gate_delta"]
            assert alpha.shape[0] == len(result["rows"])
            assert previous.shape == delta.shape == arrays[f"{side}_E"].shape
            expanded_alpha = np.repeat(alpha, previous.shape[1] // alpha.shape[1], axis=1)
            np.testing.assert_allclose(previous + expanded_alpha * delta, arrays[f"{side}_E"], atol=1e-6)


@pytest.mark.parametrize("decision,discarded", [("reject", 0), ("confirm", 1)])
def test_terminal_discarded_branch_mismatch_aborts(monkeypatch, decision, discarded) -> None:
    from contextlib import contextmanager
    import continuous_mmwave.parrad_stage3_frozen_layers as audit
    from test_parrad_controller import _ConstantDifference, _ConstantVerify

    runtime, item = _tiny_train_recording()
    ctrl = runtime.controller
    if decision == "confirm":
        ctrl.semantic_difference = _ConstantDifference(10.0)
        ctrl.verify = _ConstantVerify(10.0)
    original = audit.capture_stream_updates

    @contextmanager
    def corrupted(former):
        with original(former) as calls:
            yield calls
        if ctrl.candidate is None and len(calls) == 2:
            calls[discarded]["E"] = calls[discarded]["E"] + 0.02

    monkeypatch.setattr(audit, "capture_stream_updates", corrupted)
    for index in (0, 1):
        audit._observed_step(ctrl, torch.from_numpy(item.record.radar_tokens[index]).unsqueeze(0),
                             torch.from_numpy(item.record.radar_mask[index]).unsqueeze(0), index * 0.18, item.metadata)
    assert ctrl.candidate is not None
    with pytest.raises(RuntimeError, match="trace/controller mismatch"):
        audit._observed_step(ctrl, torch.from_numpy(item.record.radar_tokens[2]).unsqueeze(0),
                             torch.from_numpy(item.record.radar_mask[2]).unsqueeze(0), 0.36, item.metadata)
    assert "_clear_candidate" not in vars(ctrl)


def _approved_records():
    return [{"key": f"{subject}__{session}__{sensor}", "metadata": {
        "subject_id": subject, "session_id": session, "sensor_id": sensor,
    }} for subject in ("subject_02", "subject_03", "subject_04")
       for session in ("set000", "set001", "set002", "set003")
       for sensor in ("TI_xWR14xx", "TI_xWR68xx")]


@pytest.mark.parametrize("mutation", ["subject", "session", "duplicate", "missing"])
def test_cohort_preflight_rejects_before_array_loading(tmp_path, monkeypatch, mutation) -> None:
    import json
    import continuous_mmwave.parrad_stage3_runner as runner
    from continuous_mmwave.parrad_stage3_frozen_layers import load_train_audit_items

    rows = _approved_records()
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[-1] = rows[0]
    else:
        rows[0]["metadata"][f"{mutation}_id"] = "subject_05" if mutation == "subject" else "set999"
    (tmp_path / "train").mkdir()
    for row in rows:
        (tmp_path / "train" / (row["key"] + ".json")).write_text(json.dumps({"metadata": row["metadata"]}))
    (tmp_path / "index.json").write_text(json.dumps({"records": [{"split": "train", "key": row["key"]} for row in rows]}))
    def forbidden_loader(*args, **kwargs):
        pytest.fail("recording arrays loaded before cohort preflight")
    monkeypatch.setattr(runner, "_load_cache_items", forbidden_loader)
    with pytest.raises(ValueError, match="cohort"):
        load_train_audit_items(tmp_path, tmp_path / "checkpoint.pt")


def test_cohort_requires_exact_unique_subject_session_sensor_matrix() -> None:
    from continuous_mmwave.parrad_stage3_frozen_layers import validate_record_cohort

    records = _approved_records()
    validate_record_cohort(records)
    with pytest.raises(ValueError, match="cohort"):
        validate_record_cohort(records[:-1] + [records[0]])
