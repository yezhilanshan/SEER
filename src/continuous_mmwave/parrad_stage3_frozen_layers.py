"""Strict, train-only contracts for frozen Stage P3 layer diagnostics.

This module does not modify Event Former weights or controller decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import argparse
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

import numpy as np
import torch
from torch import Tensor


LAYER_NAMES = ("F", "H4", "R", "E_hat", "E", "z")
TRACE_TOLERANCE = 1e-5
AGE_UPDATES = (1, 4, 7, 13)
AUDIT_SUBJECTS = ("subject_02", "subject_03", "subject_04")
AUDIT_SESSIONS = ("set000", "set001", "set002", "set003")
AUDIT_SENSORS = ("TI_xWR14xx", "TI_xWR68xx")
SEQUENCE_SETTINGS = {
    "version": 1, "update_period_s": 0.18, "ages": list(range(1, 14)),
    "natural_fixed": {"4": [1, 3], "6": [1, 3], "7": [1, 3], "8": [1, 3, 8]},
    "natural_terminal": {"3": [1, 3], "8": [1, 3, 8]},
    "forced_fixed": {"4": [1, 3], "6": [1, 3], "7": [1, 3], "8": [1, 3, 8], "13": [1, 3, 8]},
    "sources": [["F"], ["E"], ["z"], ["F", "E"], ["F", "z"]],
    "tasks": ["update", "proposal"], "source_scale": "train-fold RMS relational norm; floor=1e-6",
    "history_order": "oldest first; sources interleaved in declared order",
    "geometry": "cosine and L2 of full scaled history against repeated frozen reference",
}


def extraction_ages(profile: str) -> tuple[int, ...]:
    if profile == "sparse":
        return AGE_UPDATES
    if profile == "continuous":
        return tuple(range(1, 14))
    raise ValueError(f"unknown trace profile: {profile}")


def validate_record_cohort(records: list[dict[str, Any]]) -> None:
    expected = {(subject, session, sensor) for subject in AUDIT_SUBJECTS
                for session in AUDIT_SESSIONS for sensor in AUDIT_SENSORS}
    identities = []
    for row in records:
        metadata = row.get("metadata", {})
        identity = tuple(metadata.get(name) for name in ("subject_id", "session_id", "sensor_id"))
        if identity not in expected or row.get("key") != "__".join(identity):
            raise ValueError("unapproved train recording cohort identity")
        identities.append(identity)
    if len(identities) != 24 or set(identities) != expected:
        raise ValueError("cohort must be the exact unique 24-record train matrix")


def load_train_audit_items(cache_dir: Path, checkpoint: Path, recording_key: str | None = None):
    """Check train sidecars before any recording array is opened."""
    from continuous_mmwave.parrad_stage3_runner import _load_cache_items

    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    records = []
    for row in index.get("records", []):
        if row.get("split") != "train":
            continue
        key = str(row["key"])
        if Path(key).name != key or "/" in key or "\\" in key:
            raise ValueError("unsafe cohort recording key")
        sidecar = json.loads((cache_dir / "train" / f"{key}.json").read_text(encoding="utf-8"))
        records.append({"key": key, "metadata": sidecar.get("metadata", {})})
    validate_record_cohort(records)
    if recording_key is not None and recording_key not in {row["key"] for row in records}:
        raise ValueError("smoke recording outside approved cohort")
    return _load_cache_items(cache_dir, checkpoint, splits=("train",),
                             recording_keys=None if recording_key is None else (recording_key,))


@contextmanager
def capture_stream_updates(event_former: Any) -> Iterator[list[dict[str, Tensor]]]:
    """Observe controller calls through its unchanged stream-update signature.

    Only this diagnostic runtime instance is wrapped, and restoration is
    guaranteed even when the controller raises.
    """
    if "stream_update" in vars(event_former):
        raise RuntimeError("stream_update is already instrumented")
    original = event_former.stream_update
    calls: list[dict[str, Tensor]] = []

    def observed(*args: Any, **kwargs: Any) -> Any:
        if "trace" in kwargs:
            raise RuntimeError("nested Event Former trace is unsupported")
        return original(*args, **kwargs, trace=calls.append)

    event_former.stream_update = observed
    try:
        yield calls
    finally:
        del event_former.stream_update


def clean_update_change_label(frame_labels: np.ndarray, reference_update: int, current_update: int) -> bool | None:
    """Whether exactly one clean GT run transition occurred since a reference.

    None excludes ignored/mixed updates, ignored gaps, and multiple boundaries.
    The cached oracle_run_id is deliberately not consulted.
    """
    labels = np.asarray(frame_labels)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError("frame_labels must have shape [updates, 2]")
    if not 0 <= reference_update <= current_update < len(labels):
        raise ValueError("invalid reference/current update range")
    segment = labels[reference_update : current_update + 1]
    if np.any(segment < 0) or np.any(segment[:, 0] != segment[:, 1]):
        return None
    boundaries = int(np.count_nonzero(segment[1:, 0] != segment[:-1, 0]))
    if boundaries > 1:
        return None
    return bool(boundaries)


@dataclass(frozen=True)
class ActiveReferenceTrace:
    candidate_id: str
    proposal_update: int
    active_ref_update: int
    active_ref_state_id: str
    layers: Mapping[str, Tensor]
    gate: Tensor | None = None
    previous: Tensor | None = None


def bind_active_reference(
    *, candidate_id: str, proposal_update: int, active_ref_update: int,
    active_ref_state_id: str, trace_state_id: str, layers: Mapping[str, Tensor],
    frozen_event: Tensor, frozen_key: Tensor,
    gate: Tensor | None = None, previous: Tensor | None = None,
) -> ActiveReferenceTrace:
    """Bind a proposal to the actual frozen Active state, or fail closed."""
    if not candidate_id or not 0 <= active_ref_update < proposal_update:
        raise ValueError("invalid candidate/reference update identity")
    if not active_ref_state_id or active_ref_state_id != trace_state_id:
        raise ValueError("Active reference state ID mismatch")
    if set(layers) != set(LAYER_NAMES):
        raise ValueError("reference trace must contain every frozen layer")
    for name, expected in (("E", frozen_event), ("z", frozen_key)):
        actual = layers[name]
        if actual.shape != expected.shape or not bool(torch.isfinite(actual).all()) or not bool(torch.isfinite(expected).all()):
            raise ValueError(f"invalid frozen {name} shape or value")
        error = float((actual.float() - expected.float()).abs().max())
        if error > TRACE_TOLERANCE:
            raise ValueError(f"frozen {name} differs from controller by {error}")
    if any(not bool(torch.isfinite(value).all()) for value in layers.values()):
        raise ValueError("nonfinite frozen layer trace")
    return ActiveReferenceTrace(
        candidate_id, proposal_update, active_ref_update, active_ref_state_id,
        {name: value.detach().clone() for name, value in layers.items()},
        None if gate is None else gate.detach().clone(),
        None if previous is None else previous.detach().clone(),
    )


def _update_label_reason(labels: np.ndarray, reference: int, current: int) -> str | None:
    segment = labels[reference : current + 1]
    if np.any(segment < 0):
        return "ignored_frame_or_gap"
    if np.any(segment[:, 0] != segment[:, 1]):
        return "mixed_update"
    if np.count_nonzero(segment[1:, 0] != segment[:-1, 0]) > 1:
        return "multiple_boundaries"
    return None


@torch.inference_mode()
def _observed_step(controller: Any, tokens: Tensor, mask: Tensor, timestamp: float, metadata: Any):
    projections: list[tuple[Tensor, Tensor]] = []

    def projection_hook(_module: Any, inputs: tuple[Tensor, ...], output: Tensor) -> None:
        projections.append((inputs[0].detach().clone(), output.detach().clone()))

    handle = controller.semantic_projection.register_forward_hook(projection_hook)
    terminal_branches: list[tuple[int, Tensor, Tensor]] = []
    if "_clear_candidate" in vars(controller):
        handle.remove()
        raise RuntimeError("candidate clearing is already instrumented")
    original_clear = controller._clear_candidate

    def observed_clear() -> None:
        # Capture both arms before the controller discards either one.
        if controller.candidate is not None and controller.shadow is not None:
            terminal_branches.extend([
                (0, controller.candidate.event_tokens.detach().clone(), controller.candidate.key_history[-1].detach().clone()),
                (1, controller.shadow.event_tokens.detach().clone(), controller.semantic_projection(controller.shadow.event_tokens).detach().clone()),
            ])
        original_clear()

    controller._clear_candidate = observed_clear
    try:
        with capture_stream_updates(controller.event_former) as calls:
            output = controller.step_tokens(tokens, mask, timestamp, metadata)
    finally:
        del controller._clear_candidate
        handle.remove()
    if len(calls) not in (1, 2):
        raise RuntimeError("unexpected Event Former call count")
    traces = []
    for call in calls:
        matching_keys = [key for event, key in projections if torch.equal(event, call["E"])]
        key = matching_keys[-1] if matching_keys else controller.semantic_projection(call["E"])
        trace = {name: call[name] for name in ("H4", "R", "E_hat", "E")}
        trace.update({"F": tokens.detach().clone(), "z": key.detach().clone()})
        if any(not bool(torch.isfinite(value).all()) for value in (*trace.values(), call["alpha"], call["E_prev"])):
            raise RuntimeError("nonfinite frozen layer trace")
        traces.append((trace, call["alpha"], call["E_prev"]))
    checks: list[tuple[int, Tensor, Tensor]] = []
    if len(calls) == 1:
        if controller.active is None:
            raise RuntimeError("one-call controller step has no Active state")
        checks.append((0, controller.active.event_tokens, controller.semantic_projection(controller.active.event_tokens)))
    elif controller.candidate is not None:
        checks.extend([
            (0, controller.candidate.event_tokens, controller.candidate.key_history[-1]),
            (1, controller.shadow.event_tokens, controller.semantic_projection(controller.shadow.event_tokens)),
        ])
    elif output.decision in ("confirm", "reject"):
        if len(terminal_branches) != 2:
            raise RuntimeError("terminal step did not preserve both branch snapshots")
        checks.extend(terminal_branches)
        arm = 0 if output.decision == "confirm" else 1
        checks.append((arm, controller.active.event_tokens, controller.semantic_projection(controller.active.event_tokens)))
    else:
        raise RuntimeError("two-call step has no live or committed Candidate/Shadow")
    max_e = max_z = 0.0
    for arm, event, key in checks:
        max_e = max(max_e, float((traces[arm][0]["E"].float() - event.float()).abs().max()))
        max_z = max(max_z, float((traces[arm][0]["z"].float() - key.float()).abs().max()))
    if max(max_e, max_z) > TRACE_TOLERANCE:
        raise RuntimeError(f"trace/controller mismatch: E={max_e}, z={max_z}")
    return output, traces, max_e, max_z


@torch.inference_mode()
def extract_recording(runtime: Any, item: Any, *, ages: tuple[int, ...] = AGE_UPDATES, step_delay_s: float = 0.0, thermal_guard: Callable[[], None] | None = None) -> dict[str, Any]:
    """Replay one train recording with exact natural reference tracking."""
    from continuous_mmwave.parrad_stage3_candidate_audit import _truth
    from continuous_mmwave.parrad_stage3_forced_horizon_probe import clean_proposal_label, fork_candidate_controller

    if item.split != "train":
        raise ValueError("frozen layer extraction accepts train records only")
    if not ages or ages[0] != 1 or ages[-1] < 2 or tuple(sorted(set(ages))) != ages or step_delay_s < 0:
        raise ValueError("ages must be unique, increasing, and start at 1")
    controller = runtime.controller
    controller.reset()
    parameter = next(controller.event_former.parameters())
    device = parameter.device
    record = item.record
    boundaries, _ = _truth(item)
    horizon = ages[-1]
    rows: list[dict[str, Any]] = []
    arrays: dict[str, list[np.ndarray]] = {f"{side}_{layer}": [] for side in ("current", "reference") for layer in LAYER_NAMES}
    arrays.update({f"{side}_{name}": [] for side in ("current", "reference") for name in ("alpha", "E_prev", "gate_delta")})
    candidates: list[dict[str, Any]] = []
    forced: list[dict[str, Any]] = []
    active_trace: tuple[int, str, Mapping[str, Tensor], Any, Tensor, Tensor] | None = None
    live: dict[str, Any] | None = None
    max_event_error = max_key_error = 0.0
    checked_steps = 0

    def observe(ctrl: Any, index: int):
        nonlocal max_event_error, max_key_error, checked_steps
        tokens = torch.from_numpy(record.radar_tokens[index].astype("float32")).unsqueeze(0).to(device)
        mask = torch.from_numpy(record.radar_mask[index].astype("bool")).unsqueeze(0).to(device)
        output, traces, error_e, error_z = _observed_step(
            ctrl, tokens, mask, float(record.timestamps[index, 0]), item.metadata,
        )
        max_event_error = max(max_event_error, error_e)
        max_key_error = max(max_key_error, error_z)
        checked_steps += 1
        return output, traces

    def append_row(candidate: dict[str, Any], cohort: str, index: int, trace: Mapping[str, Tensor], gate: Tensor, previous: Tensor, decision: str):
        reference: ActiveReferenceTrace = candidate["reference"]
        age = index - reference.proposal_update + 1
        if age not in ages:
            return
        target = clean_update_change_label(record.frame_labels, reference.active_ref_update, index)
        row = {
            "source_split": "train", "recording_key": item.key, **dict(item.metadata),
            "candidate_id": reference.candidate_id, "cohort": cohort, "age_updates": age,
            "age_seconds": (age - 1) * controller.update_period_s,
            "proposal_update": reference.proposal_update, "active_ref_update": reference.active_ref_update,
            "active_ref_state_id": reference.active_ref_state_id, "current_update": index,
            "proposal_target": candidate["info"]["proposal_target"],
            "proposal_exclusion": candidate["info"]["proposal_exclusion"],
            "update_target": target, "update_exclusion": _update_label_reason(record.frame_labels, reference.active_ref_update, index),
            "full_horizon": candidate["info"]["full_horizon"], "decision": decision,
            "gate_mean": float(gate.float().mean()), "gate_min": float(gate.min()), "gate_max": float(gate.max()),
            "layer_dtypes": {name: str(trace[name].dtype) for name in LAYER_NAMES},
            "layer_norms": {name: float(torch.linalg.vector_norm(trace[name].float())) for name in LAYER_NAMES},
            "reference_layer_norms": {name: float(torch.linalg.vector_norm(reference.layers[name].float())) for name in LAYER_NAMES},
        }
        rows.append(row)
        for name in LAYER_NAMES:
            arrays[f"current_{name}"].append(trace[name].detach().float().cpu().reshape(-1).numpy().copy())
            arrays[f"reference_{name}"].append(reference.layers[name].detach().float().cpu().reshape(-1).numpy().copy())
        if reference.gate is None or reference.previous is None:
            raise RuntimeError("missing frozen reference gate/previous state")
        for side, alpha, prev, proposal in (("current", gate, previous, trace["E_hat"]),
                                            ("reference", reference.gate, reference.previous, reference.layers["E_hat"])):
            for name, value in (("alpha", alpha), ("E_prev", prev), ("gate_delta", proposal - prev.to(proposal.dtype))):
                arrays[f"{side}_{name}"].append(value.detach().float().cpu().reshape(-1).numpy().copy())

    for index in range(len(record.radar_tokens)):
        if thermal_guard is not None and index % 100 == 0:
            thermal_guard()
        if active_trace is not None and active_trace[3] is not controller.active:
            raise RuntimeError("tracked Active state object differs from controller")
        pending = list(forced)
        had_candidate = controller.candidate is not None
        output, traces = observe(controller, index)
        if output.decision == "proposal":
            if active_trace is None or controller.reference is None:
                raise RuntimeError("Proposal has no known frozen Active trace")
            ref_update, state_id, layers, owner, ref_gate, ref_previous = active_trace
            if owner is not controller.active:
                raise RuntimeError("Proposal froze a different Active state")
            reference = bind_active_reference(
                candidate_id=f"{item.key}:{index}", proposal_update=index, active_ref_update=ref_update,
                active_ref_state_id=state_id, trace_state_id=state_id, layers=layers,
                frozen_event=controller.reference.event_tokens, frozen_key=controller.reference.semantic_key,
                gate=ref_gate, previous=ref_previous,
            )
            full = index + horizon - 1 < len(record.radar_tokens)
            proposal_target = clean_proposal_label(
                boundaries, proposal_time=float(record.timestamps[index, 0]),
                horizon_s=(horizon - 1) * controller.update_period_s, tolerance_s=0.72,
            ) if full else None
            info = {
                "candidate_id": reference.candidate_id, "proposal_update": index,
                "active_ref_update": ref_update, "active_ref_state_id": state_id,
                "proposal_target": proposal_target, "proposal_exclusion": (
                    "recording_end" if not full else "ambiguous_boundaries" if proposal_target is None else None
                ), "full_horizon": full, "terminal_age": None, "terminal_decision": "recording_end",
            }
            candidates.append(info)
            live = {"reference": reference, "info": info}
            probe = fork_candidate_controller(controller, horizon_updates=horizon - 1)
            forced.append({**live, "controller": probe})
            append_row(live, "natural", index, *traces[0], output.decision)
            append_row(live, "forced", index, *traces[0], "proposal")
        elif had_candidate:
            if live is None:
                raise RuntimeError("live Candidate lost its frozen reference")
            append_row(live, "natural", index, *traces[0], output.decision)
            if output.decision in ("confirm", "reject"):
                live["info"]["terminal_age"] = index - live["reference"].proposal_update + 1
                live["info"]["terminal_decision"] = output.decision
                arm = 0 if output.decision == "confirm" else 1
                active_trace = (index, f"{item.key}:{index}:{output.decision}", traces[arm][0], controller.active, *traces[arm][1:])
                live = None
        else:
            active_trace = (index, f"{item.key}:{index}:active", traces[0][0], controller.active, *traces[0][1:])
        for candidate in pending:
            forced_output, forced_traces = observe(candidate["controller"], index)
            if forced_output.decision in ("confirm", "reject", "proposal"):
                raise RuntimeError("forced horizon terminated or forked unexpectedly")
            append_row(candidate, "forced", index, *forced_traces[0], forced_output.decision)
            if index - candidate["reference"].proposal_update + 1 >= horizon:
                forced = [value for value in forced if value is not candidate]
        if step_delay_s:
            time.sleep(step_delay_s)
        if index % 500 == 0:
            print(json.dumps({"recording": item.key, "update": index, "total": len(record.radar_tokens), "proposals": len(candidates)}), flush=True)
    return {
        "rows": rows, "candidates": candidates,
        "arrays": {name: np.stack(values) if values else np.empty((0, 0), np.float32) for name, values in arrays.items()},
        "integrity": {"max_event_error": max_event_error, "max_key_error": max_key_error, "checked_steps": checked_steps,
                      "natural_updates": len(record.radar_tokens), "natural_decisions": dict(Counter(controller.decision_trace))},
    }


def write_recording(result: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(output_dir / "layers.npz", **result["arrays"])
    for name in ("rows", "candidates", "integrity"):
        (output_dir / f"{name}.json").write_text(json.dumps(result[name], ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    from continuous_mmwave.parrad_stage3_calibrate import build_runtime
    from continuous_mmwave.parrad_stage3_hard_negative import _apply_thresholds, _file_sha256, _load_thresholds, _load_yaml
    from continuous_mmwave.parrad_stage3_linear_probe import PROBE_SETTINGS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--selected-thresholds", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--recording-key")
    parser.add_argument("--step-delay-s", type=float, default=0.02)
    parser.add_argument("--trace-profile", choices=("sparse", "continuous"), default="sparse")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    raw = _load_yaml(args.config)
    stage3 = raw["stage3"]
    thresholds = _load_thresholds(args.selected_thresholds, args.checkpoint)
    records, fingerprint = load_train_audit_items(Path(stage3["cache_dir"]), Path(stage3["stage2_checkpoint"]), args.recording_key)
    items = records["train"]
    if args.recording_key:
        items = tuple(item for item in items if item.key == args.recording_key)
    if not items or any(item.split != "train" for item in items):
        raise ValueError("no matching train records")
    runtime = build_runtime(args.config, args.checkpoint, torch.device(args.device))
    _apply_thresholds(runtime, thresholds)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 2, "source_split": "train", "diagnostic_only": True, "test_evaluation_allowed": False,
        "gate_auxiliary_arrays": ["alpha", "E_prev", "gate_delta"],
        "status": "extracting", "checkpoint": str(args.checkpoint), "checkpoint_sha256": _file_sha256(args.checkpoint),
        "stage2_checkpoint_sha256": _file_sha256(Path(stage3["stage2_checkpoint"])), "cache_fingerprint": fingerprint,
        "threshold_artifact": str(args.selected_thresholds), "threshold_artifact_sha256": _file_sha256(args.selected_thresholds),
        "config_sha256": _file_sha256(args.config), "extractor_sha256": _file_sha256(Path(__file__)),
        "event_former_sha256": _file_sha256(Path(__file__).parent / "models" / "event_former.py"),
        "probe_code_sha256": _file_sha256(Path(__file__).with_name("parrad_stage3_linear_probe.py")),
        "ages": extraction_ages(args.trace_profile), "probe_settings": PROBE_SETTINGS, "torch_version": torch.__version__, "numpy_version": np.__version__,
        "records": [{"key": item.key, "metadata": dict(item.metadata), "updates": len(item.record.radar_tokens)} for item in items],
        "completed_records": [], "layer_storage_dtype": "float32", "trace_tolerance": TRACE_TOLERANCE,
        "step_delay_s": args.step_delay_s,
        "thermal_pause_at_c": 82, "thermal_resume_at_c": 78,
        "layer_dimensions": {"F": 1024, "H4": 7168, "R": 7168, "E_hat": 7168, "E": 7168, "z": 128},
    }
    if args.trace_profile == "continuous":
        manifest.update({
            "audit_kind": "frozen_short_horizon_sequential", "trace_profile": "continuous",
            "config": str(args.config), "stage2_checkpoint": str(stage3["stage2_checkpoint"]),
            "sequence_settings": SEQUENCE_SETTINGS,
            "sequence_code_sha256": {name: _file_sha256(Path(__file__).with_name(name)) for name in (
                "parrad_stage3_sequence_data.py", "parrad_stage3_sequence_features.py", "parrad_stage3_sequence_probe.py",
            )},
        })
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def thermal_guard() -> None:
        paused = False
        while True:
            value = subprocess.check_output([
                "nvidia-smi", "--id=0", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits",
            ], text=True).strip()
            temperature = int(value)
            if temperature < (78 if paused else 82):
                return
            paused = True
            print(json.dumps({"thermal_pause": True, "temperature_c": temperature}), flush=True)
            time.sleep(10)

    for item in items:
        started = time.monotonic()
        result = extract_recording(runtime, item, ages=extraction_ages(args.trace_profile), step_delay_s=args.step_delay_s, thermal_guard=thermal_guard if args.device == "cuda:0" else None)
        for layer, dimension in manifest["layer_dimensions"].items():
            if len(result["rows"]) and result["arrays"][f"current_{layer}"].shape[1] != dimension:
                raise RuntimeError(f"unexpected {layer} layer dimension")
        write_recording(result, args.output_dir / item.key)
        manifest["completed_records"].append({"key": item.key, "seconds": time.monotonic() - started, "integrity": result["integrity"],
                                               "artifact_sha256": {name: _file_sha256(args.output_dir / item.key / name) for name in ("layers.npz", "rows.json", "candidates.json", "integrity.json")}})
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest["completed_records"][-1]), flush=True)
    if _file_sha256(args.checkpoint) != manifest["checkpoint_sha256"]:
        raise RuntimeError("source checkpoint changed during extraction")
    if _file_sha256(Path(stage3["stage2_checkpoint"])) != manifest["stage2_checkpoint_sha256"] or _file_sha256(args.selected_thresholds) != manifest["threshold_artifact_sha256"]:
        raise RuntimeError("Stage P2 checkpoint or threshold artifact changed during extraction")
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
