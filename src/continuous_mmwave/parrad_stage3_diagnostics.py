"""Continuous evaluation, threshold calibration, and latency diagnostics for Stage P3."""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import tempfile
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from continuous_mmwave.data_parrad import PARradSample
from continuous_mmwave.parrad_controller import PARradStage3Runtime
from continuous_mmwave.parrad_stage3_data import extract_event_runs


@dataclass(frozen=True)
class BoundaryMatch:
    true_positives: int
    false_positives: int
    false_negatives: int
    absolute_errors_s: tuple[float, ...]

    @property
    def precision(self) -> float:
        return self.true_positives / max(1, self.true_positives + self.false_positives)

    @property
    def recall(self) -> float:
        return self.true_positives / max(1, self.true_positives + self.false_negatives)

    @property
    def f1(self) -> float:
        return 2 * self.precision * self.recall / max(self.precision + self.recall, 1e-12)

    @property
    def mean_absolute_error_s(self) -> float:
        return statistics.fmean(self.absolute_errors_s) if self.absolute_errors_s else math.nan


@dataclass(frozen=True)
class ControllerThresholds:
    proposal: float
    verify_confirm: float
    verify_reject: float
    candidate_timeout_updates: int
    fingerprint: str
    proposal_persistence: int = 1


@dataclass(frozen=True)
class ThresholdGrid:
    proposal: tuple[float, ...]
    verify_confirm: tuple[float, ...]
    verify_reject: tuple[float, ...]
    candidate_timeout_updates: tuple[int, ...]
    proposal_persistence: tuple[int, ...] = (1,)


@dataclass(frozen=True)
class ValidationSweepResult:
    thresholds: ControllerThresholds
    boundary_f1_036: float
    boundary_f1_072: float
    over_segmentation: float
    median_confirmation_latency_s: float
    checkpoint_name: str = ""
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_threshold_candidates(
    grid: ThresholdGrid,
    checkpoint_sha256: str,
) -> Iterator[ControllerThresholds]:
    """Enumerate the immutable fixed grid with checkpoint-bound fingerprints."""

    axes = (
        grid.proposal,
        grid.verify_confirm,
        grid.verify_reject,
        grid.candidate_timeout_updates,
        grid.proposal_persistence,
    )
    if not all(axes):
        raise ValueError("threshold grid axes must be non-empty")
    for proposal, confirm, reject, timeout, persistence in product(*axes):
        payload = {
            "checkpoint_sha256": checkpoint_sha256,
            "proposal": float(proposal),
            "verify_confirm": float(confirm),
            "verify_reject": float(reject),
            "candidate_timeout_updates": int(timeout),
            "proposal_persistence": int(persistence),
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        yield ControllerThresholds(
            proposal=payload["proposal"],
            verify_confirm=payload["verify_confirm"],
            verify_reject=payload["verify_reject"],
            candidate_timeout_updates=payload["candidate_timeout_updates"],
            fingerprint=fingerprint,
            proposal_persistence=payload["proposal_persistence"],
        )


def match_boundaries_seconds(
    predicted: Sequence[float],
    truth: Sequence[float],
    tolerance_s: float,
) -> BoundaryMatch:
    """Chronologically match sorted predicted/truth boundaries one-to-one."""

    if tolerance_s < 0 or not math.isfinite(tolerance_s):
        raise ValueError("boundary tolerance must be finite and non-negative")
    predicted_values = [float(value) for value in predicted]
    truth_values = [float(value) for value in truth]
    if any(not math.isfinite(value) for value in predicted_values + truth_values):
        raise ValueError("boundary times must be finite")
    if predicted_values != sorted(predicted_values) or truth_values != sorted(truth_values):
        raise ValueError("boundary times must be sorted")
    predicted_index = truth_index = 0
    errors: list[float] = []
    false_positives = 0
    false_negatives = 0
    while predicted_index < len(predicted_values) and truth_index < len(truth_values):
        difference = predicted_values[predicted_index] - truth_values[truth_index]
        if abs(difference) <= tolerance_s:
            errors.append(abs(difference))
            predicted_index += 1
            truth_index += 1
        elif difference < 0:
            false_positives += 1
            predicted_index += 1
        else:
            false_negatives += 1
            truth_index += 1
    false_positives += len(predicted_values) - predicted_index
    false_negatives += len(truth_values) - truth_index
    return BoundaryMatch(len(errors), false_positives, false_negatives, tuple(errors))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def calibrate_thresholds(
    results: Iterable[ValidationSweepResult],
    grid: ThresholdGrid,
    output_path: Path,
) -> ControllerThresholds:
    """Select thresholds on validation only under the latency gate."""

    if not all(
        (
            grid.proposal,
            grid.verify_confirm,
            grid.verify_reject,
            grid.candidate_timeout_updates,
            grid.proposal_persistence,
        )
    ):
        raise ValueError("threshold grid axes must be non-empty")
    allowed = [result for result in results if result.median_confirmation_latency_s <= 1.2]
    if not allowed:
        raise ValueError("no validation threshold candidate meets the 1.2 s latency gate")
    selected = max(
        allowed,
        key=lambda result: (
            result.boundary_f1_072,
            -result.over_segmentation,
            result.boundary_f1_036,
        ),
    )
    thresholds = selected.thresholds
    if (
        thresholds.proposal not in grid.proposal
        or thresholds.verify_confirm not in grid.verify_confirm
        or thresholds.verify_reject not in grid.verify_reject
        or thresholds.candidate_timeout_updates not in grid.candidate_timeout_updates
        or thresholds.proposal_persistence not in grid.proposal_persistence
    ):
        raise ValueError("validation result contains thresholds outside the fixed grid")
    _atomic_json(
        Path(output_path),
        {
            "schema_version": 1,
            "status": "calibrated_validation",
            "source_split": "validation",
            "test_evaluation_allowed": False,
            "thresholds": asdict(thresholds),
            "selected_checkpoint": {
                "name": selected.checkpoint_name,
                "path": selected.checkpoint_path,
                "sha256": selected.checkpoint_sha256,
            },
            "selection_metrics": {
                key: value
                for key, value in asdict(selected).items()
                if key != "thresholds"
            },
            "grid": asdict(grid),
        },
    )
    return thresholds


def _validation_result_from_metrics(
    thresholds: ControllerThresholds,
    checkpoint_name: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    metrics: Mapping[str, Any],
) -> ValidationSweepResult:
    try:
        result = ValidationSweepResult(
            thresholds=thresholds,
            boundary_f1_036=float(metrics["boundary_036"]["f1"]),
            boundary_f1_072=float(metrics["boundary_072"]["f1"]),
            over_segmentation=float(metrics["over_segmentation"]),
            median_confirmation_latency_s=float(metrics["median_confirmation_latency_s"]),
            checkpoint_name=checkpoint_name,
            checkpoint_path=str(checkpoint_path),
            checkpoint_sha256=checkpoint_sha256,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("controller evaluator returned incomplete validation metrics") from error
    values = (
        result.boundary_f1_036,
        result.boundary_f1_072,
        result.over_segmentation,
        result.median_confirmation_latency_s,
    )
    if any(value < 0 or math.isnan(value) for value in values):
        raise ValueError("controller validation metrics must be non-negative and not NaN")
    return result


def _load_sweep_rows(path: Path) -> list[ValidationSweepResult]:
    if not path.exists():
        return []
    results: list[ValidationSweepResult] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            row["thresholds"] = ControllerThresholds(**row["thresholds"])
            results.append(ValidationSweepResult(**row))
        except (json.JSONDecodeError, KeyError, TypeError) as error:
            raise ValueError(f"invalid validation sweep row {line_number}") from error
    return results


def run_validation_sweep(
    checkpoints: Mapping[str, Path],
    grid: ThresholdGrid,
    evaluator: Callable[[Path, ControllerThresholds], Mapping[str, Any]],
    output_dir: Path,
    *,
    validation_fingerprint: str = "unspecified",
) -> ControllerThresholds:
    """Evaluate and resume a checkpoint-by-threshold validation-only sweep."""

    if not checkpoints:
        raise ValueError("validation sweep requires at least one checkpoint")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "validation-controller-sweep.jsonl"
    resolved_checkpoints: dict[str, tuple[Path, str]] = {}
    for checkpoint_name, raw_path in checkpoints.items():
        checkpoint_path = Path(raw_path).resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        resolved_checkpoints[str(checkpoint_name)] = (
            checkpoint_path,
            _file_sha256(checkpoint_path),
        )
    context = {
        "schema_version": 1,
        "source_split": "validation",
        "validation_fingerprint": validation_fingerprint,
        "grid": asdict(grid),
        "checkpoints": {
            name: {"path": str(path), "sha256": checksum}
            for name, (path, checksum) in resolved_checkpoints.items()
        },
    }
    context = json.loads(json.dumps(context, sort_keys=True))
    context_path = output_dir / "validation-controller-context.json"
    if context_path.exists():
        existing_context = json.loads(context_path.read_text(encoding="utf-8"))
        if existing_context != context:
            raise ValueError("existing sweep context does not match this validation run")
    else:
        if rows_path.exists() and rows_path.stat().st_size:
            raise ValueError("existing sweep rows have no immutable sweep context")
        _atomic_json(context_path, context)
    results = _load_sweep_rows(rows_path)
    completed = {result.thresholds.fingerprint for result in results}
    for checkpoint_name, (checkpoint_path, checkpoint_sha256) in resolved_checkpoints.items():
        for thresholds in iter_threshold_candidates(grid, checkpoint_sha256):
            if thresholds.fingerprint in completed:
                continue
            metrics = evaluator(checkpoint_path, thresholds)
            result = _validation_result_from_metrics(
                thresholds,
                str(checkpoint_name),
                checkpoint_path,
                checkpoint_sha256,
                metrics,
            )
            with rows_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            results.append(result)
            completed.add(thresholds.fingerprint)
    threshold_path = output_dir / "validation_thresholds.json"
    selected = calibrate_thresholds(results, grid, threshold_path)
    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    context_text = json.dumps(context, sort_keys=True, separators=(",", ":"))
    threshold_payload["validation_context"] = context
    threshold_payload["validation_context_sha256"] = hashlib.sha256(
        context_text.encode("utf-8")
    ).hexdigest()
    _atomic_json(threshold_path, threshold_payload)
    return selected


def merge_validation_sweeps(
    source_dirs: Sequence[Path],
    output_dir: Path,
    *,
    target_grid: ThresholdGrid | None = None,
) -> ControllerThresholds:
    """Merge complete checkpoint or grid shards and select globally."""

    if not source_dirs:
        raise ValueError("at least one validation sweep shard is required")
    base_context: dict[str, Any] | None = None
    first_grid_payload: dict[str, Any] | None = None
    merged_checkpoints: dict[str, dict[str, str]] = {}
    merged_results: list[ValidationSweepResult] = []
    for raw_source in source_dirs:
        source = Path(raw_source)
        context_path = source / "validation-controller-context.json"
        if not context_path.is_file():
            raise FileNotFoundError(context_path)
        context = json.loads(context_path.read_text(encoding="utf-8"))
        common = {
            "schema_version": context.get("schema_version"),
            "source_split": context.get("source_split"),
            "validation_fingerprint": context.get("validation_fingerprint"),
        }
        if base_context is None:
            base_context = common
        elif common != base_context:
            raise ValueError("validation sweep shards have incompatible contexts")
        grid_payload = context.get("grid")
        if not isinstance(grid_payload, dict):
            raise ValueError("validation sweep shard contains no threshold grid")
        if first_grid_payload is None:
            first_grid_payload = grid_payload
        elif target_grid is None and grid_payload != first_grid_payload:
            raise ValueError("validation sweep shards have incompatible contexts")
        shard_grid = ThresholdGrid(
            proposal=tuple(grid_payload["proposal"]),
            verify_confirm=tuple(grid_payload["verify_confirm"]),
            verify_reject=tuple(grid_payload["verify_reject"]),
            candidate_timeout_updates=tuple(grid_payload["candidate_timeout_updates"]),
            proposal_persistence=tuple(grid_payload["proposal_persistence"]),
        )
        for name, checkpoint in context.get("checkpoints", {}).items():
            if name in merged_checkpoints and merged_checkpoints[name] != checkpoint:
                raise ValueError("validation sweep shards redefine a checkpoint name")
            checkpoint_path = Path(checkpoint["path"])
            if not checkpoint_path.is_file() or _file_sha256(checkpoint_path) != checkpoint["sha256"]:
                raise ValueError("validation sweep shard checkpoint changed after evaluation")
            if target_grid is not None:
                shard_candidates = {
                    candidate.fingerprint
                    for candidate in iter_threshold_candidates(shard_grid, checkpoint["sha256"])
                }
                target_candidates = {
                    candidate.fingerprint
                    for candidate in iter_threshold_candidates(target_grid, checkpoint["sha256"])
                }
                if not shard_candidates <= target_candidates:
                    raise ValueError("validation sweep shard grid is outside the target grid")
            merged_checkpoints[name] = checkpoint
        merged_results.extend(
            _load_sweep_rows(source / "validation-controller-sweep.jsonl")
        )
    if base_context is None or not merged_checkpoints:
        raise ValueError("validation sweep shards contain no checkpoints")
    if target_grid is None:
        if first_grid_payload is None:
            raise ValueError("validation sweep shards contain no threshold grid")
        grid = ThresholdGrid(
            proposal=tuple(first_grid_payload["proposal"]),
            verify_confirm=tuple(first_grid_payload["verify_confirm"]),
            verify_reject=tuple(first_grid_payload["verify_reject"]),
            candidate_timeout_updates=tuple(first_grid_payload["candidate_timeout_updates"]),
            proposal_persistence=tuple(first_grid_payload["proposal_persistence"]),
        )
    else:
        grid = target_grid
    expected = {
        threshold.fingerprint
        for checkpoint in merged_checkpoints.values()
        for threshold in iter_threshold_candidates(grid, checkpoint["sha256"])
    }
    actual: set[str] = set()
    for result in merged_results:
        checkpoint = merged_checkpoints.get(result.checkpoint_name)
        if checkpoint is None:
            raise ValueError("validation result references an unknown checkpoint")
        if (
            result.checkpoint_path != checkpoint["path"]
            or result.checkpoint_sha256 != checkpoint["sha256"]
            or result.thresholds.fingerprint in actual
        ):
            raise ValueError("validation result checkpoint identity is inconsistent")
        actual.add(result.thresholds.fingerprint)
    if actual != expected:
        raise ValueError("validation sweep shard is incomplete for the fixed grid")
    context = {**base_context, "grid": asdict(grid), "checkpoints": merged_checkpoints}
    context = json.loads(json.dumps(context, sort_keys=True))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "validation-controller-context.json", context)
    rows_path = output_dir / "validation-controller-sweep.jsonl"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{rows_path.name}.", suffix=".tmp", dir=output_dir
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for result in sorted(
                merged_results,
                key=lambda value: (value.checkpoint_name, value.thresholds.fingerprint),
            ):
                handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")
        os.replace(temporary_name, rows_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    threshold_path = output_dir / "validation_thresholds.json"
    selected = calibrate_thresholds(merged_results, grid, threshold_path)
    threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    context_text = json.dumps(context, sort_keys=True, separators=(",", ":"))
    threshold_payload["validation_context"] = context
    threshold_payload["validation_context_sha256"] = hashlib.sha256(
        context_text.encode("utf-8")
    ).hexdigest()
    _atomic_json(threshold_path, threshold_payload)
    return selected


def evaluate_test_once(
    evaluator: Callable[[ControllerThresholds], dict[str, Any]],
    thresholds: ControllerThresholds,
    marker_path: Path,
) -> dict[str, Any]:
    """Execute one immutable test evaluation guarded by an exclusive marker."""

    marker_path = Path(marker_path)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with marker_path.open("x", encoding="utf-8") as handle:
            json.dump({"status": "running", "threshold_fingerprint": thresholds.fingerprint}, handle)
    except FileExistsError as error:
        raise RuntimeError("test split was already evaluated for this output directory") from error
    metrics = evaluator(thresholds)
    _atomic_json(
        marker_path,
        {
            "status": "complete",
            "threshold_fingerprint": thresholds.fingerprint,
            "metrics": metrics,
        },
    )
    return metrics


def _apply_thresholds(runtime: PARradStage3Runtime, thresholds: ControllerThresholds) -> None:
    controller = runtime.controller
    controller.proposal_threshold = thresholds.proposal
    controller.confirm_threshold = thresholds.verify_confirm
    controller.reject_threshold = thresholds.verify_reject
    controller.reject_timeout_s = thresholds.candidate_timeout_updates * controller.update_period_s
    controller.proposal_persistence = thresholds.proposal_persistence


def _match_payload(match: BoundaryMatch) -> dict[str, float | int]:
    return {
        "true_positives": match.true_positives,
        "false_positives": match.false_positives,
        "false_negatives": match.false_negatives,
        "precision": match.precision,
        "recall": match.recall,
        "f1": match.f1,
        "mean_absolute_error_s": match.mean_absolute_error_s,
    }


def _merge_boundary_matches(matches: Sequence[BoundaryMatch]) -> BoundaryMatch:
    return BoundaryMatch(
        true_positives=sum(match.true_positives for match in matches),
        false_positives=sum(match.false_positives for match in matches),
        false_negatives=sum(match.false_negatives for match in matches),
        absolute_errors_s=tuple(
            error for match in matches for error in match.absolute_errors_s
        ),
    )


def _matched_confirmation_latencies(
    confirmations: Sequence[tuple[float, float]],
    truth: Sequence[float],
    tolerance_s: float,
) -> list[float]:
    """Measure GT-boundary to confirm latency for matched proposed boundaries."""

    predicted_index = truth_index = 0
    latencies: list[float] = []
    while predicted_index < len(confirmations) and truth_index < len(truth):
        proposal_time, confirmation_time = confirmations[predicted_index]
        difference = proposal_time - truth[truth_index]
        if abs(difference) <= tolerance_s:
            latencies.append(max(0.0, confirmation_time - truth[truth_index]))
            predicted_index += 1
            truth_index += 1
        elif difference < 0:
            predicted_index += 1
        else:
            truth_index += 1
    return latencies


def _runtime_device(runtime: PARradStage3Runtime) -> torch.device:
    encoder = runtime.radar_encoder
    if isinstance(encoder, torch.nn.Module):
        parameter = next(encoder.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(encoder.buffers(), None)
        if buffer is not None:
            return buffer.device
    return torch.device("cpu")


@torch.inference_mode()
def evaluate_controller(
    runtime: PARradStage3Runtime,
    recordings: Sequence[PARradSample],
    thresholds: ControllerThresholds,
    *,
    capture_state_trace: bool = False,
) -> dict[str, Any]:
    """Run complete recordings; labels are consumed only after online inference."""

    _apply_thresholds(runtime, thresholds)
    full_matches036: list[BoundaryMatch] = []
    full_matches072: list[BoundaryMatch] = []
    proposal_matches036: list[BoundaryMatch] = []
    proposal_matches072: list[BoundaryMatch] = []
    fixed_matches036: list[BoundaryMatch] = []
    fixed_matches072: list[BoundaryMatch] = []
    recording_reports: list[dict[str, Any]] = []
    candidate_updates = total_updates = proposals = confirms = 0
    state_norms: list[float] = []
    state_trace: list[dict[str, Any]] = []
    confirmation_latencies: list[float] = []
    device = _runtime_device(runtime)
    for sample in recordings:
        runtime.reset()
        recording_trace: list[dict[str, Any]] = []
        predicted: list[float] = []
        confirmations: list[tuple[float, float]] = []
        proposed: list[float] = []
        baseline_previous: Tensor | None = None
        baseline_previous_mask: Tensor | None = None
        baseline_prototype = None
        baseline_ready = False
        baseline_streak = 0
        step = 2
        if hasattr(runtime.radar_encoder, "config"):
            step = int(runtime.radar_encoder.config.frames_per_update)
        for start in range(0, sample.labels.numel(), step):
            stop = min(start + step, sample.labels.numel())
            valid = stop - start
            rd = sample.rd[start:stop].to(device, non_blocking=True)
            md = sample.md[start:stop].to(device, non_blocking=True)
            if valid < step:
                rd = torch.cat((rd, rd.new_zeros((step - valid, *rd.shape[1:]))))
                md = torch.cat((md, md.new_zeros((step - valid, md.shape[1]))))
            frame_mask = torch.arange(step, device=device).unsqueeze(0) < valid
            state_before = runtime.controller.state
            branch_before = (
                runtime.controller.candidate
                if runtime.controller.candidate is not None
                else runtime.controller.active
            )
            state_norm_before = (
                float(torch.linalg.vector_norm(branch_before.event_tokens))
                if branch_before is not None
                else 0.0
            )
            candidate_age_before = (
                runtime.controller.candidate.age_updates
                if runtime.controller.candidate is not None
                else None
            )
            output = runtime.step(
                rd.unsqueeze(0),
                md.unsqueeze(0),
                torch.tensor([sample.sensor_index], dtype=torch.long, device=device),
                frame_mask,
                float(sample.timestamps[start]),
                {
                    "subject_id": sample.subject_id,
                    "session_id": sample.session_id,
                    "sensor_id": sample.sensor_id,
                },
            )
            total_updates += 1
            candidate_updates += output.state == "candidate"
            proposals += output.decision == "proposal"
            confirms += output.decision == "confirm"
            if output.decision == "confirm" and output.finalization_job is not None:
                predicted.append(output.finalization_job.end_time)
                confirmations.append(
                    (output.finalization_job.end_time, float(sample.timestamps[start]))
                )

            # Independent physical Proposal-only baseline. It never enters Candidate,
            # so Verify thresholds cannot suppress later physical proposals.
            baseline_tokens = runtime.controller._previous_radar
            baseline_mask = runtime.controller._previous_mask
            if baseline_tokens is None or baseline_mask is None:
                raise RuntimeError("controller did not retain the encoded radar update")
            if baseline_previous is None or baseline_previous_mask is None:
                local_probability = 0.0
            else:
                common_mask = baseline_mask.bool() & baseline_previous_mask.bool()
                local_probability = float(
                    torch.sigmoid(
                        runtime.controller.local_change(
                            baseline_previous, baseline_tokens, common_mask
                        )
                    )[0]
                )
            radar_logits, updated_baseline_prototype = runtime.controller.radar_transition(
                baseline_tokens,
                baseline_mask,
                baseline_prototype,
                runtime.controller.update_period_s,
            )
            radar_probability = float(torch.sigmoid(radar_logits)[0])
            baseline_probability = 1 - (1 - local_probability) * (1 - radar_probability)
            baseline_streak = (
                baseline_streak + 1
                if baseline_probability >= thresholds.proposal
                else 0
            )
            if baseline_ready and baseline_streak >= thresholds.proposal_persistence:
                proposed.append(float(sample.timestamps[start]))
                _, baseline_prototype = runtime.controller.radar_transition(
                    baseline_tokens,
                    baseline_mask,
                    None,
                    runtime.controller.update_period_s,
                )
                baseline_streak = 0
            else:
                baseline_prototype = updated_baseline_prototype
            baseline_previous = baseline_tokens.clone()
            baseline_previous_mask = baseline_mask.clone()
            baseline_ready = True

            branch = runtime.controller.candidate if runtime.controller.candidate is not None else runtime.controller.active
            if branch is not None:
                state_norm_after = float(torch.linalg.vector_norm(branch.event_tokens))
                state_norms.append(state_norm_after)
            else:
                state_norm_after = 0.0
            if capture_state_trace:
                candidate_age_after = (
                    runtime.controller.candidate.age_updates
                    if runtime.controller.candidate is not None
                    else None
                )
                candidate_elapsed_s = (
                    candidate_age_before * runtime.controller.update_period_s
                    if candidate_age_before is not None
                    else None
                )
                recording_trace.append(
                    {
                        "subject_id": sample.subject_id,
                        "session_id": sample.session_id,
                        "sensor_id": sample.sensor_id,
                        "update_index": start // step,
                        "frame_index": start,
                        "timestamp": float(sample.timestamps[start]),
                        "state_before": state_before,
                        "state_after": output.state,
                        "decision": output.decision,
                        "state_norm_before": state_norm_before,
                        "state_norm_after": state_norm_after,
                        "candidate_age_before": candidate_age_before,
                        "candidate_age_after": candidate_age_after,
                        "candidate_elapsed_s": candidate_elapsed_s,
                        "timeout_eligible": (
                            candidate_elapsed_s is not None
                            and candidate_elapsed_s >= runtime.controller.reject_timeout_s
                        ),
                        "proposal_probability": output.proposal_probability,
                        "difference_probability": output.difference_probability,
                        "stability": output.stability,
                        "verify_probability": output.verify_probability,
                    }
                )
        truth = [
            float(run.boundary_from_previous)
            for run in extract_event_runs(sample.labels, sample.timestamps)
            if run.boundary_from_previous is not None
        ]
        match036 = match_boundaries_seconds(predicted, truth, 0.36)
        match072 = match_boundaries_seconds(predicted, truth, 0.72)
        proposal036 = match_boundaries_seconds(proposed, truth, 0.36)
        proposal072 = match_boundaries_seconds(proposed, truth, 0.72)
        fixed_delay = [value + runtime.controller.update_period_s for value in proposed]
        fixed036 = match_boundaries_seconds(fixed_delay, truth, 0.36)
        fixed072 = match_boundaries_seconds(fixed_delay, truth, 0.72)
        if capture_state_trace:
            for row in recording_trace:
                row["label"] = int(sample.labels[row["frame_index"]])
                row["nearest_truth_distance_s"] = (
                    min(abs(row["timestamp"] - boundary) for boundary in truth)
                    if truth
                    else math.inf
                )
                row["recording_boundary_f1_036"] = match036.f1
                row["recording_boundary_f1_072"] = match072.f1
            state_trace.extend(recording_trace)
        full_matches036.append(match036)
        full_matches072.append(match072)
        proposal_matches036.append(proposal036)
        proposal_matches072.append(proposal072)
        fixed_matches036.append(fixed036)
        fixed_matches072.append(fixed072)
        confirmation_latencies.extend(
            _matched_confirmation_latencies(confirmations, truth, 0.72)
        )
        recording_reports.append(
            {
                "subject_id": sample.subject_id,
                "session_id": sample.session_id,
                "sensor_id": sample.sensor_id,
                "predicted_boundaries": predicted,
                "confirmation_times": [value[1] for value in confirmations],
                "proposal_boundaries": proposed,
                "truth_boundaries": truth,
                "boundary_036": _match_payload(match036),
                "boundary_072": _match_payload(match072),
            }
        )
    match036 = _merge_boundary_matches(full_matches036)
    match072 = _merge_boundary_matches(full_matches072)
    proposal036 = _merge_boundary_matches(proposal_matches036)
    proposal072 = _merge_boundary_matches(proposal_matches072)
    fixed036 = _merge_boundary_matches(fixed_matches036)
    fixed072 = _merge_boundary_matches(fixed_matches072)
    predicted_count = match036.true_positives + match036.false_positives
    truth_count = match036.true_positives + match036.false_negatives
    proposal_count = proposal036.true_positives + proposal036.false_positives
    metrics = {
        "thresholds": asdict(thresholds),
        "boundary_036": _match_payload(match036),
        "boundary_072": _match_payload(match072),
        "over_segmentation": match036.false_positives / max(1, truth_count),
        "under_segmentation": match036.false_negatives / max(1, truth_count),
        "predicted_to_truth_ratio": predicted_count / max(1, truth_count),
        "median_confirmation_latency_s": (
            statistics.median(confirmation_latencies) if confirmation_latencies else math.inf
        ),
        "candidate_duty_cycle": candidate_updates / max(1, total_updates),
        "proposal_to_confirm_ratio": proposals / max(1, confirms),
        "false_proposal_rate": proposal072.false_positives / max(1, proposal_count),
        "state_norm": {
            "maximum": max(state_norms, default=0.0),
            "mean": statistics.fmean(state_norms) if state_norms else 0.0,
            "all_finite": all(math.isfinite(value) for value in state_norms),
        },
        "baselines": {
            "proposal_only": {
                "boundary_036": _match_payload(proposal036),
                "boundary_072": _match_payload(proposal072),
            },
            "proposal_fixed_delay": {
                "delay_s": runtime.controller.update_period_s,
                "boundary_036": _match_payload(fixed036),
                "boundary_072": _match_payload(fixed072),
            },
            "full_controller": {
                "boundary_036": _match_payload(match036),
                "boundary_072": _match_payload(match072),
            },
        },
        "recordings": recording_reports,
    }
    if capture_state_trace:
        metrics["state_trace"] = state_trace
    return metrics


@torch.inference_mode()
def latency_benchmark(
    runtime: PARradStage3Runtime,
    updates: Sequence[tuple[Tensor, Tensor, Tensor, Tensor, float]],
    warmup: int = 50,
    repetitions: int = 500,
) -> dict[str, float]:
    """Benchmark complete runtime updates with synchronized CUDA timing."""

    if not updates or warmup < 0 or repetitions <= 0:
        raise ValueError("latency benchmark requires updates and valid repetition counts")
    device = updates[0][0].device
    runtime.reset()
    for index in range(warmup):
        rd, md, sensor, mask, timestamp = updates[index % len(updates)]
        runtime.step(rd, md, sensor, mask, timestamp + index * 10_000)
    latencies: list[float] = []
    by_state: dict[str, list[float]] = {"active": [], "candidate": [], "control_commit": []}
    runtime.reset()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for index in range(repetitions):
        rd, md, sensor, mask, _ = updates[index % len(updates)]
        state_before = runtime.controller.state
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        runtime.step(rd, md, sensor, mask, index * runtime.controller.update_period_s)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)
        decision = runtime.controller.decision_trace[-1]
        category = "control_commit" if decision in {"confirm", "reject"} else state_before
        by_state[category].append(elapsed_ms)
    ordered = sorted(latencies)
    percentile95 = ordered[min(len(ordered) - 1, math.ceil(len(ordered) * 0.95) - 1)]
    result = {
        "p50_ms": statistics.median(ordered),
        "p95_ms": percentile95,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / 2**20 if device.type == "cuda" else 0.0,
    }
    for state, values in by_state.items():
        result[f"{state}_p50_ms"] = statistics.median(values) if values else math.nan
        result[f"{state}_count"] = float(len(values))
    return result
