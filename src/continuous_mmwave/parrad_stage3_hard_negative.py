"""Validation-safe Candidate lifecycle ablation and closed-loop Verify fine-tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from continuous_mmwave.parrad_stage3_data import (
    Stage3CacheItem,
    VerifyFeatureDataset,
    VerifyFeatureSample,
    VerifyTrajectoryDataset,
    collate_verify_batches,
    collate_verify_feature_batches,
    extract_event_runs,
)
from continuous_mmwave.parrad_stage3_diagnostics import (
    BoundaryMatch,
    ControllerThresholds,
    ThresholdGrid,
    _match_payload,
    _matched_confirmation_latencies,
    _merge_boundary_matches,
    match_boundaries_seconds,
    run_validation_sweep,
)
from continuous_mmwave.parrad_stage3_runner import (
    Stage3TrainingConfig,
    _load_cache_items,
    _move_batch,
    train_stage3_phase,
)
from continuous_mmwave.stages.parrad_stage3 import PARradStage3Module


def _apply_thresholds(runtime: Any, thresholds: ControllerThresholds) -> None:
    controller = runtime.controller
    controller.proposal_threshold = thresholds.proposal
    controller.confirm_threshold = thresholds.verify_confirm
    controller.reject_threshold = thresholds.verify_reject
    controller.reject_timeout_s = (
        thresholds.candidate_timeout_updates * controller.update_period_s
    )
    controller.proposal_persistence = thresholds.proposal_persistence


def _recording_truth(item: Stage3CacheItem) -> tuple[float, ...]:
    labels = torch.from_numpy(item.record.frame_labels.reshape(-1).astype("int64"))
    timestamps = torch.from_numpy(item.record.timestamps.reshape(-1).astype("float64"))
    return tuple(
        float(run.boundary_from_previous)
        for run in extract_event_runs(labels, timestamps)
        if run.boundary_from_previous is not None
    )


@torch.inference_mode()
def mine_closed_loop_verify_samples(
    runtime: Any,
    records: Sequence[Stage3CacheItem],
    thresholds: ControllerThresholds,
    *,
    boundary_tolerance_s: float = 0.72,
) -> tuple[VerifyFeatureSample, ...]:
    """Mine Candidate updates, assigning labels only after online inference."""

    if boundary_tolerance_s <= 0 or not math.isfinite(boundary_tolerance_s):
        raise ValueError("boundary tolerance must be finite and positive")
    _apply_thresholds(runtime, thresholds)
    controller = runtime.controller
    event_former = getattr(controller, "event_former", None)
    parameter = (
        next(event_former.parameters(), None)
        if event_former is not None and hasattr(event_former, "parameters")
        else None
    )
    device = parameter.device if parameter is not None else torch.device("cpu")
    samples: list[VerifyFeatureSample] = []
    for item in records:
        controller.reset()
        truth = _recording_truth(item)
        record = item.record
        for update_index in range(len(record.radar_tokens)):
            tokens = torch.from_numpy(record.radar_tokens[update_index].astype("float32"))
            mask = torch.from_numpy(record.radar_mask[update_index].astype("bool"))
            timestamp = float(record.timestamps[update_index].reshape(-1)[0])
            output = controller.step_tokens(
                tokens.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
                timestamp,
                item.metadata,
            )
            values = (
                output.difference_probability,
                output.stability,
                output.candidate_maturity,
                output.candidate_proposal_time,
            )
            if any(value is None for value in values):
                continue
            difference, stability, maturity, proposal_time = (
                float(value) for value in values
            )
            target = float(
                any(
                    abs(proposal_time - boundary) <= boundary_tolerance_s
                    for boundary in truth
                )
            )
            samples.append(
                VerifyFeatureSample(
                    inputs=(difference, stability, maturity),
                    target=target,
                    weight=1.0,
                    kind=(
                        "closed_loop_true_candidate"
                        if target
                        else "closed_loop_hard_negative"
                    ),
                    recording_key=item.key,
                    proposal_time=proposal_time,
                )
            )
    if not samples:
        raise ValueError("closed-loop mining produced no Candidate updates")
    return tuple(samples)


@torch.inference_mode()
def evaluate_cached_controller(
    runtime: Any,
    records: Sequence[Stage3CacheItem],
    thresholds: ControllerThresholds,
) -> dict[str, Any]:
    """Evaluate full cached recordings without reopening raw H5 files."""

    _apply_thresholds(runtime, thresholds)
    controller = runtime.controller
    event_former = getattr(controller, "event_former", None)
    parameter = (
        next(event_former.parameters(), None)
        if event_former is not None and hasattr(event_former, "parameters")
        else None
    )
    device = parameter.device if parameter is not None else torch.device("cpu")
    matches036: list[BoundaryMatch] = []
    matches072: list[BoundaryMatch] = []
    confirmation_latencies: list[float] = []
    decision_counts: Counter[str] = Counter()
    candidate_updates = total_updates = 0
    state_norms: list[float] = []
    for item in records:
        controller.reset()
        truth = list(_recording_truth(item))
        predicted: list[float] = []
        confirmations: list[tuple[float, float]] = []
        record = item.record
        for update_index in range(len(record.radar_tokens)):
            timestamp = float(record.timestamps[update_index].reshape(-1)[0])
            tokens = torch.from_numpy(record.radar_tokens[update_index].astype("float32"))
            mask = torch.from_numpy(record.radar_mask[update_index].astype("bool"))
            output = controller.step_tokens(
                tokens.unsqueeze(0).to(device),
                mask.unsqueeze(0).to(device),
                timestamp,
                item.metadata,
            )
            total_updates += 1
            candidate_updates += output.state == "candidate"
            decision_counts[output.decision] += 1
            if output.decision == "confirm" and output.finalization_job is not None:
                predicted.append(output.finalization_job.end_time)
                confirmations.append((output.finalization_job.end_time, timestamp))
            branch = controller.candidate if controller.candidate is not None else controller.active
            if branch is not None:
                state_norms.append(float(torch.linalg.vector_norm(branch.event_tokens)))
        match036 = match_boundaries_seconds(predicted, truth, 0.36)
        match072 = match_boundaries_seconds(predicted, truth, 0.72)
        matches036.append(match036)
        matches072.append(match072)
        confirmation_latencies.extend(
            _matched_confirmation_latencies(confirmations, truth, 0.72)
        )
    merged036 = _merge_boundary_matches(matches036)
    merged072 = _merge_boundary_matches(matches072)
    truth_count = merged036.true_positives + merged036.false_negatives
    predicted_count = merged036.true_positives + merged036.false_positives
    return {
        "thresholds": asdict(thresholds),
        "boundary_036": _match_payload(merged036),
        "boundary_072": _match_payload(merged072),
        "over_segmentation": merged036.false_positives / max(1, truth_count),
        "under_segmentation": merged036.false_negatives / max(1, truth_count),
        "predicted_to_truth_ratio": predicted_count / max(1, truth_count),
        "median_confirmation_latency_s": (
            statistics.median(confirmation_latencies)
            if confirmation_latencies
            else math.inf
        ),
        "candidate_duty_cycle": candidate_updates / max(1, total_updates),
        "decision_counts": dict(decision_counts),
        "state_norm": {
            "maximum": max(state_norms, default=0.0),
            "mean": sum(state_norms) / max(1, len(state_norms)),
            "all_finite": all(math.isfinite(value) for value in state_norms),
        },
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Stage P3 config root must be a mapping")
    return payload


def _load_thresholds(
    path: Path,
    checkpoint: Path,
    *,
    require_checkpoint_match: bool = True,
) -> ControllerThresholds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("source_split") != "validation" or payload.get("test_evaluation_allowed") is not False:
        raise ValueError("hard-negative training requires a validation-only threshold artifact")
    selected = payload.get("selected_checkpoint", {})
    if require_checkpoint_match and selected.get("sha256") != _file_sha256(checkpoint):
        raise ValueError("threshold artifact checkpoint SHA does not match the source checkpoint")
    return ControllerThresholds(**payload["thresholds"])


def _build_heads(raw: Mapping[str, Any]) -> PARradStage3Module:
    model = dict(raw.get("model", {}))
    stage3 = dict(raw.get("stage3", {}))
    return PARradStage3Module(
        radar_dim=int(model.get("radar_dim", 256)),
        semantic_dim=int(model.get("semantic_dim", 128)),
        prototype_dim=int(stage3.get("prototype_dim", 128)),
        update_period_s=float(model.get("frame_period_s", 0.09))
        * int(model.get("frames_per_update", 2)),
        tau_fast_s=float(stage3.get("tau_fast_s", 2.0)),
        tau_stability=float(stage3.get("tau_stability", 0.1)),
        physical_tbptt_steps=int(stage3.get("physical_tbptt_steps", 32)),
    )


@torch.inference_mode()
def _materialize_oracle_features(
    module: PARradStage3Module,
    records: tuple[Stage3CacheItem, ...],
    batch_size: int,
    device: torch.device,
) -> tuple[VerifyFeatureSample, ...]:
    dataset = VerifyTrajectoryDataset(records)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_verify_batches,
    )
    result: list[VerifyFeatureSample] = []
    offset = 0
    module.eval()
    for raw_batch in loader:
        batch = _move_batch(raw_batch, device)
        metrics = module.verify_step(batch)
        inputs = metrics["verify_inputs"].detach().cpu()
        for row, target, weight in zip(
            inputs,
            raw_batch.target,
            raw_batch.weight,
            strict=True,
        ):
            kind = dataset.samples[offset].kind
            result.append(
                VerifyFeatureSample(
                    inputs=tuple(float(value) for value in row),
                    target=float(target),
                    weight=float(weight),
                    kind=f"oracle_{kind}",
                )
            )
            offset += 1
    return tuple(result)


def _balanced_rehearsal_samples(
    oracle: tuple[VerifyFeatureSample, ...],
    closed_loop: tuple[VerifyFeatureSample, ...],
    hard_negative_weight: float,
) -> tuple[VerifyFeatureSample, ...]:
    if hard_negative_weight <= 0 or not math.isfinite(hard_negative_weight):
        raise ValueError("hard-negative weight must be finite and positive")
    combined = [
        replace(
            sample,
            weight=sample.weight
            * (hard_negative_weight if sample.kind == "closed_loop_hard_negative" else 1.0),
        )
        for sample in (*oracle, *closed_loop)
    ]
    positive_weight = sum(sample.weight for sample in combined if sample.target >= 0.5)
    negative_weight = sum(sample.weight for sample in combined if sample.target < 0.5)
    if positive_weight <= 0 or negative_weight <= 0:
        raise ValueError("hard-negative rehearsal requires both positive and negative samples")
    scale = negative_weight / positive_weight
    return tuple(
        replace(sample, weight=sample.weight * scale)
        if sample.target >= 0.5
        else sample
        for sample in combined
    )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def _cache_validation_fingerprint(
    cache_fingerprint: str,
    records: Sequence[Stage3CacheItem],
) -> str:
    digest = hashlib.sha256(cache_fingerprint.encode("utf-8"))
    for item in records:
        digest.update(item.key.encode("utf-8"))
        digest.update(item.record.frame_labels.tobytes())
        digest.update(item.record.timestamps.tobytes())
    return digest.hexdigest()


def run_cached_timeout_reject_ablation(
    config_path: Path,
    checkpoint: Path,
    selected_thresholds: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    timeout_shard_index: int | None = None,
    proposal: float = 0.55,
    reuse_threshold_values: bool = False,
) -> ControllerThresholds:
    """Run the 18-point Candidate lifecycle ablation on validation cache only."""

    from continuous_mmwave.parrad_stage3_calibrate import (
        build_runtime,
        timeout_grid_shards,
        timeout_reject_grid,
    )

    raw = _load_yaml(config_path)
    stage3 = dict(raw.get("stage3", {}))
    base = _load_thresholds(
        selected_thresholds,
        checkpoint,
        require_checkpoint_match=not reuse_threshold_values,
    )
    grid = timeout_reject_grid(base, proposal=proposal)
    if timeout_shard_index is not None:
        shards = timeout_grid_shards(grid)
        if not 0 <= timeout_shard_index < len(shards):
            raise ValueError("timeout shard index is outside the lifecycle grid")
        grid = shards[timeout_shard_index]
    records, cache_fingerprint = _load_cache_items(
        Path(str(stage3["cache_dir"])),
        Path(str(stage3["stage2_checkpoint"])),
        splits=("validation",),
    )
    validation_records = records["validation"]
    target_device = _resolve_device(device)
    runtime = build_runtime(config_path, checkpoint, target_device)

    def evaluator(_checkpoint: Path, thresholds: ControllerThresholds) -> Mapping[str, Any]:
        return evaluate_cached_controller(runtime, validation_records, thresholds)

    return run_validation_sweep(
        {"verify": checkpoint},
        grid,
        evaluator,
        output_dir,
        validation_fingerprint=_cache_validation_fingerprint(
            cache_fingerprint, validation_records
        ),
    )


def run_hard_negative_verify_training(
    config_path: Path,
    checkpoint: Path,
    selected_thresholds: Path,
    output_dir: Path,
    *,
    device: str = "auto",
    epochs: int = 12,
    learning_rate: float = 5e-4,
    hard_negative_weight: float = 2.0,
    boundary_tolerance_s: float = 0.72,
) -> Path:
    """Fine-tune only Verify on oracle rehearsal plus mined closed-loop candidates."""

    from continuous_mmwave.parrad_stage3_calibrate import build_runtime

    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning rate must be positive")
    raw = _load_yaml(config_path)
    stage3 = dict(raw.get("stage3", {}))
    training = dict(raw.get("training", {}))
    thresholds = _load_thresholds(selected_thresholds, checkpoint)
    records, cache_fingerprint = _load_cache_items(
        Path(str(stage3["cache_dir"])),
        Path(str(stage3["stage2_checkpoint"])),
        splits=("train", "validation"),
    )
    target_device = _resolve_device(device)
    runtime = build_runtime(config_path, checkpoint, target_device)
    train_closed = mine_closed_loop_verify_samples(
        runtime,
        records["train"],
        thresholds,
        boundary_tolerance_s=boundary_tolerance_s,
    )
    validation_closed = mine_closed_loop_verify_samples(
        runtime,
        records["validation"],
        thresholds,
        boundary_tolerance_s=boundary_tolerance_s,
    )

    module = _build_heads(raw).to(target_device)
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if source.get("phase") != "verify" or source.get("cache_fingerprint") != cache_fingerprint:
        raise ValueError("source Verify checkpoint does not match the configured cache")
    module.load_state_dict(source["model_state"], strict=True)
    batch_size = int(stage3.get("batch_size", 64))
    oracle = _materialize_oracle_features(
        module, records["train"], batch_size, target_device
    )
    training_samples = _balanced_rehearsal_samples(
        oracle, train_closed, hard_negative_weight
    )
    validation_samples = tuple(replace(sample, weight=1.0) for sample in validation_closed)

    settings = Stage3TrainingConfig(
        cache_dir=Path(str(stage3["cache_dir"])),
        stage2_checkpoint=Path(str(stage3["stage2_checkpoint"])),
        output_dir=Path(output_dir),
        seed=int(stage3.get("seed", training.get("seed", 42))),
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=float(stage3.get("weight_decay", 0.01)),
        epochs_physical=1,
        epochs_semantic=1,
        epochs_verify=epochs,
        early_stopping_patience=int(stage3.get("early_stopping_patience", 5)),
        grad_clip_norm=float(stage3.get("grad_clip_norm", 1.0)),
        device=str(target_device),
        amp=bool(stage3.get("amp", training.get("amp", True))),
    )
    settings.validate()
    settings.output_dir.mkdir(parents=True)
    generator = torch.Generator().manual_seed(settings.seed)
    train_loader = DataLoader(
        VerifyFeatureDataset(training_samples),
        batch_size=settings.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_verify_feature_batches,
    )
    validation_loader = DataLoader(
        VerifyFeatureDataset(validation_samples),
        batch_size=settings.batch_size,
        shuffle=False,
        collate_fn=collate_verify_feature_batches,
    )
    result = train_stage3_phase(
        module,
        "verify",
        train_loader,
        validation_loader,
        settings,
        cache_fingerprint,
    )
    summary = {
        "schema_version": 1,
        "source_split": ["train", "validation"],
        "test_evaluation_allowed": False,
        "source_checkpoint": str(checkpoint),
        "source_checkpoint_sha256": _file_sha256(checkpoint),
        "mining_thresholds": asdict(thresholds),
        "boundary_tolerance_s": boundary_tolerance_s,
        "hard_negative_weight": hard_negative_weight,
        "sample_counts": {
            "oracle_train": len(oracle),
            "closed_loop_train": len(train_closed),
            "closed_loop_validation": len(validation_closed),
            "closed_loop_train_positive": sum(sample.target >= 0.5 for sample in train_closed),
            "closed_loop_train_negative": sum(sample.target < 0.5 for sample in train_closed),
            "closed_loop_validation_positive": sum(sample.target >= 0.5 for sample in validation_closed),
            "closed_loop_validation_negative": sum(sample.target < 0.5 for sample in validation_closed),
        },
        "best_epoch": result.best_epoch,
        "best_metrics": dict(result.best_metrics),
        "best_checkpoint": str(result.best_checkpoint),
        "best_checkpoint_sha256": _file_sha256(result.best_checkpoint),
    }
    (settings.output_dir / "hard-negative-summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return result.best_checkpoint


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", required=True, type=Path)
    common.add_argument("--checkpoint", required=True, type=Path)
    common.add_argument("--selected-thresholds", required=True, type=Path)
    common.add_argument("--output-dir", required=True, type=Path)
    common.add_argument("--device", default="auto")
    ablate = subparsers.add_parser("ablate", parents=[common])
    ablate.add_argument("--timeout-shard-index", type=int)
    ablate.add_argument("--proposal", type=float, default=0.55)
    ablate.add_argument(
        "--reuse-threshold-values",
        action="store_true",
        help="reuse numeric validation thresholds for a new checkpoint that will be recalibrated",
    )
    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--epochs", type=int, default=12)
    train.add_argument("--learning-rate", type=float, default=5e-4)
    train.add_argument("--hard-negative-weight", type=float, default=2.0)
    train.add_argument("--boundary-tolerance-s", type=float, default=0.72)
    args = parser.parse_args(argv)
    if args.command == "ablate":
        selected = run_cached_timeout_reject_ablation(
            args.config,
            args.checkpoint,
            args.selected_thresholds,
            args.output_dir,
            device=args.device,
            timeout_shard_index=args.timeout_shard_index,
            proposal=args.proposal,
            reuse_threshold_values=args.reuse_threshold_values,
        )
        print(json.dumps(asdict(selected), sort_keys=True))
    else:
        best = run_hard_negative_verify_training(
            args.config,
            args.checkpoint,
            args.selected_thresholds,
            args.output_dir,
            device=args.device,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hard_negative_weight=args.hard_negative_weight,
            boundary_tolerance_s=args.boundary_tolerance_s,
        )
        print(json.dumps({"best_checkpoint": str(best)}, sort_keys=True))


if __name__ == "__main__":
    main()
