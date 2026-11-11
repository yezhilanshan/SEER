"""Validation-only closed-loop calibration entrypoint for PARrad Stage P3."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import yaml

from continuous_mmwave.checkpoint import load_training_checkpoint
from continuous_mmwave.data_parrad import (
    PARradManifestDataset,
    PARradSample,
    build_sensor_vocabulary,
)
from continuous_mmwave.parrad_controller import PARradOnlineController, PARradStage3Runtime
from continuous_mmwave.parrad_stage3_diagnostics import (
    ControllerThresholds,
    ThresholdGrid,
    evaluate_controller,
    run_validation_sweep,
)
from continuous_mmwave.stages.parrad_stage3 import PARradStage3Module


def fixed_threshold_grid(
    update_period_s: float = 0.18,
    reject_timeout_s: float = 1.8,
) -> ThresholdGrid:
    """Return the threshold grid frozen in the Stage P3 design."""

    if update_period_s <= 0 or reject_timeout_s <= 0:
        raise ValueError("controller update period and reject timeout must be positive")
    timeout_updates = max(1, math.ceil(reject_timeout_s / update_period_s))
    return ThresholdGrid(
        proposal=(0.3, 0.4, 0.5, 0.6, 0.7),
        verify_confirm=(0.5, 0.6, 0.7, 0.8),
        verify_reject=(0.2, 0.3, 0.4),
        candidate_timeout_updates=(timeout_updates,),
        proposal_persistence=(1, 2),
    )


def coarse_threshold_grid(
    update_period_s: float = 0.18,
    reject_timeout_s: float = 1.8,
) -> ThresholdGrid:
    """Reduced production grid used to choose a checkpoint before refinement."""

    if update_period_s <= 0 or reject_timeout_s <= 0:
        raise ValueError("controller update period and reject timeout must be positive")
    timeout_updates = max(1, math.ceil(reject_timeout_s / update_period_s))
    return ThresholdGrid(
        proposal=(0.4, 0.5, 0.6),
        # The sequence-v2 Verify head produces calibrated validation scores in
        # roughly [0.33, 0.57].  Values >= 0.6 therefore suppress every
        # confirmation and cannot exercise the closed-loop controller.
        verify_confirm=(0.45, 0.5),
        verify_reject=(0.2, 0.3),
        candidate_timeout_updates=(timeout_updates,),
        proposal_persistence=(1, 2),
    )


def proposal_grid_shards(grid: ThresholdGrid) -> tuple[ThresholdGrid, ...]:
    """Partition a grid by proposal threshold without changing its candidates."""

    if not grid.proposal:
        raise ValueError("proposal threshold axis must be non-empty")
    return tuple(
        ThresholdGrid(
            proposal=(proposal,),
            verify_confirm=grid.verify_confirm,
            verify_reject=grid.verify_reject,
            candidate_timeout_updates=grid.candidate_timeout_updates,
            proposal_persistence=grid.proposal_persistence,
        )
        for proposal in grid.proposal
    )


def _centred_thresholds(value: float, step: float = 0.05) -> tuple[float, ...]:
    if not 0 <= value <= 1 or step <= 0:
        raise ValueError("threshold centre must be in [0, 1] and step must be positive")
    return tuple(
        sorted(
            {
                round(candidate, 6)
                for candidate in (value - step, value, value + step)
                if 0 <= candidate <= 1
            }
        )
    )


def refinement_threshold_grid(
    selected: ControllerThresholds,
    proposal_step: float = 0.05,
    verify_step: float = 0.01,
    verify_points: int = 6,
) -> ThresholdGrid:
    """Build the validation-informed local grid around the coarse winner.

    The coarse sequence-v2 sweep exposed a sharp Verify operating-point cliff
    between 0.45 and 0.50, while the two reject thresholds were effectively
    identical.  Spend the refinement budget on that interval and keep the
    uninformative axes fixed.
    """

    if verify_step <= 0 or verify_points <= 0:
        raise ValueError("verify refinement step and point count must be positive")
    verify_confirm = tuple(
        sorted(
            {
                round(selected.verify_confirm + offset * verify_step, 6)
                for offset in range(verify_points)
                if selected.verify_confirm + offset * verify_step <= 1
            }
        )
    )
    return ThresholdGrid(
        proposal=_centred_thresholds(selected.proposal, proposal_step),
        verify_confirm=verify_confirm,
        verify_reject=(selected.verify_reject,),
        candidate_timeout_updates=(selected.candidate_timeout_updates,),
        proposal_persistence=(selected.proposal_persistence,),
    )


def timeout_reject_grid(
    selected: ControllerThresholds,
    proposal: float | None = None,
) -> ThresholdGrid:
    """Isolate Candidate lifetime and Reject effects near the balanced operating point."""

    proposal_value = selected.proposal if proposal is None else float(proposal)
    if not 0 <= proposal_value <= 1:
        raise ValueError("proposal threshold must be in [0, 1]")
    return ThresholdGrid(
        proposal=(proposal_value,),
        verify_confirm=(0.45, 0.46, 0.47),
        verify_reject=(0.2, 0.4),
        candidate_timeout_updates=(6, 8, 10),
        proposal_persistence=(selected.proposal_persistence,),
    )


def timeout_grid_shards(grid: ThresholdGrid) -> tuple[ThresholdGrid, ...]:
    """Partition a lifecycle ablation by timeout without changing candidates."""

    if not grid.candidate_timeout_updates:
        raise ValueError("candidate timeout axis must be non-empty")
    return tuple(
        ThresholdGrid(
            proposal=grid.proposal,
            verify_confirm=grid.verify_confirm,
            verify_reject=grid.verify_reject,
            candidate_timeout_updates=(timeout,),
            proposal_persistence=grid.proposal_persistence,
        )
        for timeout in grid.candidate_timeout_updates
    )

def smoke_threshold_grid(
    update_period_s: float = 0.18,
    reject_timeout_s: float = 1.8,
) -> ThresholdGrid:
    """One conservative point for production-wiring smoke validation."""

    if update_period_s <= 0 or reject_timeout_s <= 0:
        raise ValueError("controller update period and reject timeout must be positive")
    timeout_updates = max(1, math.ceil(reject_timeout_s / update_period_s))
    return ThresholdGrid(
        proposal=(0.5,),
        # Keep the wiring smoke point inside the observed validation score
        # support so that it tests the confirmation path as well as proposal.
        verify_confirm=(0.45,),
        verify_reject=(0.3,),
        candidate_timeout_updates=(timeout_updates,),
        proposal_persistence=(1,),
    )


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Stage P3 config root must be a mapping")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_validation_recordings(config_path: Path) -> tuple[PARradSample, ...]:
    """Load only train metadata and validation arrays; never open the test split."""

    from continuous_mmwave.train import load_experiment_config

    config = load_experiment_config(config_path)
    train_manifest = Path(config.data["train_manifest"])
    validation_manifest = Path(config.data["validation_manifest"])
    sensor_vocabulary = build_sensor_vocabulary(train_manifest, validation_manifest)
    dataset = PARradManifestDataset(validation_manifest, config.model, sensor_vocabulary)
    return tuple(dataset[index] for index in range(len(dataset)))


def _validation_fingerprint(config_path: Path, recordings: Sequence[PARradSample]) -> str:
    raw = _load_yaml(config_path)
    manifest = Path(str(raw["data"]["validation_manifest"]))
    digest = hashlib.sha256(manifest.read_bytes())
    for sample in recordings:
        identity = {
            "subject_id": sample.subject_id,
            "session_id": sample.session_id,
            "sensor_id": sample.sensor_id,
            "environment_id": sample.environment_id,
            "frames": int(sample.labels.numel()),
            "first_timestamp": float(sample.timestamps[0]),
            "last_timestamp": float(sample.timestamps[-1]),
        }
        digest.update(json.dumps(identity, sort_keys=True).encode("utf-8"))
        digest.update(sample.labels.contiguous().numpy().tobytes())
        digest.update(sample.timestamps.contiguous().numpy().tobytes())
    return digest.hexdigest()


def _validate_checkpoint_fingerprint(
    raw_config: Mapping[str, Any],
    stage3_payload: Mapping[str, Any],
) -> None:
    stage3 = dict(raw_config.get("stage3", {}))
    cache_dir = Path(str(stage3["cache_dir"]))
    stage2_checkpoint = Path(str(stage3["stage2_checkpoint"]))
    index = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    fingerprint = index["fingerprint"]
    if fingerprint["checkpoint_sha256"] != _sha256(stage2_checkpoint):
        raise ValueError("Stage P3 cache does not match the configured Stage P2 checkpoint")
    fingerprint_text = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()
    if stage3_payload.get("cache_fingerprint") != expected:
        raise ValueError("Stage P3 checkpoint does not match the configured cache fingerprint")


def build_runtime(
    config_path: Path,
    stage3_checkpoint: Path,
    device: torch.device,
) -> PARradStage3Runtime:
    """Reconstruct frozen Stage P2 plus one trained Stage P3 checkpoint."""

    from continuous_mmwave.train import _build_parrad_stage2, load_experiment_config

    raw = _load_yaml(config_path)
    experiment = load_experiment_config(config_path)
    stage3 = dict(raw.get("stage3", {}))
    model = dict(raw.get("model", {}))
    stage2_checkpoint = Path(str(stage3["stage2_checkpoint"]))
    stage2_module, _ = _build_parrad_stage2(experiment)
    load_training_checkpoint(stage2_checkpoint, stage2_module)
    stage3_payload = torch.load(stage3_checkpoint, map_location="cpu", weights_only=False)
    if stage3_payload.get("phase") != "verify":
        raise ValueError("closed-loop calibration requires a Verify-phase Stage P3 checkpoint")
    _validate_checkpoint_fingerprint(raw, stage3_payload)
    heads = PARradStage3Module(
        radar_dim=int(model.get("radar_dim", 256)),
        semantic_dim=int(model.get("semantic_dim", 128)),
        prototype_dim=int(stage3.get("prototype_dim", 128)),
        update_period_s=float(model.get("frame_period_s", 0.09))
        * int(model.get("frames_per_update", 2)),
        tau_fast_s=float(stage3.get("tau_fast_s", 2.0)),
        tau_stability=float(stage3.get("tau_stability", 0.1)),
    )
    heads.load_state_dict(stage3_payload["model_state"], strict=True)
    stage2_module.to(device).eval()
    heads.to(device).eval()
    for parameter in stage2_module.parameters():
        parameter.requires_grad = False
    for parameter in heads.parameters():
        parameter.requires_grad = False
    controller = PARradOnlineController(
        event_former=stage2_module.event_former,
        semantic_projection=stage2_module.semantic_projection,
        local_change=heads.local_change,
        radar_transition=heads.radar_transition,
        semantic_difference=heads.semantic_difference,
        verify=heads.verify,
        update_period_s=heads.update_period_s,
        proposal_threshold=0.5,
        proposal_persistence=1,
        confirm_threshold=0.6,
        confirm_persistence=2,
        reject_threshold=0.3,
        reject_persistence=2,
        reject_timeout_s=float(stage3.get("reject_timeout_s", 1.8)),
        maturity_seconds=float(stage3.get("maturity_seconds", 1.2)),
        tau_stability=heads.tau_stability,
    )
    return PARradStage3Runtime(stage2_module.radar_encoder, controller)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA calibration was requested but CUDA is unavailable")
    return device


def run_closed_loop_calibration(
    config_path: Path,
    checkpoints: Mapping[str, Path] | None = None,
    *,
    output_dir: Path | None = None,
    device: str = "auto",
    grid: ThresholdGrid | None = None,
    max_recordings: int | None = None,
) -> ControllerThresholds:
    """Run a resumable validation-only sweep without opening the test manifest."""

    config_path = Path(config_path)
    raw = _load_yaml(config_path)
    stage3 = dict(raw.get("stage3", {}))
    stage3_output = Path(str(stage3["output_dir"]))
    if checkpoints is None:
        checkpoints = {
            "best_verify": stage3_output / "best_verify.pt",
            "last_verify": stage3_output / "last_verify.pt",
        }
    destination = stage3_output if output_dir is None else Path(output_dir)
    recordings = load_validation_recordings(config_path)
    if max_recordings is not None:
        if max_recordings <= 0:
            raise ValueError("max_recordings must be positive")
        recordings = recordings[:max_recordings]
    if not recordings:
        raise ValueError("validation split contains no recordings")
    validation_fingerprint = _validation_fingerprint(config_path, recordings)
    target_device = _resolve_device(device)
    current_path: Path | None = None
    current_runtime: PARradStage3Runtime | None = None
    details_dir = destination / "validation-controller-details"

    def evaluator(checkpoint: Path, thresholds: ControllerThresholds) -> Mapping[str, Any]:
        nonlocal current_path, current_runtime
        if current_path != checkpoint or current_runtime is None:
            current_runtime = None
            gc.collect()
            if target_device.type == "cuda":
                torch.cuda.empty_cache()
            current_runtime = build_runtime(config_path, checkpoint, target_device)
            current_path = checkpoint
        metrics = evaluate_controller(current_runtime, recordings, thresholds)
        _atomic_json(details_dir / f"{thresholds.fingerprint}.json", metrics)
        return metrics

    if grid is None:
        update_period_s = float(raw.get("model", {}).get("frame_period_s", 0.09)) * int(
            raw.get("model", {}).get("frames_per_update", 2)
        )
        grid = fixed_threshold_grid(
            update_period_s,
            float(stage3.get("reject_timeout_s", 1.8)),
        )
    return run_validation_sweep(
        checkpoints,
        grid,
        evaluator,
        destination,
        validation_fingerprint=validation_fingerprint,
    )


def _parse_checkpoints(values: Sequence[str]) -> dict[str, Path]:
    checkpoints: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--checkpoint must use NAME=PATH")
        name, path = value.split("=", 1)
        if not name or not path or name in checkpoints:
            raise ValueError("checkpoint names and paths must be non-empty and unique")
        checkpoints[name] = Path(path)
    return checkpoints


def _thresholds_from_artifact(path: Path) -> ControllerThresholds:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("status") != "calibrated_validation":
        raise ValueError("refinement requires a calibrated validation artifact")
    if payload.get("source_split") != "validation" or payload.get("test_evaluation_allowed") is not False:
        raise ValueError("refinement artifact is not validation-only")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise ValueError("refinement artifact contains no selected thresholds")
    return ControllerThresholds(**thresholds)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-recordings", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--grid-profile",
        choices=("full", "coarse", "refine"),
        default="full",
    )
    parser.add_argument("--selected-thresholds", type=Path)
    parser.add_argument("--proposal-shard-index", type=int)
    args = parser.parse_args(argv)
    checkpoints = _parse_checkpoints(args.checkpoint) if args.checkpoint else None
    output_dir = args.output_dir
    grid = None
    raw = _load_yaml(args.config)
    model = dict(raw.get("model", {}))
    stage3 = dict(raw.get("stage3", {}))
    update_period_s = float(model.get("frame_period_s", 0.09)) * int(
        model.get("frames_per_update", 2)
    )
    if args.smoke:
        if args.grid_profile != "full" or args.selected_thresholds is not None:
            parser.error("--smoke cannot be combined with another grid profile")
        grid = smoke_threshold_grid(
            update_period_s,
            float(stage3.get("reject_timeout_s", 1.8)),
        )
        if output_dir is None:
            output_dir = Path(str(stage3["output_dir"])) / "calibration-smoke"
    elif args.grid_profile == "coarse":
        grid = coarse_threshold_grid(
            update_period_s,
            float(stage3.get("reject_timeout_s", 1.8)),
        )
    elif args.grid_profile == "refine":
        if args.selected_thresholds is None:
            parser.error("--grid-profile refine requires --selected-thresholds")
        grid = refinement_threshold_grid(_thresholds_from_artifact(args.selected_thresholds))
    elif args.selected_thresholds is not None:
        parser.error("--selected-thresholds is only valid with --grid-profile refine")
    if args.proposal_shard_index is not None:
        if grid is None:
            grid = fixed_threshold_grid(
                update_period_s,
                float(stage3.get("reject_timeout_s", 1.8)),
            )
        shards = proposal_grid_shards(grid)
        if not 0 <= args.proposal_shard_index < len(shards):
            parser.error("--proposal-shard-index is outside the selected grid")
        grid = shards[args.proposal_shard_index]
    selected = run_closed_loop_calibration(
        args.config,
        checkpoints,
        output_dir=output_dir,
        device=args.device,
        grid=grid,
        max_recordings=args.max_recordings,
    )
    print(json.dumps(asdict(selected), sort_keys=True))


if __name__ == "__main__":
    main()
