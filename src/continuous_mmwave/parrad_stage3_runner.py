"""Independent three-phase trainer for PARrad Stage P3 transition heads."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.utils.data import DataLoader

from continuous_mmwave.parrad_stage3_data import (
    CacheFingerprint,
    PhysicalBatch,
    PhysicalSequenceBatch,
    PhysicalSequenceDataset,
    PhysicalTransitionDataset,
    SemanticPairBatch,
    SemanticPairDataset,
    Stage3CacheItem,
    VerifyBatch,
    VerifyFeatureBatch,
    VerifyTrajectoryDataset,
    collate_physical_batches,
    collate_physical_sequence_batches,
    collate_semantic_batches,
    collate_verify_batches,
    load_stage3_cache,
)
from continuous_mmwave.stages.parrad_stage3 import PARradStage3Module


@dataclass(frozen=True)
class Stage3TrainingConfig:
    cache_dir: Path
    stage2_checkpoint: Path
    output_dir: Path
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2
    epochs_physical: int = 30
    epochs_semantic: int = 30
    epochs_verify: int = 20
    early_stopping_patience: int = 5
    grad_clip_norm: float = 1.0
    device: str = "auto"
    amp: bool = True

    def validate(self, *, allow_existing_output: bool = False) -> None:
        if not self.cache_dir.is_dir() or not (self.cache_dir / "index.json").is_file():
            raise FileNotFoundError(f"Stage P3 cache index not found under {self.cache_dir}")
        if not self.stage2_checkpoint.is_file():
            raise FileNotFoundError(self.stage2_checkpoint)
        if self.batch_size <= 0 or self.learning_rate <= 0 or self.grad_clip_norm <= 0:
            raise ValueError("batch size, learning rate, and grad clip norm must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight decay must be non-negative")
        if min(self.epochs_physical, self.epochs_semantic, self.epochs_verify) <= 0:
            raise ValueError("all Stage P3 epoch counts must be positive")
        if self.early_stopping_patience <= 0:
            raise ValueError("early stopping patience must be positive")
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not allow_existing_output:
            raise FileExistsError(f"Stage P3 output directory is not new: {self.output_dir}")


@dataclass(frozen=True)
class PhaseTrainingResult:
    phase: Literal["physical", "semantic", "verify"]
    best_checkpoint: Path
    last_checkpoint: Path
    best_epoch: int
    best_metrics: Mapping[str, float]


@dataclass(frozen=True)
class Stage3TrainingResult:
    output_dir: Path
    phase_summaries: tuple[Mapping[str, Any], ...]
    final_checkpoint: Path
    cache_fingerprint: str


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_settings(payload: Mapping[str, Any], *, allow_existing_output: bool) -> Stage3TrainingConfig:
    values = dict(payload.get("stage3", {}))
    required = ("cache_dir", "stage2_checkpoint", "output_dir")
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"Stage P3 configuration is missing: {missing}")
    training = dict(payload.get("training", {}))
    settings = Stage3TrainingConfig(
        cache_dir=Path(values.pop("cache_dir")),
        stage2_checkpoint=Path(values.pop("stage2_checkpoint")),
        output_dir=Path(values.pop("output_dir")),
        device=str(values.pop("device", training.get("device", "auto"))),
        amp=bool(values.pop("amp", training.get("amp", True))),
        **{
            key: values[key]
            for key in asdict(Stage3TrainingConfig(Path("."), Path("."), Path(".")))
            if key in values and key not in required + ("device", "amp")
        },
    )
    settings.validate(allow_existing_output=allow_existing_output)
    return settings


def _load_cache_items(
    cache_dir: Path,
    stage2_checkpoint: Path,
    splits: tuple[str, ...] = ("train", "validation"),
    recording_keys: tuple[str, ...] | None = None,
) -> tuple[dict[str, tuple[Stage3CacheItem, ...]], str]:
    payload = json.loads((cache_dir / "index.json").read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported Stage P3 cache index schema")
    try:
        fingerprint = CacheFingerprint(**payload["fingerprint"])
    except (KeyError, TypeError) as error:
        raise ValueError("invalid Stage P3 cache fingerprint") from error
    if fingerprint.checkpoint_sha256 != _file_sha256(stage2_checkpoint):
        raise ValueError("Stage P3 cache fingerprint does not match the Stage P2 checkpoint")
    fingerprint_text = json.dumps(asdict(fingerprint), sort_keys=True, separators=(",", ":"))
    fingerprint_hash = hashlib.sha256(fingerprint_text.encode("utf-8")).hexdigest()
    grouped: dict[str, list[Stage3CacheItem]] = {}
    for row in payload.get("records", []):
        split, key = str(row["split"]), str(row["key"])
        if split not in splits:
            continue
        if recording_keys is not None and key not in recording_keys:
            continue
        root = cache_dir / split
        sidecar = json.loads((root / f"{key}.json").read_text(encoding="utf-8"))
        metadata = sidecar.get("metadata", {})
        record = load_stage3_cache(root, key, fingerprint)
        grouped.setdefault(split, []).append(
            Stage3CacheItem(split=split, key=key, metadata=metadata, record=record)
        )
    for required in splits:
        if not grouped.get(required):
            raise ValueError(f"Stage P3 cache has no {required} records")
    return {split: tuple(items) for split, items in grouped.items()}, fingerprint_hash


def _device(settings: Stage3TrainingConfig) -> torch.device:
    if settings.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(settings.device)


def _move_batch(
    batch: PhysicalBatch | PhysicalSequenceBatch | SemanticPairBatch | VerifyBatch | VerifyFeatureBatch,
    device: torch.device,
):
    return type(batch)(**{name: value.to(device) for name, value in asdict(batch).items()})


def _step(module: PARradStage3Module, phase: str, batch):
    if phase == "physical":
        return module.physical_step(batch)
    if phase == "semantic":
        return module.semantic_step(batch)
    if phase == "verify":
        return module.verify_step(batch)
    raise ValueError(f"unknown Stage P3 phase: {phase}")


def _binary_metrics(targets: Tensor, probabilities: Tensor) -> dict[str, float]:
    targets = targets.detach().float().cpu()
    probabilities = probabilities.detach().float().cpu()
    positives = targets >= 0.5
    negatives = ~positives
    predictions = probabilities >= 0.5
    tp = int((predictions & positives).sum())
    tn = int((~predictions & negatives).sum())
    fp = int((predictions & negatives).sum())
    fn = int((~predictions & positives).sum())
    tpr = tp / max(1, tp + fn)
    tnr = tn / max(1, tn + fp)
    if positives.any() and negatives.any():
        comparisons = probabilities[positives][:, None] - probabilities[negatives][None, :]
        auroc = float((comparisons.gt(0).float() + 0.5 * comparisons.eq(0).float()).mean())
    else:
        auroc = 0.5
    order = torch.argsort(probabilities, descending=True)
    sorted_positive = positives[order].float()
    cumulative_positive = sorted_positive.cumsum(0)
    precision = cumulative_positive / torch.arange(1, len(order) + 1)
    pr_auc = float((precision * sorted_positive).sum() / max(1, int(positives.sum())))
    cumulative_negative = (~positives[order]).float().cumsum(0)
    false_positive_rate = cumulative_negative / max(1, int(negatives.sum()))
    recall = cumulative_positive / max(1, int(positives.sum()))
    allowed_recall = recall[false_positive_rate <= 0.10]
    return {
        "pr_auc": pr_auc,
        "auroc": auroc,
        "balanced_accuracy": (tpr + tnr) / 2,
        "false_confirm_rate": fp / max(1, fp + tn),
        "recall_at_fpr_10": float(allowed_recall.max()) if allowed_recall.numel() else 0.0,
    }


@torch.inference_mode()
def _evaluate_phase(module, phase: str, loader: DataLoader, device: torch.device) -> dict[str, float]:
    module.eval()
    losses: list[float] = []
    probabilities: list[Tensor] = []
    local_probabilities: list[Tensor] = []
    radar_probabilities: list[Tensor] = []
    targets: list[Tensor] = []
    for raw in loader:
        batch = _move_batch(raw, device)
        metrics = _step(module, phase, batch)
        losses.append(float(metrics["loss"].detach()))
        if phase == "physical":
            local = torch.sigmoid(metrics["local_logits"])
            radar = torch.sigmoid(metrics["radar_logits"])
            valid = metrics.get("valid")
            target = batch.local_target
            if valid is not None:
                local = local[valid]
                radar = radar[valid]
                target = target[valid]
            local_probabilities.append(local)
            radar_probabilities.append(radar)
            probabilities.append(1 - (1 - local) * (1 - radar))
            targets.append(target)
        else:
            probabilities.append(torch.sigmoid(metrics["logits"]))
            targets.append(batch.target)
    if not losses:
        raise ValueError(f"{phase} validation loader is empty")
    result = _binary_metrics(torch.cat(targets), torch.cat(probabilities))
    if phase == "physical":
        for prefix, values in (
            ("local", local_probabilities),
            ("radar", radar_probabilities),
        ):
            component = _binary_metrics(torch.cat(targets), torch.cat(values))
            result.update({f"{prefix}_{name}": value for name, value in component.items()})
    result["loss"] = sum(losses) / len(losses)
    return result


def _score(phase: str, metrics: Mapping[str, float]) -> tuple[float, ...]:
    if phase == "physical":
        return (
            float(metrics["recall_at_fpr_10"]),
            float(metrics["pr_auc"]),
            -float(metrics["loss"]),
        )
    if phase == "semantic":
        return float(metrics["auroc"]), float(metrics["balanced_accuracy"])
    safe = float(metrics["false_confirm_rate"]) <= 0.10
    return (
        float(safe),
        float(metrics["pr_auc"]) if safe else -float(metrics["false_confirm_rate"]),
        float(metrics["balanced_accuracy"]),
        float(metrics["auroc"]),
    )


def _save_checkpoint(
    path: Path,
    module: PARradStage3Module,
    optimizer: torch.optim.Optimizer,
    *,
    phase: str,
    epoch: int,
    step: int,
    cache_fingerprint: str,
    metrics: Mapping[str, float],
) -> None:
    torch.save(
        {
            "schema_version": 1,
            "phase": phase,
            "epoch": epoch,
            "step": step,
            "cache_fingerprint": cache_fingerprint,
            "model_state": module.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "validation_metrics": dict(metrics),
        },
        path,
    )


def train_stage3_phase(
    module: PARradStage3Module,
    phase: Literal["physical", "semantic", "verify"],
    train_loader: DataLoader,
    validation_loader: DataLoader,
    settings: Stage3TrainingConfig,
    cache_fingerprint: str,
    resume: Path | None = None,
    max_steps: int | None = None,
) -> PhaseTrainingResult:
    module.set_phase(phase)
    device = _device(settings)
    module.to(device)
    parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings.learning_rate, weight_decay=settings.weight_decay)
    start_epoch = 0
    step = 0
    if resume is not None:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        if payload.get("phase") != phase or payload.get("cache_fingerprint") != cache_fingerprint:
            raise ValueError("resume phase or cache fingerprint mismatch")
        module.load_state_dict(payload["model_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        start_epoch = int(payload["epoch"]) + 1
        step = int(payload["step"])
    epoch_count = getattr(settings, f"epochs_{phase}")
    best_score: tuple[float, ...] | None = None
    best_epoch = -1
    best_metrics: Mapping[str, float] = {}
    stale_epochs = 0
    best_path = settings.output_dir / f"best_{phase}.pt"
    last_path = settings.output_dir / f"last_{phase}.pt"
    metrics_path = settings.output_dir / f"metrics-{phase}.jsonl"
    validation_path = settings.output_dir / f"validation-{phase}.jsonl"
    for epoch in range(start_epoch, epoch_count):
        module.train()
        for raw in train_loader:
            batch = _move_batch(raw, device)
            optimizer.zero_grad(set_to_none=True)
            amp_enabled = settings.amp and device.type == "cuda"
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                metrics = _step(module, phase, batch)
            metrics["loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, settings.grad_clip_norm)
            optimizer.step()
            step += 1
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"phase": phase, "epoch": epoch, "step": step, "loss": float(metrics["loss"].detach())}) + "\n")
            if max_steps is not None and step >= max_steps:
                break
        validation = _evaluate_phase(module, phase, validation_loader, device)
        with validation_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"phase": phase, "epoch": epoch, "step": step, **validation}) + "\n")
        current_score = _score(phase, validation)
        improved = best_score is None or current_score > best_score
        if improved:
            best_score = current_score
            best_epoch = epoch
            best_metrics = validation
            stale_epochs = 0
            _save_checkpoint(
                best_path, module, optimizer, phase=phase, epoch=epoch, step=step,
                cache_fingerprint=cache_fingerprint, metrics=validation,
            )
        else:
            stale_epochs += 1
        _save_checkpoint(
            last_path, module, optimizer, phase=phase, epoch=epoch, step=step,
            cache_fingerprint=cache_fingerprint, metrics=validation,
        )
        if (max_steps is not None and step >= max_steps) or stale_epochs >= settings.early_stopping_patience:
            break
    if not best_path.is_file():
        raise RuntimeError(f"{phase} phase produced no validation checkpoint")
    module.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state"])
    return PhaseTrainingResult(phase, best_path, last_path, best_epoch, best_metrics)


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ValueError("Stage P3 config root must be a mapping")
    return payload


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _prepare_cache_if_missing(payload: Mapping[str, Any], config_path: Path) -> None:
    stage3 = dict(payload.get("stage3", {}))
    cache_value = stage3.get("cache_dir")
    checkpoint_value = stage3.get("stage2_checkpoint")
    if not cache_value or not checkpoint_value:
        return
    cache_dir = Path(cache_value)
    if (cache_dir / "index.json").is_file():
        return
    from continuous_mmwave.parrad_stage3_data import audit_stage3_manifests, build_stage3_cache
    from continuous_mmwave.train import load_experiment_config

    experiment = load_experiment_config(config_path)
    manifests = {
        split: Path(experiment.data[f"{split}_manifest"])
        for split in ("train", "validation", "test")
    }
    audits = audit_stage3_manifests(manifests, experiment.model)
    audit_path = stage3.get("audit_path")
    if audit_path:
        _atomic_json(
            Path(audit_path),
            {split: asdict(audit) for split, audit in audits.items()},
        )
    build_stage3_cache(experiment, Path(checkpoint_value), cache_dir)


def run_parrad_stage3(
    config_path: Path,
    resume: Path | None = None,
    max_steps_per_phase: int | None = None,
    evaluate_test: bool = False,
) -> Stage3TrainingResult:
    config_path = Path(config_path)
    payload = _load_yaml(config_path)
    _prepare_cache_if_missing(payload, config_path)
    settings = _parse_settings(payload, allow_existing_output=resume is not None)
    random.seed(settings.seed)
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    records, cache_fingerprint = _load_cache_items(settings.cache_dir, settings.stage2_checkpoint)
    settings.output_dir.mkdir(parents=True, exist_ok=resume is not None)
    model = dict(payload.get("model", {}))
    stage3 = dict(payload.get("stage3", {}))
    module = PARradStage3Module(
        radar_dim=int(model.get("radar_dim", 256)),
        semantic_dim=int(model.get("semantic_dim", 128)),
        prototype_dim=int(stage3.get("prototype_dim", 128)),
        update_period_s=float(model.get("frame_period_s", 0.09)) * int(model.get("frames_per_update", 2)),
        tau_fast_s=float(stage3.get("tau_fast_s", 2.0)),
        tau_stability=float(stage3.get("tau_stability", 0.1)),
        physical_tbptt_steps=int(stage3.get("physical_tbptt_steps", 32)),
    )
    physical_train = PhysicalSequenceDataset(
        records["train"], float(stage3.get("sigma_boundary_s", 0.36))
    )
    physical_validation = PhysicalSequenceDataset(
        records["validation"],
        float(stage3.get("sigma_boundary_s", 0.36)),
        positive_weight=physical_train.positive_weight,
    )
    datasets = {
        "physical": (
            physical_train,
            physical_validation,
            collate_physical_sequence_batches,
        ),
        "semantic": (
            SemanticPairDataset(records["train"], settings.seed),
            SemanticPairDataset(records["validation"], settings.seed),
            collate_semantic_batches,
        ),
        "verify": (
            VerifyTrajectoryDataset(records["train"], settings.seed),
            VerifyTrajectoryDataset(records["validation"], settings.seed),
            collate_verify_batches,
        ),
    }
    generator = torch.Generator().manual_seed(settings.seed)
    resume_phase = None
    if resume is not None:
        resume_phase = torch.load(resume, map_location="cpu", weights_only=False).get("phase")
    summaries: list[Mapping[str, Any]] = []
    for phase in ("physical", "semantic", "verify"):
        train_dataset, validation_dataset, collate = datasets[phase]
        train_loader = DataLoader(
            train_dataset,
            batch_size=settings.batch_size,
            shuffle=True,
            generator=generator,
            collate_fn=collate,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            collate_fn=collate,
        )
        result = train_stage3_phase(
            module,
            phase,
            train_loader,
            validation_loader,
            settings,
            cache_fingerprint,
            resume=resume if phase == resume_phase else None,
            max_steps=max_steps_per_phase,
        )
        summaries.append(
            {
                "phase": phase,
                "best_epoch": result.best_epoch,
                "best_checkpoint": str(result.best_checkpoint),
                **result.best_metrics,
            }
        )
    result = Stage3TrainingResult(
        output_dir=settings.output_dir,
        phase_summaries=tuple(summaries),
        final_checkpoint=settings.output_dir / "best_verify.pt",
        cache_fingerprint=cache_fingerprint,
    )
    threshold_path = settings.output_dir / "validation_thresholds.json"
    if not threshold_path.exists():
        _atomic_json(
            threshold_path,
            {
                "schema_version": 1,
                "status": "pending_closed_loop_validation",
                "source_split": "validation",
                "test_evaluation_allowed": False,
                "cache_fingerprint": cache_fingerprint,
                "stage3_checkpoint_sha256": _file_sha256(result.final_checkpoint),
                "reason": "head training is complete; full-recording controller threshold sweep is still required",
                "phase_summaries": list(result.phase_summaries),
            },
        )
    if evaluate_test:
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
        if threshold_payload.get("status") != "calibrated" or not threshold_payload.get("test_evaluation_allowed"):
            raise RuntimeError("test evaluation requires frozen closed-loop validation thresholds")
        raise RuntimeError("test evaluation must be run through the explicit test-once diagnostics entrypoint")
    return result
