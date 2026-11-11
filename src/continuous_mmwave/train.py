"""CLI for v2.2 Stage 1 and Stage 2 training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import yaml
from torch import Tensor
from torch.nn import functional
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from continuous_mmwave.checkpoint import (
    load_stage1_radar_state,
    load_training_checkpoint,
    save_stage1_radar_export,
    save_training_checkpoint,
)
from continuous_mmwave.config import FIXED_EVENT_PROMPT, ModelConfig
from continuous_mmwave.data import ManifestDataset, RadarBatch, radar_collate
from continuous_mmwave.data_parrad import (
    PARradBatch,
    PARradManifestDataset,
    PARradSample,
    PARradWindowDataset,
    assert_disjoint_splits,
    build_sensor_vocabulary,
    mask_label_boundaries,
    parrad_collate,
)
from continuous_mmwave.models.event_former import (
    Gemma3Adapter,
    Qwen2Adapter,
    StreamingSemanticEventFormer,
    TinyLLMAdapter,
)
from continuous_mmwave.models.radar_encoder import StreamingRadarEncoder
from continuous_mmwave.models.parrad_encoder import (
    PARradAnyStreamState,
    build_parrad_encoder,
    detach_parrad_stream_state,
)
from continuous_mmwave.models.semantic import SemanticProjection
from continuous_mmwave.parrad_config import PARradConfig
from continuous_mmwave.stages.parrad_stage1 import (
    PARradStage1Module,
    token_classification_targets,
)
from continuous_mmwave.stages.parrad_stage2 import (
    PARradStage2Module,
    default_parrad_event_templates,
    parrad_event_targets,
    select_event_captions,
    set_parrad_stage2_phase,
)
from continuous_mmwave.stages.stage1 import Stage1Module
from continuous_mmwave.stages.stage2 import Stage2Module


@dataclass(frozen=True)
class ExperimentConfig:
    """一份完整实验配置：模型超参与四个松散分组的 dict，直接映射 YAML 顶层键。"""

    model: ModelConfig | PARradConfig
    data: dict[str, Any]
    training: dict[str, Any]
    llm: dict[str, Any]
    stage: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """序列化后存入 checkpoint，用于断点恢复时校验配置未变。"""
        return {
            "model": self.model.to_dict(),
            "data": self.data,
            "training": self.training,
            "llm": self.llm,
            "stage": self.stage,
        }


class WarmupCosineSchedule:
    """Small per-update warmup/cosine scheduler with an explicit phase LR."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        base_lr: float,
        total_steps: int,
        warmup_ratio: float,
        min_lr_ratio: float,
    ) -> None:
        if base_lr <= 0 or total_steps <= 0:
            raise ValueError("base_lr and total_steps must be positive")
        if not 0 <= warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 <= min_lr_ratio <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = min(total_steps - 1, max(0, round(total_steps * warmup_ratio)))
        self.min_lr_ratio = min_lr_ratio
        self.step_count = 0

    def step(self) -> float:
        self.step_count += 1
        if self.warmup_steps and self.step_count <= self.warmup_steps:
            factor = self.step_count / self.warmup_steps
        else:
            decay_steps = max(1, self.total_steps - self.warmup_steps)
            progress = min(1.0, (self.step_count - self.warmup_steps) / decay_steps)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine
        learning_rate = self.base_lr * factor
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        return learning_rate


@dataclass
class PhaseEarlyStopping:
    """Track validation improvements independently inside one training phase."""

    patience: int
    best_score: tuple[float, float, float] | None = None
    non_improving_epochs: int = 0

    def __post_init__(self) -> None:
        if self.patience <= 0:
            raise ValueError("patience must be positive")

    @staticmethod
    def score(*, accuracy: float, loss: float, macro_f1: float | None = None) -> tuple[float, float, float]:
        primary = accuracy if macro_f1 is None else macro_f1
        return primary, accuracy, -loss

    def update(
        self,
        *,
        accuracy: float,
        loss: float,
        macro_f1: float | None = None,
    ) -> tuple[bool, bool]:
        score = self.score(accuracy=accuracy, loss=loss, macro_f1=macro_f1)
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.best_score = score
            self.non_improving_epochs = 0
        else:
            self.non_improving_epochs += 1
        return improved, self.non_improving_epochs >= self.patience


def _parrad_stage2_phase(epoch: int, warmup_epochs: int) -> str:
    if epoch < 0 or warmup_epochs < 0:
        raise ValueError("epoch and semantic_warmup_epochs must be non-negative")
    return "semantic_warmup" if epoch < warmup_epochs else "joint"


def _parrad_stage2_text_weight(
    epoch: int,
    warmup_epochs: int,
    ramp_epochs: int,
    final_weight: float,
) -> float:
    if ramp_epochs < 0 or final_weight < 0:
        raise ValueError("caption ramp and final weight must be non-negative")
    if _parrad_stage2_phase(epoch, warmup_epochs) == "semantic_warmup":
        return 0.0
    if ramp_epochs == 0:
        return final_weight
    joint_epoch = epoch - warmup_epochs
    return final_weight * min((joint_epoch + 1) / ramp_epochs, 1.0)

def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    groups = {key: dict(raw.get(key, {})) for key in ("data", "training", "llm", "stage")}
    model_values = dict(raw.get("model", {}))
    variant = str(groups["stage"].get("variant", "default"))
    if variant == "parrad":
        for key in ("rd_tcn_dilations", "md_tcn_dilations"):
            if key in model_values:
                model_values[key] = tuple(model_values[key])
        model = PARradConfig(**model_values)
    elif variant == "default":
        if "tcn_dilations" in model_values:
            # YAML 只能给出 list，需转回 ModelConfig 要求的 tuple。
            model_values["tcn_dilations"] = tuple(model_values["tcn_dilations"])
        model = ModelConfig(**model_values)
    else:
        raise ValueError(f"unsupported training variant: {variant}")
    return ExperimentConfig(model, groups["data"], groups["training"], groups["llm"], groups["stage"])


def seed_everything(seed: int) -> None:
    """统一播种四个随机源，保证实验可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parameter_report(module: torch.nn.Module) -> dict[str, Any]:
    """记录总参数量与可训练参数量，训练开始时写盘以便核对冻结范围。"""
    named = list(module.named_parameters())
    return {
        "total_parameters": sum(p.numel() for _, p in named),
        "trainable_parameters": sum(p.numel() for _, p in named if p.requires_grad),
        "trainable_names": [name for name, p in named if p.requires_grad],
    }


def _step(
    loss: Tensor,
    optimizer: torch.optim.Optimizer,
    parameters: Iterable[Tensor],
    clip: float,
    scaler: torch.amp.GradScaler | None,
) -> bool:
    """反向传播、裁剪并返回优化器是否真正执行了参数更新。"""
    parameters = list(parameters)
    if scaler is None:
        loss.backward()
        clip_grad_norm_(parameters, clip)
        optimizer.step()
        return True
    else:
        scale_before = scaler.get_scale()
        scaler.scale(loss).backward()
        # 先 unscale_ 再裁剪，否则裁剪阈值作用在放大后的梯度上。
        scaler.unscale_(optimizer)
        clip_grad_norm_(parameters, clip)
        scaler.step(optimizer)
        scaler.update()
        return scaler.get_scale() >= scale_before


def train_stage1_batch(
    module: Stage1Module,
    batch: RadarBatch,
    optimizer: torch.optim.Optimizer,
    clip: float,
    *,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """单个 Stage 1 训练步，返回写入 metrics 的标量指标。"""
    module.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(batch.rd.device.type, enabled=amp_enabled):
        losses = module.training_step(batch)
    _step(losses.total, optimizer, (p for p in module.parameters() if p.requires_grad), clip, scaler)
    return {
        "loss": float(losses.total.detach()),
        "mpjpe": float(losses.mpjpe.detach()),
        "velocity": float(losses.velocity.detach()),
    }


def train_parrad_stage1_batch(
    module: PARradStage1Module,
    batch: PARradBatch,
    optimizer: torch.optim.Optimizer,
    clip: float,
    *,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Run one bounded PARrad activity-classification update."""
    module.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(batch.rd.device.type, enabled=amp_enabled):
        losses = module.training_step(batch)
    _step(losses.loss, optimizer, (p for p in module.parameters() if p.requires_grad), clip, scaler)
    return {
        "loss": float(losses.loss.detach()),
        "classification_loss": float(losses.classification_loss.detach()),
        "token_classification_loss": float(losses.token_classification_loss.detach()),
        "temporal_smoothness_loss": float(losses.temporal_smoothness_loss.detach()),
        "temporal_smoothness_weight": module.temporal_smoothness_weight,
        "frame_classification_weight": module.frame_classification_weight,
        "token_classification_weight": module.token_classification_weight,
        "accuracy": float(losses.accuracy.detach()),
        "token_accuracy": float(losses.token_accuracy.detach()),
        "supervised_tokens": float(losses.supervised_tokens.detach()),
    }


def _slice_parrad_batch(batch: PARradBatch, start: int, stop: int) -> PARradBatch:
    return PARradBatch(
        rd=batch.rd[:, start:stop],
        md=batch.md[:, start:stop],
        labels=batch.labels[:, start:stop],
        timestamps=batch.timestamps[:, start:stop],
        frame_mask=batch.frame_mask[:, start:stop],
        sensor_index=batch.sensor_index,
        metadata=batch.metadata,
    )


def train_parrad_streaming_batch(
    module: PARradStage1Module,
    batch: PARradBatch,
    optimizer: torch.optim.Optimizer,
    clip: float,
    *,
    chunk_frames: int,
    device: torch.device,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
    max_updates: int | None = None,
    initial_state: PARradAnyStreamState | None = None,
    return_state: bool = False,
    before_optimizer_step: Callable[[], float] | None = None,
) -> dict[str, float | int] | tuple[dict[str, float | int], PARradAnyStreamState]:
    """Train consecutive chunks while carrying detached TCN state between updates."""
    step = module.encoder.config.frames_per_update
    if chunk_frames <= 0 or chunk_frames % step:
        raise ValueError("chunk_frames must be a positive multiple of frames_per_update")
    module.train()
    state = initial_state
    loss_sum = 0.0
    classification_loss_sum = 0.0
    temporal_smoothness_loss_sum = 0.0
    token_classification_loss_sum = 0.0
    correct = 0
    token_correct = 0.0
    frames = 0
    supervised_tokens = 0
    chunks = 0
    optimizer_steps = 0
    learning_rate = float(optimizer.param_groups[0]["lr"])
    for start in range(0, batch.rd.shape[1], chunk_frames):
        chunk = _move_parrad(_slice_parrad_batch(batch, start, start + chunk_frames), device)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            logits, output = module(chunk, state=state)
        state = detach_parrad_stream_state(output.state)
        chunks += 1
        valid = chunk.labels.ne(-100)
        valid_frames = int(valid.sum().item())
        if valid_frames == 0:
            continue
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            token_logits = module.token_logits(output)
            token_labels = token_classification_targets(chunk.labels, step)
            losses = module.loss_from_logits(logits, chunk.labels, token_logits, token_labels)
        optimizer.zero_grad(set_to_none=True)
        if before_optimizer_step is not None:
            learning_rate = before_optimizer_step()
        _step(losses.loss, optimizer, (p for p in module.parameters() if p.requires_grad), clip, scaler)
        loss_sum += float(losses.loss.detach()) * valid_frames
        classification_loss_sum += float(losses.classification_loss.detach()) * valid_frames
        temporal_smoothness_loss_sum += float(losses.temporal_smoothness_loss.detach()) * valid_frames
        chunk_tokens = int(losses.supervised_tokens.detach())
        token_classification_loss_sum += float(losses.token_classification_loss.detach()) * chunk_tokens
        token_correct += float(losses.token_accuracy.detach()) * chunk_tokens
        correct += int((logits.argmax(-1)[valid] == chunk.labels[valid]).sum().item())
        frames += valid_frames
        supervised_tokens += chunk_tokens
        optimizer_steps += 1
        if max_updates is not None and optimizer_steps >= max_updates:
            break
    if frames == 0:
        raise ValueError("streaming batch requires at least one supervised frame")
    metrics: dict[str, float | int] = {
        "loss": loss_sum / frames,
        "classification_loss": classification_loss_sum / frames,
        "token_classification_loss": (
            token_classification_loss_sum / supervised_tokens if supervised_tokens else 0.0
        ),
        "temporal_smoothness_loss": temporal_smoothness_loss_sum / frames,
        "temporal_smoothness_weight": module.temporal_smoothness_weight,
        "frame_classification_weight": module.frame_classification_weight,
        "token_classification_weight": module.token_classification_weight,
        "accuracy": correct / frames,
        "token_accuracy": token_correct / supervised_tokens if supervised_tokens else 0.0,
        "supervised_tokens": supervised_tokens,
        "chunks": chunks,
        "optimizer_steps": optimizer_steps,
        "learning_rate": learning_rate,
    }
    if state is None:
        raise RuntimeError("streaming batch produced no encoder state")
    return (metrics, state) if return_state else metrics


def _collate_recording_chunk(
    samples: tuple[PARradSample, ...],
    labels: tuple[Tensor, ...],
    start: int,
    chunk_frames: int,
) -> PARradBatch:
    batch = len(samples)
    first = samples[0]
    rd = first.rd.new_zeros((batch, chunk_frames, *first.rd.shape[1:]))
    md = first.md.new_zeros((batch, chunk_frames, first.md.shape[1]))
    batch_labels = torch.full((batch, chunk_frames), -100, dtype=torch.long)
    timestamps = first.timestamps.new_zeros((batch, chunk_frames))
    frame_mask = torch.zeros((batch, chunk_frames), dtype=torch.bool)
    for index, sample in enumerate(samples):
        stop = min(start + chunk_frames, sample.rd.shape[0])
        length = max(0, stop - start)
        if length == 0:
            continue
        rd[index, :length] = sample.rd[start:stop]
        md[index, :length] = sample.md[start:stop]
        batch_labels[index, :length] = labels[index][start:stop]
        timestamps[index, :length] = sample.timestamps[start:stop]
        frame_mask[index, :length] = True
    return PARradBatch(
        rd=rd,
        md=md,
        labels=batch_labels,
        timestamps=timestamps,
        frame_mask=frame_mask,
        sensor_index=torch.tensor([sample.sensor_index for sample in samples], dtype=torch.long),
        metadata=tuple(
            {
                "subject_id": sample.subject_id,
                "session_id": sample.session_id,
                "sensor_id": sample.sensor_id,
                "environment_id": sample.environment_id,
            }
            for sample in samples
        ),
    )


def _iter_parrad_recording_sequences(
    recordings: tuple[PARradSample, ...],
    *,
    sequence_frames: int,
    recording_batch_size: int,
    boundary_ignore_frames: int,
    order: Sequence[int] | None = None,
) -> Iterator[tuple[bool, PARradBatch]]:
    """Yield ordered recording slices and mark where recurrent state must reset."""
    if sequence_frames <= 0:
        raise ValueError("sequence_frames must be positive")
    if recording_batch_size <= 0:
        raise ValueError("recording_batch_size must be positive")
    indices = tuple(torch.randperm(len(recordings)).tolist()) if order is None else tuple(order)
    if sorted(indices) != list(range(len(recordings))):
        raise ValueError("order must contain every recording index exactly once")
    ordered = tuple(recordings[index] for index in indices)
    for group_start in range(0, len(ordered), recording_batch_size):
        group = ordered[group_start : group_start + recording_batch_size]
        masked_labels = tuple(
            mask_label_boundaries(sample.labels, boundary_ignore_frames) for sample in group
        )
        max_frames = max(sample.rd.shape[0] for sample in group)
        for start in range(0, max_frames, sequence_frames):
            yield start == 0, _collate_recording_chunk(
                group,
                masked_labels,
                start,
                sequence_frames,
            )


def _add_confusion(confusion: Tensor, predictions: Tensor, targets: Tensor) -> None:
    if targets.numel() == 0:
        return
    classes = confusion.shape[0]
    counts = torch.bincount(
        (targets.to(torch.long).cpu() * classes + predictions.to(torch.long).cpu()),
        minlength=classes * classes,
    )
    confusion += counts.reshape(classes, classes)


def _classification_metrics(confusion: Tensor) -> tuple[float, float, dict[str, float]]:
    values = confusion.to(torch.float64)
    support = values.sum(dim=1)
    predicted = values.sum(dim=0)
    true_positive = values.diag()
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator))
    present = support > 0
    macro_recall = float(recall[present].mean()) if present.any() else 0.0
    macro_f1 = float(f1[present].mean()) if present.any() else 0.0
    return macro_recall, macro_f1, {str(index): float(value) for index, value in enumerate(recall)}


def _confusion_summary(confusion: Tensor) -> dict[str, Any]:
    values = confusion.to(torch.float64)
    support = values.sum(dim=1)
    predicted = values.sum(dim=0)
    true_positive = values.diag()
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    denominator = precision + recall
    f1 = torch.where(denominator > 0, 2 * precision * recall / denominator, torch.zeros_like(denominator))
    present = support > 0
    total = float(support.sum())
    return {
        "frames": total,
        "accuracy": float(true_positive.sum()) / total if total else 0.0,
        "macro_recall": float(recall[present].mean()) if present.any() else 0.0,
        "macro_f1": float(f1[present].mean()) if present.any() else 0.0,
        "per_class_metrics": {
            str(index): {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": float(support[index]),
            }
            for index in range(confusion.shape[0])
        },
    }


def _sequence_diagnostics(
    *,
    predictions: tuple[Tensor, ...],
    targets: tuple[Tensor, ...],
    metadata: tuple[dict[str, Any], ...],
    num_classes: int,
    boundary_tolerance_frames: int,
) -> dict[str, Any]:
    """Summarize class confusion, domain groups, and temporal boundary stability."""
    if not (len(predictions) == len(targets) == len(metadata)):
        raise ValueError("predictions, targets, and metadata must have the same length")
    if boundary_tolerance_frames < 0:
        raise ValueError("boundary_tolerance_frames must be non-negative")
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    grouped: dict[str, dict[str, Tensor]] = {
        "subject": {},
        "sensor": {},
        "recording": {},
    }
    true_boundary_count = 0
    predicted_boundary_count = 0
    matched_boundary_count = 0
    boundary_errors: list[int] = []
    within_segment_switches = 0
    within_segment_pairs = 0
    for prediction, target, item_metadata in zip(predictions, targets, metadata, strict=True):
        if prediction.shape != target.shape or prediction.ndim != 1:
            raise ValueError("each prediction/target pair must be one-dimensional and shape-matched")
        prediction = prediction.to(torch.long).cpu()
        target = target.to(torch.long).cpu()
        valid = target.ne(-100)
        _add_confusion(confusion, prediction[valid], target[valid])
        group_keys = {
            "subject": str(item_metadata["subject_id"]),
            "sensor": str(item_metadata["sensor_id"]),
            "recording": "/".join(
                str(item_metadata[key]) for key in ("subject_id", "session_id", "sensor_id")
            ),
        }
        for group_name, key in group_keys.items():
            group_confusion = grouped[group_name].setdefault(
                key, torch.zeros_like(confusion)
            )
            _add_confusion(group_confusion, prediction[valid], target[valid])

        adjacent_valid = valid[1:] & valid[:-1]
        same_true_segment = adjacent_valid & target[1:].eq(target[:-1])
        within_segment_pairs += int(same_true_segment.sum())
        within_segment_switches += int(
            ((prediction[1:] != prediction[:-1]) & same_true_segment).sum()
        )
        true_boundaries: list[int] = []
        seen_valid = bool(valid[0])
        for index in range(1, target.numel()):
            if not bool(valid[index]):
                continue
            if seen_valid and (
                not bool(valid[index - 1]) or int(target[index]) != int(target[index - 1])
            ):
                true_boundaries.append(index)
            seen_valid = True
        predicted_boundaries = (
            prediction[1:] != prediction[:-1]
        ).nonzero().flatten().add(1).tolist()
        true_boundary_count += len(true_boundaries)
        predicted_boundary_count += len(predicted_boundaries)
        available = set(range(len(predicted_boundaries)))
        for boundary in true_boundaries:
            if predicted_boundaries:
                boundary_errors.append(min(abs(boundary - candidate) for candidate in predicted_boundaries))
            else:
                boundary_errors.append(int(target.numel()))
            candidates = [
                index
                for index in available
                if abs(boundary - predicted_boundaries[index]) <= boundary_tolerance_frames
            ]
            if candidates:
                selected = min(candidates, key=lambda index: abs(boundary - predicted_boundaries[index]))
                available.remove(selected)
                matched_boundary_count += 1

    boundary_precision = (
        matched_boundary_count / predicted_boundary_count if predicted_boundary_count else 0.0
    )
    boundary_recall = matched_boundary_count / true_boundary_count if true_boundary_count else 0.0
    boundary_f1 = (
        2 * boundary_precision * boundary_recall / (boundary_precision + boundary_recall)
        if boundary_precision + boundary_recall
        else 0.0
    )
    summary = _confusion_summary(confusion)
    return {
        "confusion_matrix": confusion.tolist(),
        "per_class_metrics": summary["per_class_metrics"],
        "temporal": {
            "true_boundaries": float(true_boundary_count),
            "predicted_boundaries": float(predicted_boundary_count),
            "matched_boundaries": float(matched_boundary_count),
            "boundary_precision": boundary_precision,
            "boundary_recall": boundary_recall,
            "boundary_f1": boundary_f1,
            "boundary_mae_frames": (
                sum(boundary_errors) / len(boundary_errors) if boundary_errors else 0.0
            ),
            "switch_excess_ratio": (
                predicted_boundary_count / true_boundary_count if true_boundary_count else 0.0
            ),
            "within_segment_switches": float(within_segment_switches),
            "within_segment_switch_rate": (
                within_segment_switches / within_segment_pairs if within_segment_pairs else 0.0
            ),
        },
        "groups": {
            group_name: {
                key: {
                    metric: value
                    for metric, value in _confusion_summary(group_confusion).items()
                    if metric != "per_class_metrics"
                }
                for key, group_confusion in values.items()
            }
            for group_name, values in grouped.items()
        },
    }


@torch.no_grad()
def evaluate_parrad_stage1_streaming(
    module: PARradStage1Module,
    recordings: tuple[PARradSample, ...],
    device: torch.device,
    *,
    chunk_frames: int,
    recording_batch_size: int,
    boundary_ignore_frames: int = 0,
    boundary_tolerance_frames: int = 2,
    amp_enabled: bool = False,
) -> dict[str, Any]:
    """Evaluate continuous recordings without using labels to choose chunk boundaries."""
    step = module.encoder.config.frames_per_update
    if chunk_frames <= 0 or chunk_frames % step:
        raise ValueError("chunk_frames must be a positive multiple of frames_per_update")
    if recording_batch_size <= 0:
        raise ValueError("recording_batch_size must be positive")
    module.eval()
    loss_sum = 0.0
    loss_denominator = 0.0
    token_loss_sum = 0.0
    token_loss_denominator = 0.0
    correct = 0
    frames = 0
    num_classes = module.classifier.out_features
    trimmed_confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)
    token_confusion = torch.zeros_like(trimmed_confusion)
    raw_confusion = torch.zeros_like(trimmed_confusion)
    raw_correct = 0
    raw_frames = 0
    segment_correct = 0
    segments = 0
    sensor_counts: dict[str, list[int]] = {}
    diagnostic_predictions: list[Tensor] = []
    diagnostic_targets: list[Tensor] = []
    diagnostic_metadata: list[dict[str, Any]] = []
    for group_start in range(0, len(recordings), recording_batch_size):
        group = recordings[group_start : group_start + recording_batch_size]
        masked_labels = tuple(
            mask_label_boundaries(sample.labels, boundary_ignore_frames) for sample in group
        )
        state: PARradAnyStreamState | None = None
        group_predictions: list[list[Tensor]] = [[] for _ in group]
        max_frames = max(sample.rd.shape[0] for sample in group)
        for start in range(0, max_frames, chunk_frames):
            batch = _move_parrad(
                _collate_recording_chunk(group, masked_labels, start, chunk_frames),
                device,
            )
            with torch.amp.autocast(device.type, enabled=amp_enabled):
                logits, output = module(batch, state=state)
                loss_numerator = functional.cross_entropy(
                    logits.flatten(0, 1),
                    batch.labels.flatten(),
                    weight=module.class_weights,
                    ignore_index=-100,
                    reduction="sum",
                )
                token_logits = module.token_logits(output)
                token_labels = token_classification_targets(batch.labels, step)
                token_loss_numerator = functional.cross_entropy(
                    token_logits.flatten(0, 1),
                    token_labels.flatten(),
                    weight=module.class_weights,
                    ignore_index=-100,
                    reduction="sum",
                )
            state = output.state
            predictions = logits.argmax(-1).cpu()
            for index, sample in enumerate(group):
                length = min(chunk_frames, max(0, sample.rd.shape[0] - start))
                if length == 0:
                    continue
                sample_predictions = predictions[index, :length]
                group_predictions[index].append(sample_predictions)
                raw_targets = sample.labels[start : start + length]
                raw_valid = raw_targets.ne(-100)
                _add_confusion(raw_confusion, sample_predictions[raw_valid], raw_targets[raw_valid])
                raw_correct += int((sample_predictions[raw_valid] == raw_targets[raw_valid]).sum().item())
                raw_frames += int(raw_valid.sum().item())
                trimmed_targets = batch.labels[index, :length].cpu()
                trimmed_valid = trimmed_targets.ne(-100)
                sensor_correct, sensor_total = sensor_counts.setdefault(sample.sensor_id, [0, 0])
                sensor_counts[sample.sensor_id] = [
                    sensor_correct
                    + int((sample_predictions[trimmed_valid] == trimmed_targets[trimmed_valid]).sum().item()),
                    sensor_total + int(trimmed_valid.sum().item()),
                ]
            valid = batch.labels.ne(-100)
            valid_frames = int(valid.sum().item())
            if valid_frames == 0:
                continue
            loss_sum += float(loss_numerator.detach())
            if module.class_weights is None:
                loss_denominator += valid_frames
            else:
                loss_denominator += float(module.class_weights[batch.labels[valid]].sum().detach())
            batch_predictions = logits.argmax(-1)
            correct += int((batch_predictions[valid] == batch.labels[valid]).sum().item())
            _add_confusion(
                trimmed_confusion,
                batch_predictions[valid],
                batch.labels[valid],
            )
            frames += valid_frames
            token_valid = token_labels.ne(-100)
            if token_valid.any():
                token_loss_sum += float(token_loss_numerator.detach())
                if module.class_weights is None:
                    token_loss_denominator += int(token_valid.sum())
                else:
                    token_loss_denominator += float(
                        module.class_weights[token_labels[token_valid]].sum().detach()
                    )
                token_predictions = token_logits.argmax(-1)
                _add_confusion(
                    token_confusion,
                    token_predictions[token_valid],
                    token_labels[token_valid],
                )
        for sample_index, (sample, prediction_parts) in enumerate(
            zip(group, group_predictions, strict=True)
        ):
            recording_predictions = torch.cat(prediction_parts)
            diagnostic_predictions.append(recording_predictions)
            diagnostic_targets.append(masked_labels[sample_index].cpu())
            diagnostic_metadata.append(
                {
                    "subject_id": sample.subject_id,
                    "session_id": sample.session_id,
                    "sensor_id": sample.sensor_id,
                }
            )
            labels = sample.labels
            run_start = 0
            while run_start < labels.numel():
                label = int(labels[run_start])
                run_stop = run_start + 1
                while run_stop < labels.numel() and int(labels[run_stop]) == label:
                    run_stop += 1
                if label != -100:
                    majority = int(
                        torch.bincount(
                            recording_predictions[run_start:run_stop], minlength=num_classes
                        ).argmax()
                    )
                    segment_correct += int(majority == label)
                    segments += 1
                run_start = run_stop
    if frames == 0:
        raise ValueError("PARrad evaluation requires at least one supervised frame")
    macro_recall, macro_f1, per_class_recall = _classification_metrics(trimmed_confusion)
    token_summary = _confusion_summary(token_confusion)
    raw_macro_recall, raw_macro_f1, _ = _classification_metrics(raw_confusion)
    result = {
        "loss": loss_sum / loss_denominator,
        "accuracy": correct / frames,
        "frames": float(frames),
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "token_classification_loss": (
            token_loss_sum / token_loss_denominator if token_loss_denominator else 0.0
        ),
        "token_accuracy": token_summary["accuracy"],
        "token_macro_recall": token_summary["macro_recall"],
        "token_macro_f1": token_summary["macro_f1"],
        "supervised_tokens": token_summary["frames"],
        "frame_classification_weight": module.frame_classification_weight,
        "token_classification_weight": module.token_classification_weight,
        "raw_accuracy": raw_correct / raw_frames if raw_frames else 0.0,
        "raw_frames": float(raw_frames),
        "raw_macro_recall": raw_macro_recall,
        "raw_macro_f1": raw_macro_f1,
        "segment_accuracy": segment_correct / segments if segments else 0.0,
        "segments": float(segments),
        "per_class_recall": per_class_recall,
        "sensor_accuracy": {
            sensor_id: sensor_correct / sensor_total if sensor_total else 0.0
            for sensor_id, (sensor_correct, sensor_total) in sensor_counts.items()
        },
    }
    result.update(
        _sequence_diagnostics(
            predictions=tuple(diagnostic_predictions),
            targets=tuple(diagnostic_targets),
            metadata=tuple(diagnostic_metadata),
            num_classes=num_classes,
            boundary_tolerance_frames=boundary_tolerance_frames,
        )
    )
    return result


@torch.no_grad()
def evaluate_parrad_stage1(
    module: PARradStage1Module,
    loader: DataLoader,
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> dict[str, float]:
    """Evaluate frame-weighted loss and accuracy without updating model state."""
    module.eval()
    loss_sum = 0.0
    loss_denominator = 0.0
    correct = 0
    frames = 0
    for raw in loader:
        batch = _move_parrad(raw, device)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            logits, _ = module(batch)
            loss_numerator = functional.cross_entropy(
                logits.flatten(0, 1),
                batch.labels.flatten(),
                weight=module.class_weights,
                ignore_index=-100,
                reduction="sum",
            )
        valid = batch.labels.ne(-100)
        valid_frames = int(valid.sum().item())
        loss_sum += float(loss_numerator.detach())
        if module.class_weights is None:
            loss_denominator += valid_frames
        else:
            loss_denominator += float(module.class_weights[batch.labels[valid]].sum().detach())
        correct += int((logits.argmax(-1)[valid] == batch.labels[valid]).sum().item())
        frames += valid_frames
    if frames == 0:
        raise ValueError("PARrad evaluation requires at least one supervised frame")
    return {"loss": loss_sum / loss_denominator, "accuracy": correct / frames, "frames": float(frames)}


def train_stage2_batch(
    module: Stage2Module | PARradStage2Module,
    batch: RadarBatch | PARradBatch,
    ids: Tensor,
    mask: Tensor,
    optimizer: torch.optim.Optimizer,
    clip: float,
    *,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """单个 Stage 2 训练步；总损失在此处按权重合并三项损失。"""
    module.train()
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(batch.rd.device.type, enabled=amp_enabled):
        output = module.training_step(batch, ids, mask)
        auxiliary = total = output.losses.total(module.align_weight, module.consistency_weight)
        auxiliary_accuracy = total.new_zeros(())
        if isinstance(module, PARradStage2Module):
            if output.stream_class_logits is None or output.commit_class_logits is None:
                raise RuntimeError("PARrad Stage 2 requires auxiliary class logits")
            targets = parrad_event_targets(batch.labels, batch.frame_mask)
            auxiliary = (
                functional.cross_entropy(output.stream_class_logits, targets)
                + functional.cross_entropy(output.commit_class_logits, targets)
            ) / 2
            auxiliary_accuracy = (
                (output.stream_class_logits.argmax(-1) == targets).float().mean()
                + (output.commit_class_logits.argmax(-1) == targets).float().mean()
            ) / 2
            if module.phase == "semantic_warmup":
                total = auxiliary + module.warmup_consistency_weight * output.losses.consistency
            else:
                total = (
                    module.text_weight * output.losses.text
                    + module.align_weight * output.losses.align
                    + module.consistency_weight * output.losses.consistency
                    + module.auxiliary_weight * auxiliary
                )
    optimizer_stepped = _step(total, optimizer, (p for p in module.parameters() if p.requires_grad), clip, scaler)
    return {
        "loss": float(total.detach()),
        "text": float(output.losses.text.detach()),
        "align": float(output.losses.align.detach()),
        "consistency": float(output.losses.consistency.detach()),
        "event_state_norm": float(output.stream_state_norms[:, -1].mean().detach()),
        "auxiliary_classification_loss": float(auxiliary.detach()),
        "auxiliary_accuracy": float(auxiliary_accuracy.detach()),
        "optimizer_steps": float(optimizer_stepped),
        "optimizer_step_skipped": float(not optimizer_stepped),
    }


@torch.no_grad()
def evaluate_parrad_stage2(
    module: PARradStage2Module,
    loader: DataLoader,
    tokenizer: Any,
    device: torch.device,
    *,
    epoch: int,
    amp_enabled: bool = False,
) -> dict[str, float]:
    """Evaluate P2 on single-label event segments without changing stream state."""
    module.eval()
    sums = {
        "loss": 0.0,
        "text": 0.0,
        "align": 0.0,
        "consistency": 0.0,
        "stream_commit_cosine": 0.0,
        "event_state_norm": 0.0,
        "auxiliary_classification_loss": 0.0,
    }
    samples = 0
    auxiliary_correct = 0
    auxiliary_examples = 0
    auxiliary_confusion = torch.zeros(
        module.auxiliary_classifier.out_features,
        module.auxiliary_classifier.out_features,
        dtype=torch.long,
    )
    templates = default_parrad_event_templates()
    for raw in loader:
        batch = _move_parrad(raw, device)
        captions = select_event_captions(batch.labels, batch.frame_mask, templates, epoch=epoch)
        ids, mask = _tokenize_captions(tokenizer, captions)
        with torch.amp.autocast(device.type, enabled=amp_enabled):
            output = module.training_step(batch, ids.to(device), mask.to(device))
            if output.stream_class_logits is None or output.commit_class_logits is None:
                raise RuntimeError("PARrad Stage 2 requires auxiliary class logits")
            targets = parrad_event_targets(batch.labels, batch.frame_mask)
            auxiliary = (
                functional.cross_entropy(output.stream_class_logits, targets)
                + functional.cross_entropy(output.commit_class_logits, targets)
            ) / 2
            if module.phase == "semantic_warmup":
                total = auxiliary + module.warmup_consistency_weight * output.losses.consistency
            else:
                total = (
                    module.text_weight * output.losses.text
                    + module.align_weight * output.losses.align
                    + module.consistency_weight * output.losses.consistency
                    + module.auxiliary_weight * auxiliary
                )
            cosine = functional.cosine_similarity(output.stream_key, output.commit_key, dim=-1).mean()
            state_norm = output.stream_state_norms[:, -1].mean()
        count = batch.labels.shape[0]
        values = {
            "loss": total,
            "text": output.losses.text,
            "align": output.losses.align,
            "consistency": output.losses.consistency,
            "stream_commit_cosine": cosine,
            "event_state_norm": state_norm,
            "auxiliary_classification_loss": auxiliary,
        }
        for name, value in values.items():
            sums[name] += float(value.detach()) * count
        samples += count
        for predictions in (output.stream_class_logits.argmax(-1), output.commit_class_logits.argmax(-1)):
            auxiliary_correct += int((predictions == targets).sum().item())
            auxiliary_examples += targets.numel()
            for actual, predicted in zip(targets.cpu(), predictions.cpu(), strict=True):
                auxiliary_confusion[int(actual), int(predicted)] += 1
    if samples < 2:
        raise ValueError("PARrad Stage 2 evaluation requires at least two event segments")
    _, auxiliary_macro_f1, _ = _classification_metrics(auxiliary_confusion)
    return {
        **{name: value / samples for name, value in sums.items()},
        "auxiliary_accuracy": auxiliary_correct / auxiliary_examples,
        "auxiliary_macro_f1": auxiliary_macro_f1,
        "samples": float(samples),
    }


def _stage2_evaluation_batch_size(samples: int, requested: int) -> int:
    """Choose bounded batches while ensuring the final InfoNCE batch is never a singleton."""
    if samples < 2:
        raise ValueError("Stage 2 evaluation requires at least two samples")
    if requested < 2:
        raise ValueError("Stage 2 evaluation_batch_size must be at least 2")
    requested = min(samples, requested)
    for candidate in range(requested, samples + 1):
        if samples % candidate != 1:
            return candidate
    return samples


def _validation_score(stage: str, is_parrad: bool, metrics: dict[str, Any]) -> tuple[float, float, float]:
    if is_parrad and stage == "stage2":
        if metrics.get("phase") == "semantic_warmup":
            return (
                float(metrics["auxiliary_macro_f1"]),
                float(metrics["auxiliary_accuracy"]),
                -float(metrics["auxiliary_classification_loss"]),
            )
        return (
            -float(metrics["loss"]),
            float(metrics["stream_commit_cosine"]),
            -float(metrics["consistency"]),
        )
    if is_parrad and stage == "stage1" and float(metrics.get("token_classification_weight", 0.0)) > 0:
        return PhaseEarlyStopping.score(
            accuracy=float(metrics["token_accuracy"]),
            loss=float(metrics["token_classification_loss"]),
            macro_f1=float(metrics["token_macro_f1"]),
        )
    return PhaseEarlyStopping.score(
        accuracy=float(metrics["accuracy"]),
        loss=float(metrics["loss"]),
        macro_f1=float(metrics["macro_f1"]) if is_parrad else None,
    )


class TinyTokenizer:
    """冒烟测试专用的确定性分词器：按空白切词，再用 sha256 映射到词表 id。"""

    def __init__(self, vocab: int) -> None:
        self.vocab = vocab

    def __call__(self, captions: tuple[str | None, ...]) -> tuple[Tensor, Tensor]:
        """把 caption 元组编码为 padding 后的 [B,L] ids 与 mask。"""
        encoded = []
        for caption in captions:
            if caption is None:
                raise ValueError("Stage 2 requires captions")
            # 4 起始、3 结束（对应 TinyLLMAdapter 的裸词表布局），中间为哈希得到的词 id。
            tokens = [
                4 + int.from_bytes(hashlib.sha256(word.encode()).digest()[:4], "little") % (self.vocab - 5)
                for word in caption.split()
            ] + [3]
            encoded.append(tokens)
        length = max(map(len, encoded))
        ids = torch.zeros((len(encoded), length), dtype=torch.long)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for i, tokens in enumerate(encoded):
            ids[i, : len(tokens)] = torch.tensor(tokens)
            mask[i, : len(tokens)] = True
        return ids, mask


def _build_stage1(config: ExperimentConfig) -> Stage1Module:
    if not isinstance(config.model, ModelConfig):
        raise TypeError("default Stage 1 requires ModelConfig")
    return Stage1Module(
        StreamingRadarEncoder(config.model), config.model.num_joints, float(config.stage.get("velocity_weight", 1))
    )


def _build_parrad_stage1(
    config: ExperimentConfig, class_weights: Tensor | None = None
) -> PARradStage1Module:
    if not isinstance(config.model, PARradConfig):
        raise TypeError("PARrad Stage 1 requires PARradConfig")
    configured_weights = config.stage.get("class_weights")
    if class_weights is None and configured_weights is not None:
        if configured_weights == "balanced":
            raise ValueError("balanced class weights require the training dataset")
        class_weights = torch.tensor(configured_weights, dtype=torch.float32)
    encoder = build_parrad_encoder(config.model)
    initial_checkpoint = config.stage.get("initial_radar_checkpoint")
    if initial_checkpoint:
        load_stage1_radar_state(initial_checkpoint, encoder)
    module = PARradStage1Module(
        encoder,
        int(config.stage["num_classes"]),
        class_weights,
        float(
            config.stage.get(
                "label_run_temporal_smoothness_weight",
                config.stage.get("temporal_smoothness_weight", 0.0),
            )
        ),
        float(config.stage.get("frame_classification_weight", 1.0)),
        float(config.stage.get("token_classification_weight", 0.0)),
    )
    if bool(config.stage.get("freeze_radar_backbone", False)):
        for parameter in module.encoder.parameters():
            parameter.requires_grad = False
        for name, parameter in module.encoder.named_parameters():
            if name.startswith("token_resampler.") or name.startswith("token_projection."):
                parameter.requires_grad = True
    return module


def _build_parrad_stage2(config: ExperimentConfig) -> tuple[PARradStage2Module, Any]:
    if not isinstance(config.model, PARradConfig):
        raise TypeError("PARrad Stage 2 requires PARradConfig")
    checkpoint = config.stage.get("stage1_checkpoint")
    if not checkpoint:
        raise ValueError("PARrad Stage 2 requires stage1_checkpoint")
    radar = build_parrad_encoder(config.model)
    load_stage1_radar_state(checkpoint, radar)
    adapter, tokenizer, prompt_ids = _build_llm_backend(config)
    module = PARradStage2Module(
        radar,
        StreamingSemanticEventFormer(adapter, config.model, prompt_ids),
        SemanticProjection(config.model.llm_dim, config.model.semantic_dim),
        float(config.stage.get("align_weight", 1)),
        float(config.stage.get("consistency_weight", 1)),
        float(config.stage.get("contrastive_temperature", 0.07)),
        bool(config.stage.get("checkpoint_streaming", True)),
        num_classes=int(config.stage["num_classes"]),
        auxiliary_weight=float(config.stage.get("auxiliary_weight", 0.5)),
        text_weight=0.0,
        warmup_consistency_weight=float(config.stage.get("warmup_consistency_weight", 0.1)),
    )
    return module, tokenizer


def _tokenize_captions(tokenizer: Any, captions: tuple[str, ...]) -> tuple[Tensor, Tensor]:
    if isinstance(tokenizer, TinyTokenizer):
        return tokenizer(captions)
    encoded = tokenizer(
        list(captions),
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"].bool()


def _balanced_class_weights(dataset: PARradWindowDataset, num_classes: int) -> tuple[Tensor, list[float]]:
    counts = torch.zeros(num_classes, dtype=torch.long)
    for index in range(len(dataset)):
        sample = dataset[index]
        labels = sample.labels[sample.labels.ne(-100)]
        if labels.numel():
            counts += torch.bincount(labels, minlength=num_classes)[:num_classes]
    missing = counts.eq(0).nonzero().flatten().tolist()
    if missing:
        raise ValueError(f"training split is missing PARrad classes: {missing}")
    total = int(counts.sum().item())
    values = [total / (num_classes * int(count)) for count in counts.tolist()]
    return torch.tensor(values, dtype=torch.float32), values


def _build_llm_backend(config: ExperimentConfig) -> tuple[Any, Any, Tensor]:
    """Build one explicitly selected LLM adapter and its matching tokenizer."""
    backend = str(config.llm.get("backend", "qwen2"))
    if backend == "tiny":
        vocab = int(config.llm.get("vocab_size", 128))
        adapter = TinyLLMAdapter(vocab, config.model.llm_dim, config.model.llm_layers, config.model.llm_heads)
        return adapter, TinyTokenizer(vocab), torch.tensor(config.llm.get("prompt_ids", [1, 2, 3]))
    if backend not in {"qwen2", "gemma3"}:
        raise ValueError(f"unsupported LLM backend: {backend}")

    from transformers import AutoTokenizer

    default_name = "Qwen/Qwen2.5-0.5B-Instruct" if backend == "qwen2" else "google/gemma-3-270m-it"
    name = str(config.llm.get("name_or_path", default_name))
    adapter_type = Qwen2Adapter if backend == "qwen2" else Gemma3Adapter
    dtype: str | torch.dtype = config.llm.get("torch_dtype", "auto")
    if isinstance(dtype, str) and dtype != "auto":
        aliases = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if dtype not in aliases:
            raise ValueError(f"unsupported torch_dtype: {dtype}")
        dtype = aliases[dtype]
    adapter = adapter_type.from_pretrained(
        name, config.model.llm_layers, config.model.lora_rank, torch_dtype=dtype
    )
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_ids = tokenizer(FIXED_EVENT_PROMPT, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    return adapter, tokenizer, prompt_ids


def _build_stage2(config: ExperimentConfig) -> tuple[Stage2Module, Any]:
    """构建 Stage 2：加载 Stage 1 的 Radar 权重，再构建所选 LLM 后端。"""
    radar = StreamingRadarEncoder(config.model)
    if not config.stage.get("stage1_checkpoint"):
        raise ValueError("Stage 2 requires stage1_checkpoint")
    load_stage1_radar_state(config.stage["stage1_checkpoint"], radar)
    adapter, tokenizer, prompt_ids = _build_llm_backend(config)
    module = Stage2Module(
        radar,
        StreamingSemanticEventFormer(adapter, config.model, prompt_ids),
        SemanticProjection(config.model.llm_dim, config.model.semantic_dim),
        float(config.stage.get("align_weight", 1)),
        float(config.stage.get("consistency_weight", 1)),
        float(config.stage.get("contrastive_temperature", 0.07)),
        bool(config.stage.get("checkpoint_streaming", True)),
    )
    return module, tokenizer


def _move(batch: RadarBatch, device: torch.device) -> RadarBatch:
    """把张量搬到设备；captions/event_ids 是 Python 字符串，原样保留。"""
    return RadarBatch(
        batch.rd.to(device),
        batch.points.to(device),
        batch.point_mask.to(device),
        None if batch.skeleton is None else batch.skeleton.to(device),
        batch.captions,
        batch.event_ids,
        None if batch.frame_mask is None else batch.frame_mask.to(device),
    )


def _move_parrad(batch: PARradBatch, device: torch.device) -> PARradBatch:
    return PARradBatch(
        rd=batch.rd.to(device),
        md=batch.md.to(device),
        labels=batch.labels.to(device),
        timestamps=batch.timestamps.to(device),
        frame_mask=batch.frame_mask.to(device),
        sensor_index=batch.sensor_index.to(device),
        metadata=batch.metadata,
    )


def run_training(
    stage: str, config_path: str | Path, max_steps: int | None = None, resume: str | Path | None = None
) -> Path:
    """训练主流程；``max_steps`` 用于冒烟测试早期退出。"""
    config = load_experiment_config(config_path)
    variant = str(config.stage.get("variant", "default"))
    is_parrad = variant == "parrad"
    seed_everything(int(config.training.get("seed", 42)))
    # device=auto 时优先 CUDA，否则回落到 CPU。
    device = torch.device(
        "cuda"
        if config.training.get("device", "auto") == "auto" and torch.cuda.is_available()
        else config.training.get("device", "cpu")
        if config.training.get("device", "auto") != "auto"
        else "cpu"
    )
    output = Path(config.training.get("output_dir", f"outputs/{stage}"))
    output.mkdir(parents=True, exist_ok=True)
    validation_dataset: PARradWindowDataset | None = None
    test_dataset: PARradWindowDataset | None = None
    streaming_recordings: tuple[PARradSample, ...] = ()
    resolved_class_weights: Tensor | None = None
    train_window_mode = "fixed"
    boundary_frames = 0
    streaming_chunk_frames = 0
    streaming_sequence_frames = 0
    streaming_finetune_epochs = 0
    if is_parrad:
        if not isinstance(config.model, PARradConfig):
            raise TypeError("PARrad variant requires PARradConfig")
        manifests = {"train": config.data["train_manifest"]}
        if config.data.get("validation_manifest"):
            manifests["validation"] = config.data["validation_manifest"]
        if config.data.get("test_manifest"):
            manifests["test"] = config.data["test_manifest"]
        if "test" in manifests and "validation" not in manifests:
            raise ValueError("PARrad test_manifest requires validation_manifest for best-checkpoint selection")
        vocabulary = build_sensor_vocabulary(*manifests.values())
        recording_splits = {
            name: PARradManifestDataset(path, config=config.model, sensor_vocabulary=vocabulary)
            for name, path in manifests.items()
        }
        assert_disjoint_splits(
            train=recording_splits["train"].rows,
            validation=recording_splits.get("validation", ()).rows if "validation" in recording_splits else (),
            test=recording_splits.get("test", ()).rows if "test" in recording_splits else (),
        )
        window_frames = int(config.data.get("window_frames", 64))
        window_stride = int(config.data.get("window_stride", window_frames))
        boundary_seconds = float(config.data.get("label_boundary_ignore_seconds", 0.0))
        if boundary_seconds < 0:
            raise ValueError("label_boundary_ignore_seconds must be non-negative")
        boundary_frames = int(round(boundary_seconds / config.model.frame_period_s))
        if boundary_seconds > 0 and boundary_frames == 0:
            boundary_frames = 1
        train_window_mode = str(config.data.get("train_window_mode", "fixed"))
        if stage == "stage2" and train_window_mode != "label_run":
            raise ValueError("PARrad Stage 2 requires train_window_mode=label_run")
        min_label_run_frames = int(config.data.get("min_label_run_frames", 1))
        dataset = PARradWindowDataset(
            recording_splits["train"],
            window_frames=window_frames,
            window_stride=window_stride,
            label_boundary_ignore_frames=boundary_frames,
            window_mode=train_window_mode,
            min_label_run_frames=min_label_run_frames,
        )
        if "validation" in recording_splits:
            validation_dataset = PARradWindowDataset(
                recording_splits["validation"],
                window_frames=window_frames,
                window_stride=window_stride,
                label_boundary_ignore_frames=boundary_frames if stage == "stage2" else 0,
                window_mode="label_run" if stage == "stage2" else "fixed",
                min_label_run_frames=min_label_run_frames,
            )
        streaming_finetune_epochs = int(config.training.get("streaming_finetune_epochs", 0))
        if streaming_finetune_epochs < 0:
            raise ValueError("streaming_finetune_epochs must be non-negative")
        streaming_chunk_frames = int(config.data.get("streaming_chunk_frames", window_frames))
        streaming_sequence_frames = int(
            config.data.get("streaming_sequence_frames", streaming_chunk_frames * 4)
        )
        if streaming_finetune_epochs:
            if streaming_sequence_frames < streaming_chunk_frames:
                raise ValueError("streaming_sequence_frames must be at least streaming_chunk_frames")
            if streaming_sequence_frames % streaming_chunk_frames:
                raise ValueError("streaming_sequence_frames must be a multiple of streaming_chunk_frames")
            streaming_recordings = dataset.recordings
        if "test" in recording_splits:
            test_dataset = PARradWindowDataset(
                recording_splits["test"],
                window_frames=window_frames,
                window_stride=window_stride,
                label_boundary_ignore_frames=boundary_frames if stage == "stage2" else 0,
                window_mode="label_run" if stage == "stage2" else "fixed",
                min_label_run_frames=min_label_run_frames,
            )
        collate_fn = partial(parrad_collate, frames_per_update=config.model.frames_per_update)
        if int(config.training.get("num_workers", 0)) != 0:
            raise ValueError("PARrad cached recordings require num_workers=0")
        weight_spec = config.stage.get("class_weights")
        if weight_spec == "balanced":
            resolved_class_weights, weight_values = _balanced_class_weights(
                dataset, int(config.stage["num_classes"])
            )
        elif weight_spec is not None:
            weight_values = [float(value) for value in weight_spec]
            resolved_class_weights = torch.tensor(weight_values, dtype=torch.float32)
        else:
            weight_values = None
        if weight_values is not None:
            (output / "class_weights.json").write_text(json.dumps(weight_values, indent=2), encoding="utf-8")
    else:
        dataset = ManifestDataset(config.data["train_manifest"])
        collate_fn = radar_collate
    batch_size = int(config.training.get("batch_size", 2))
    # Stage 2 的 InfoNCE 需要至少 2 个样本作为负例，且不能超过数据集大小。
    if stage == "stage2" and (batch_size < 2 or batch_size > len(dataset)):
        raise ValueError("Stage 2 batch_size must be at least 2 and not exceed dataset size")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(config.training.get("num_workers", 0)),
        collate_fn=collate_fn,
        # Stage 2 丢弃不足一个 batch 的尾部数据，避免 batch=1 时无法计算对比损失。
        drop_last=stage == "stage2",
    )
    evaluation_batch_size = int(config.training.get("evaluation_batch_size", batch_size))
    if stage == "stage2" and validation_dataset is not None:
        validation_batch_size = _stage2_evaluation_batch_size(len(validation_dataset), evaluation_batch_size)
    else:
        validation_batch_size = evaluation_batch_size
    if stage == "stage2" and test_dataset is not None:
        test_batch_size = _stage2_evaluation_batch_size(len(test_dataset), evaluation_batch_size)
    else:
        test_batch_size = evaluation_batch_size
    validation_loader = (
        None
        if validation_dataset is None or (is_parrad and stage == "stage1")
        else DataLoader(
            validation_dataset,
            batch_size=validation_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    )
    test_loader = (
        None
        if test_dataset is None or (is_parrad and stage == "stage1")
        else DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn,
        )
    )
    has_validation = validation_dataset is not None if is_parrad else validation_loader is not None
    has_test = test_dataset is not None if is_parrad else test_loader is not None
    evaluation_recording_batch_size = int(
        config.training.get("evaluation_recording_batch_size", evaluation_batch_size)
    )
    if is_parrad:
        module, tokenizer = (
            (_build_parrad_stage1(config, resolved_class_weights), None)
            if stage == "stage1"
            else _build_parrad_stage2(config)
        )
    else:
        module, tokenizer = (_build_stage1(config), None) if stage == "stage1" else _build_stage2(config)
    module.to(device)
    stage2_warmup_epochs = int(config.stage.get("semantic_warmup_epochs", 0))
    stage2_text_ramp_epochs = int(config.stage.get("text_weight_ramp_epochs", 0))
    stage2_text_weight_final = float(config.stage.get("text_weight_final", 1.0))
    if is_parrad and stage == "stage2":
        set_parrad_stage2_phase(module, _parrad_stage2_phase(0, stage2_warmup_epochs))
        module.text_weight = _parrad_stage2_text_weight(
            0, stage2_warmup_epochs, stage2_text_ramp_epochs, stage2_text_weight_final
        )
    (output / "trainable_parameters.json").write_text(json.dumps(parameter_report(module), indent=2), encoding="utf-8")
    joint_adapter_ids = module._joint_adapter_parameter_ids if isinstance(module, PARradStage2Module) else frozenset()
    trainable = [p for p in module.parameters() if p.requires_grad or id(p) in joint_adapter_ids]
    lr, weight_decay = (
        float(config.training.get("learning_rate", 1e-4)),
        float(config.training.get("weight_decay", 1e-2)),
    )
    if stage == "stage2":
        # Radar Encoder 已有预训练权重，用独立（默认 0.1×）的小学习率继续微调。
        radar_ids = {id(p) for p in module.radar_encoder.parameters()}
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": [p for p in trainable if id(p) in radar_ids],
                    "lr": float(config.training.get("radar_learning_rate", lr * 0.1)),
                },
                {"params": [p for p in trainable if id(p) not in radar_ids], "lr": lr},
            ],
            weight_decay=weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    # AMP 仅在 CUDA 上启用，CPU 下 GradScaler 无意义。
    amp = bool(config.training.get("amp", True)) and device.type == "cuda"
    scaler = (
        torch.amp.GradScaler("cuda", init_scale=float(config.training.get("amp_init_scale", 65536)))
        if amp
        else None
    )
    history: list[dict[str, float]] = []
    step = 0
    start_epoch = 0
    metrics_path = output / "metrics.jsonl"
    validation_metrics_path = output / "validation_metrics.jsonl"
    best_path = output / "best.pt"
    warmup_best_path = output / "best_semantic_warmup.pt"
    best_validation_score: tuple[float, float, float] | None = None
    best_validation_phase: str | None = None
    if resume:
        # 恢复优化器/AMP 状态与全部随机源，保证与中断前严格接着跑。
        payload = load_training_checkpoint(resume, module, optimizer, True, scaler)
        if payload["stage"] != stage or payload["config"]["model"] != config.model.to_dict():
            raise ValueError("resume checkpoint stage/model mismatch")
        history, step, start_epoch = payload["metric_history"], payload["step"], payload["epoch"] + 1
        if has_validation and validation_metrics_path.exists():
            previous_validation = [
                json.loads(line)
                for line in validation_metrics_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if previous_validation:
                resume_phase = (
                    _parrad_stage2_phase(start_epoch, stage2_warmup_epochs)
                    if is_parrad and stage == "stage2"
                    else None
                )
                if resume_phase is not None:
                    previous_validation = [
                        row for row in previous_validation if row.get("phase") == resume_phase
                    ]
                    best_validation_phase = resume_phase
            if previous_validation:
                best_record = max(
                    previous_validation,
                    key=lambda row: _validation_score(stage, is_parrad, row),
                )
                best_validation_score = _validation_score(stage, is_parrad, best_record)
    else:
        metrics_path.write_text("", encoding="utf-8")
        if has_validation:
            validation_metrics_path.write_text("", encoding="utf-8")
    last_epoch = start_epoch - 1
    primary_epochs = int(config.training.get("epochs", 1))
    total_epochs = primary_epochs + streaming_finetune_epochs
    include_phase = is_parrad and (
        (stage == "stage1" and (train_window_mode != "fixed" or streaming_finetune_epochs > 0))
        or (stage == "stage2" and stage2_warmup_epochs > 0)
    )
    warmup_ratio = float(config.training.get("warmup_ratio", 0.05))
    min_lr_ratio = float(config.training.get("min_lr_ratio", 0.1))
    label_schedule = (
        WarmupCosineSchedule(
            optimizer,
            base_lr=lr,
            total_steps=max(1, len(loader) * primary_epochs),
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )
        if is_parrad and stage == "stage1"
        else None
    )
    streaming_schedule: WarmupCosineSchedule | None = None
    if is_parrad and stage == "stage1" and streaming_finetune_epochs:
        recording_chunks = sum(
            math.ceil(sample.rd.shape[0] / streaming_chunk_frames) for sample in streaming_recordings
        )
        estimated_steps_per_epoch = max(1, math.ceil(recording_chunks / batch_size))
        streaming_schedule = WarmupCosineSchedule(
            optimizer,
            base_lr=float(config.training.get("streaming_learning_rate", lr * 0.3)),
            total_steps=estimated_steps_per_epoch * streaming_finetune_epochs,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )
    early_stoppers: dict[str, PhaseEarlyStopping] = {}
    if has_validation:
        patience = int(config.training.get("early_stopping_patience", 4))
        early_stoppers = {
            "label": PhaseEarlyStopping(patience),
            "streaming": PhaseEarlyStopping(patience),
        }
    stopped_phases: set[str] = set()
    for epoch in range(start_epoch, total_epochs):
        streaming_phase = is_parrad and stage == "stage1" and epoch >= primary_epochs
        if is_parrad and stage == "stage2":
            phase = _parrad_stage2_phase(epoch, stage2_warmup_epochs)
            if phase != best_validation_phase:
                best_validation_score = None
                best_validation_phase = phase
            set_parrad_stage2_phase(module, phase)
            module.text_weight = _parrad_stage2_text_weight(
                epoch,
                stage2_warmup_epochs,
                stage2_text_ramp_epochs,
                stage2_text_weight_final,
            )
        phase_key = "streaming" if streaming_phase else "label"
        if phase_key in stopped_phases:
            continue
        last_epoch = epoch
        if not (is_parrad and stage == "stage2"):
            phase = "streaming_finetune" if streaming_phase else train_window_mode
        if is_parrad and stage == "stage1":
            weight_key = (
                "streaming_temporal_smoothness_weight"
                if streaming_phase
                else "label_run_temporal_smoothness_weight"
            )
            module.temporal_smoothness_weight = float(
                config.stage.get(weight_key, config.stage.get("temporal_smoothness_weight", 0.0))
            )
        if streaming_phase:
            if not streaming_recordings:
                raise ValueError("streaming fine-tune requires training recordings")
            epoch_batches: Iterable[Any] = _iter_parrad_recording_sequences(
                streaming_recordings,
                sequence_frames=streaming_sequence_frames,
                recording_batch_size=batch_size,
                boundary_ignore_frames=boundary_frames,
            )
        else:
            epoch_batches = loader
        stream_state: PARradAnyStreamState | None = None
        for item in epoch_batches:
            optimizer_steps = 1
            if streaming_phase:
                new_group, raw = item
                if new_group:
                    stream_state = None
                remaining = None if max_steps is None else max_steps - step
                if remaining is not None and remaining <= 0:
                    break
                streaming_metrics, stream_state = train_parrad_streaming_batch(
                    module,
                    raw,
                    optimizer,
                    float(config.training.get("grad_clip_norm", 1)),
                    chunk_frames=streaming_chunk_frames,
                    device=device,
                    amp_enabled=amp,
                    scaler=scaler,
                    max_updates=remaining,
                    initial_state=stream_state,
                    return_state=True,
                    before_optimizer_step=None if streaming_schedule is None else streaming_schedule.step,
                )
                optimizer_steps = int(streaming_metrics.pop("optimizer_steps"))
                metrics = streaming_metrics
            else:
                raw = item
            if not streaming_phase and is_parrad and stage == "stage1":
                batch = _move_parrad(raw, device)
                if label_schedule is None:
                    raise RuntimeError("PARrad label-run schedule was not initialized")
                learning_rate = label_schedule.step()
                metrics = train_parrad_stage1_batch(
                    module,
                    batch,
                    optimizer,
                    float(config.training.get("grad_clip_norm", 1)),
                    amp_enabled=amp,
                    scaler=scaler,
                )
                metrics["learning_rate"] = learning_rate
            elif not streaming_phase and is_parrad:
                batch = _move_parrad(raw, device)
                templates = default_parrad_event_templates()
                captions = select_event_captions(
                    batch.labels,
                    batch.frame_mask,
                    templates,
                    epoch=epoch,
                )
                if tokenizer is None:
                    raise RuntimeError("PARrad Stage 2 tokenizer was not initialized")
                ids, mask = _tokenize_captions(tokenizer, captions)
                metrics = train_stage2_batch(
                    module,
                    batch,
                    ids.to(device),
                    mask.to(device),
                    optimizer,
                    float(config.training.get("grad_clip_norm", 1)),
                    amp_enabled=amp,
                    scaler=scaler,
                )
            elif not streaming_phase and stage == "stage1":
                batch = _move(raw, device)
                metrics = train_stage1_batch(
                    module,
                    batch,
                    optimizer,
                    float(config.training.get("grad_clip_norm", 1)),
                    amp_enabled=amp,
                    scaler=scaler,
                )
            elif not streaming_phase:
                batch = _move(raw, device)
                ids, mask = _tokenize_captions(tokenizer, batch.captions)
                metrics = train_stage2_batch(
                    module,
                    batch,
                    ids.to(device),
                    mask.to(device),
                    optimizer,
                    float(config.training.get("grad_clip_norm", 1)),
                    amp_enabled=amp,
                    scaler=scaler,
                )
            optimizer_steps = int(metrics.pop("optimizer_steps", optimizer_steps))
            step += optimizer_steps
            record = {"epoch": float(epoch), "step": float(step), **metrics}
            if include_phase:
                record["phase"] = phase
            history.append(record)
            # 逐行追加 JSONL，即使训练意外中断也能保留已完成的指标。
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            if max_steps is not None and step >= max_steps:
                break
        reached_max_steps = max_steps is not None and step >= max_steps
        if has_validation and not reached_max_steps:
            if is_parrad and stage == "stage1":
                validation_metrics = evaluate_parrad_stage1_streaming(
                    module,
                    validation_dataset.recordings,
                    device,
                    chunk_frames=streaming_chunk_frames,
                    recording_batch_size=evaluation_recording_batch_size,
                    boundary_ignore_frames=boundary_frames,
                    amp_enabled=amp,
                )
            elif is_parrad:
                if validation_loader is None or tokenizer is None:
                    raise RuntimeError("PARrad Stage 2 validation was not initialized")
                validation_metrics = evaluate_parrad_stage2(
                    module,
                    validation_loader,
                    tokenizer,
                    device,
                    epoch=epoch,
                    amp_enabled=amp,
                )
            else:
                validation_metrics = evaluate_parrad_stage1(
                    module, validation_loader, device, amp_enabled=amp
                )
            if is_parrad and stage == "stage2":
                validation_metrics["phase"] = phase
            validation_record = {"epoch": float(epoch), "step": float(step), **validation_metrics}
            if include_phase:
                validation_record["phase"] = phase
            with validation_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(validation_record) + "\n")
            score = _validation_score(stage, is_parrad, validation_metrics)
            if is_parrad and stage == "stage1":
                use_token_score = module.token_classification_weight > 0
                phase_improved, phase_should_stop = early_stoppers[phase_key].update(
                    accuracy=float(
                        validation_metrics["token_accuracy"]
                        if use_token_score
                        else validation_metrics["accuracy"]
                    ),
                    loss=float(
                        validation_metrics["token_classification_loss"]
                        if use_token_score
                        else validation_metrics["loss"]
                    ),
                    macro_f1=float(
                        validation_metrics["token_macro_f1"]
                        if use_token_score
                        else validation_metrics["macro_f1"]
                    ),
                )
                if phase_improved:
                    phase_best_path = output / (
                        "best_streaming.pt" if streaming_phase else "best_label_run.pt"
                    )
                    save_training_checkpoint(
                        phase_best_path,
                        stage=stage,
                        model=module,
                        optimizer=optimizer,
                        epoch=epoch,
                        step=step,
                        config=config.to_dict(),
                        metric_history=history,
                        scaler=scaler,
                    )
                if phase_should_stop:
                    stopped_phases.add(phase_key)
            if best_validation_score is None or score > best_validation_score:
                best_validation_score = score
                current_best_path = (
                    warmup_best_path
                    if is_parrad and stage == "stage2" and phase == "semantic_warmup"
                    else best_path
                )
                save_training_checkpoint(
                    current_best_path,
                    stage=stage,
                    model=module,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=step,
                    config=config.to_dict(),
                    metric_history=history,
                    trainable_only=stage == "stage2",
                    scaler=scaler,
                )
        if reached_max_steps:
            break
        if streaming_phase and phase_key in stopped_phases:
            break
    save_training_checkpoint(
        output / "last.pt",
        stage=stage,
        model=module,
        optimizer=optimizer,
        epoch=last_epoch,
        step=step,
        config=config.to_dict(),
        metric_history=history,
        # Stage 2 的 Qwen 主干不更新，只保存可训练参数以控制体积。
        trainable_only=stage == "stage2",
        scaler=scaler,
    )
    selected_best_path = best_path if best_path.exists() else warmup_best_path
    if has_validation and selected_best_path.exists():
        load_training_checkpoint(selected_best_path, module)
        if has_test:
            if is_parrad and stage == "stage1":
                test_metrics = evaluate_parrad_stage1_streaming(
                    module,
                    test_dataset.recordings,
                    device,
                    chunk_frames=streaming_chunk_frames,
                    recording_batch_size=evaluation_recording_batch_size,
                    boundary_ignore_frames=boundary_frames,
                    amp_enabled=amp,
                )
            elif is_parrad:
                if test_loader is None or tokenizer is None:
                    raise RuntimeError("PARrad Stage 2 test evaluation was not initialized")
                test_metrics = evaluate_parrad_stage2(
                    module,
                    test_loader,
                    tokenizer,
                    device,
                    epoch=last_epoch,
                    amp_enabled=amp,
                )
            else:
                test_metrics = evaluate_parrad_stage1(module, test_loader, device, amp_enabled=amp)
            (output / "test_metrics.json").write_text(
                json.dumps({"checkpoint": best_path.name, **test_metrics}, indent=2) + "\n",
                encoding="utf-8",
            )
    # Stage 1 额外导出不含 Pose Head/临时分类头的最佳 Radar Encoder，供后续阶段使用。
    if stage == "stage1":
        save_stage1_radar_export(output / "radar_encoder.pt", module, config.model.to_dict())
    return output


def main(argv: Sequence[str] | None = None) -> None:
    """CLI 入口：``python -m continuous_mmwave.train <stage> --config <yaml>``。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("stage1", "stage2", "stage3"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "stage3":
        from continuous_mmwave.parrad_stage3_runner import run_parrad_stage3

        result = run_parrad_stage3(args.config, args.resume, args.max_steps, args.evaluate_test)
        print(f"finished stage3; outputs: {result.output_dir}")
        return
    print(f"finished {args.stage}; outputs: {run_training(args.stage, args.config, args.max_steps, args.resume)}")


if __name__ == "__main__":
    main()
