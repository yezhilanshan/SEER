"""Read-only diagnostics for PARrad Stage P2 representation and loss behavior."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils import clip_grad_norm_

from continuous_mmwave.checkpoint import load_training_checkpoint
from continuous_mmwave.data_parrad import (
    PARradBatch,
    PARradManifestDataset,
    PARradSample,
    PARradWindowDataset,
    build_sensor_vocabulary,
    parrad_collate,
)
from continuous_mmwave.parrad_config import PARradConfig
from continuous_mmwave.stages.parrad_stage2 import (
    default_parrad_event_templates,
    parrad_event_targets,
    select_event_captions,
    set_parrad_stage2_phase,
)
from continuous_mmwave.stages.stage2 import causal_caption_cross_entropy
from continuous_mmwave.train import (
    ExperimentConfig,
    _build_parrad_stage2,
    _move_parrad,
    _parrad_stage2_phase,
    _parrad_stage2_text_weight,
    _tokenize_captions,
    load_experiment_config,
    seed_everything,
)


def _label(sample: PARradSample) -> int:
    labels = torch.unique(sample.labels[sample.labels.ne(-100)])
    if labels.numel() != 1:
        raise ValueError("P2 diagnostics require single-label event windows")
    return int(labels.item())


def _dataset(config: ExperimentConfig, split: str) -> PARradWindowDataset:
    if not isinstance(config.model, PARradConfig):
        raise TypeError("PARrad Stage 2 diagnostics require PARradConfig")
    manifest_key = f"{split}_manifest"
    if manifest_key not in config.data:
        raise ValueError(f"missing {manifest_key}")
    manifests = [
        config.data[key]
        for key in ("train_manifest", "validation_manifest", "test_manifest")
        if config.data.get(key)
    ]
    vocabulary = build_sensor_vocabulary(*manifests)
    recordings = PARradManifestDataset(config.data[manifest_key], config.model, vocabulary)
    boundary_seconds = float(config.data.get("label_boundary_ignore_seconds", 0.0))
    boundary_frames = int(round(boundary_seconds / config.model.frame_period_s))
    if boundary_seconds > 0 and boundary_frames == 0:
        boundary_frames = 1
    window_frames = int(config.data.get("window_frames", 64))
    return PARradWindowDataset(
        recordings,
        window_frames=window_frames,
        window_stride=int(config.data.get("window_stride", window_frames)),
        label_boundary_ignore_frames=boundary_frames,
        window_mode="label_run",
        min_label_run_frames=int(config.data.get("min_label_run_frames", 1)),
    )


def _balanced_samples(dataset: PARradWindowDataset, num_classes: int, per_class: int) -> list[PARradSample]:
    selected: dict[int, list[PARradSample]] = {label: [] for label in range(num_classes)}
    for index in range(len(dataset)):
        sample = dataset[index]
        label = _label(sample)
        if label in selected and len(selected[label]) < per_class:
            selected[label].append(sample)
        if all(len(samples) >= per_class for samples in selected.values()):
            break
    missing = [label for label, samples in selected.items() if len(samples) < per_class]
    if missing:
        raise ValueError(f"diagnostic split is missing labels: {missing}")
    return [sample for label in range(num_classes) for sample in selected[label]]


def false_negative_diagnostics(labels: list[int], batch_size: int, seed: int) -> dict[str, float]:
    if batch_size < 2 or len(labels) < batch_size:
        raise ValueError("false-negative diagnostics require at least one batch")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(labels), generator=generator).tolist()
    batches = [order[start : start + batch_size] for start in range(0, len(order), batch_size)]
    batches = [batch for batch in batches if len(batch) == batch_size]
    duplicate_batches = 0
    false_negative_pairs = 0
    off_diagonal_pairs = 0
    affected_samples = 0
    for batch in batches:
        batch_labels = [labels[index] for index in batch]
        counts: dict[int, int] = defaultdict(int)
        for label in batch_labels:
            counts[label] += 1
        duplicate_batches += int(any(count > 1 for count in counts.values()))
        affected_samples += sum(count for count in counts.values() if count > 1)
        false_negative_pairs += sum(count * (count - 1) for count in counts.values())
        off_diagonal_pairs += batch_size * (batch_size - 1)
    samples = len(batches) * batch_size
    return {
        "samples": float(samples),
        "batches": float(len(batches)),
        "duplicate_batch_fraction": duplicate_batches / max(1, len(batches)),
        "affected_sample_fraction": affected_samples / max(1, samples),
        "false_negative_pair_fraction": false_negative_pairs / max(1, off_diagonal_pairs),
    }


def _off_diagonal(matrix: Tensor) -> Tensor:
    mask = ~torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
    return matrix[mask]


def masked_token_mean(tokens: Tensor, mask: Tensor) -> Tensor:
    """Pool valid sequence tokens and return zero for an entirely masked sample."""
    if tokens.ndim != 3 or mask.shape != tokens.shape[:2]:
        raise ValueError("tokens/mask must have shapes [B,L,D]/[B,L]")
    valid = mask.bool()
    denominator = valid.sum(1, keepdim=True).clamp_min(1)
    pooled = (tokens * valid.unsqueeze(-1)).sum(1) / denominator
    return torch.where(valid.any(1, keepdim=True), pooled, torch.zeros_like(pooled))


def summarize_gate_values(gates: Tensor) -> dict[str, float]:
    """Summarize convex state-update gates without assuming a batch shape."""
    if gates.numel() == 0:
        raise ValueError("gate diagnostics require at least one value")
    values = gates.detach().float()
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
        "near_zero_fraction": float(values.lt(0.01).float().mean()),
        "near_one_fraction": float(values.gt(0.99).float().mean()),
    }


def action_classification_diagnostics(
    stream_logits: Tensor,
    commit_logits: Tensor,
    targets: Tensor,
) -> dict[str, float]:
    if stream_logits.shape != commit_logits.shape or stream_logits.ndim != 2:
        raise ValueError("stream/commit class logits must share [B,C]")
    if targets.shape != (stream_logits.shape[0],):
        raise ValueError("action targets must be [B]")
    stream_accuracy = (stream_logits.argmax(-1) == targets).float().mean()
    commit_accuracy = (commit_logits.argmax(-1) == targets).float().mean()
    loss = (
        torch.nn.functional.cross_entropy(stream_logits, targets)
        + torch.nn.functional.cross_entropy(commit_logits, targets)
    ) / 2
    return {
        "stream_accuracy": float(stream_accuracy),
        "commit_accuracy": float(commit_accuracy),
        "mean_accuracy": float((stream_accuracy + commit_accuracy) / 2),
        "classification_loss": float(loss),
    }


def representation_diagnostics(keys: Tensor) -> dict[str, float]:
    keys = keys.detach().float()
    centered = keys - keys.mean(0, keepdim=True)
    singular_values = torch.linalg.svdvals(centered)
    spectrum = singular_values.square()
    probabilities = spectrum / spectrum.sum().clamp_min(torch.finfo(spectrum.dtype).eps)
    effective_rank = torch.exp(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
    cosine = keys @ keys.T
    return {
        "mean_feature_variance": float(centered.var(0, unbiased=False).mean()),
        "effective_rank": float(effective_rank),
        "numerical_rank": float(torch.linalg.matrix_rank(centered)),
        "off_diagonal_cosine_mean": float(_off_diagonal(cosine).mean()),
        "off_diagonal_cosine_std": float(_off_diagonal(cosine).std(unbiased=False)),
    }


def alignment_diagnostics(event_keys: Tensor, text_keys: Tensor) -> dict[str, float]:
    similarities = event_keys.detach().float() @ text_keys.detach().float().T
    positive = similarities.diagonal()
    negative = _off_diagonal(similarities)
    return {
        "positive_cosine_mean": float(positive.mean()),
        "positive_cosine_std": float(positive.std(unbiased=False)),
        "negative_cosine_mean": float(negative.mean()),
        "negative_cosine_std": float(negative.std(unbiased=False)),
        "positive_negative_margin": float(positive.mean() - negative.mean()),
        "top1_retrieval_accuracy": float((similarities.argmax(1) == torch.arange(similarities.shape[0], device=similarities.device)).float().mean()),
    }


def _history(path: Path) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("training metrics history is empty")
    keys = ("loss", "text", "align", "consistency", "event_state_norm")
    quarters = []
    for quarter in range(4):
        start = len(rows) * quarter // 4
        stop = len(rows) * (quarter + 1) // 4
        subset = rows[start:stop]
        quarters.append({key: sum(float(row[key]) for row in subset) / len(subset) for key in keys})
    return {
        "steps": len(rows),
        "first": {key: float(rows[0][key]) for key in keys},
        "last": {key: float(rows[-1][key]) for key in keys},
        "quarter_means": quarters,
        "random_infonce_baseline": math.log(4),
    }


def _parameter_groups(module: nn.Module) -> dict[str, list[Tensor]]:
    groups: dict[str, list[Tensor]] = defaultdict(list)
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.startswith("radar_encoder."):
            group = "radar_encoder"
        elif name.startswith("event_former.adapter."):
            group = "qwen_lora"
        elif name.startswith("event_former."):
            group = "event_former"
        elif name.startswith("semantic_projection."):
            group = "semantic_projection"
        else:
            group = "other"
        groups[group].append(parameter)
    return dict(groups)


def cross_attention_gradient_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the complete cross-attention gradient audit, not only top-k rows."""
    prefix = "event_former.cross_attention."
    return [row for row in rows if str(row.get("name", "")).startswith(prefix)]


def _gradient_norms(module: nn.Module, batch: PARradBatch, ids: Tensor, mask: Tensor, amp: bool) -> dict[str, Any]:
    module.train()
    groups = _parameter_groups(module)
    named_trainable = [(name, parameter) for name, parameter in module.named_parameters() if parameter.requires_grad]
    results: dict[str, Any] = {}
    losses = ("text", "align", "consistency", "total")
    for loss_name in losses:
        torch.manual_seed(1234)
        module.zero_grad(set_to_none=True)
        with torch.amp.autocast(batch.rd.device.type, enabled=amp):
            output = module.training_step(batch, ids, mask)
            value = (
                output.losses.total(module.align_weight, module.consistency_weight)
                if loss_name == "total"
                else getattr(output.losses, loss_name)
            )
        value.backward()
        group_results = {}
        for group_name, parameters in groups.items():
            gradients = [parameter.grad.detach().float() for parameter in parameters if parameter.grad is not None]
            squared = sum(float(gradient.square().sum()) for gradient in gradients)
            group_results[group_name] = {
                "global_l2_norm": math.sqrt(squared),
                "parameters_with_gradient": len(gradients),
                "nonzero_elements": sum(int(gradient.ne(0).sum()) for gradient in gradients),
            }
        per_parameter = [
            {
                "name": name,
                "l2_norm": float(parameter.grad.detach().float().norm()),
                "nonzero_elements": int(parameter.grad.detach().ne(0).sum()),
            }
            for name, parameter in named_trainable
            if parameter.grad is not None
        ]
        critical = {
            row["name"]: {"l2_norm": row["l2_norm"], "nonzero_elements": row["nonzero_elements"]}
            for row in per_parameter
            if row["name"].endswith(".alpha")
            or row["name"] == "event_former.base_event_tokens"
            or "event_former.state_gate" in row["name"]
            or row["name"] == "radar_encoder.token_resampler.queries"
        }
        top = sorted(per_parameter, key=lambda row: row["l2_norm"], reverse=True)[:12]
        results[loss_name] = {
            "value": float(value.detach()),
            "groups": group_results,
            "critical_parameter_gradients": critical,
            "cross_attention_parameter_gradients": cross_attention_gradient_rows(per_parameter),
            "top_parameter_gradients": top,
        }
    module.zero_grad(set_to_none=True)
    return results


def _checkpoint_probe_trainable(module: nn.Module) -> list[nn.Parameter]:
    """Match the Stage 2 runner's optimizer union across warmup/joint phases."""
    joint_adapter_ids = getattr(module, "_joint_adapter_parameter_ids", frozenset())
    return [
        parameter
        for parameter in module.parameters()
        if parameter.requires_grad or id(parameter) in joint_adapter_ids
    ]


def _checkpoint_probe_scaler(training: dict[str, Any], amp: bool) -> Any | None:
    if not amp:
        return None
    return torch.amp.GradScaler(
        "cuda", init_scale=float(training.get("amp_init_scale", 65536))
    )


def _parameter_snapshots(parameters: Iterable[nn.Parameter]) -> dict[int, Tensor]:
    return {id(parameter): parameter.detach().float().cpu().clone() for parameter in parameters}


def _max_parameter_change(parameters: Iterable[nn.Parameter], snapshots: dict[int, Tensor]) -> float:
    changes = [
        float((parameter.detach().float().cpu() - snapshots[id(parameter)]).abs().max())
        for parameter in parameters
    ]
    return max(changes, default=0.0)


def _optimizer_bootstrap_probe(
    module: nn.Module,
    batch: PARradBatch,
    ids: Tensor,
    mask: Tensor,
    config: ExperimentConfig,
    checkpoint: Path,
    amp: bool,
    steps: int = 10,
) -> list[dict[str, Any]]:
    """Continue a checkpoint in memory and expose AMP skips plus radar-path opening."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    warmup_epochs = int(config.stage.get("semantic_warmup_epochs", 0))
    phase = _parrad_stage2_phase(int(payload["epoch"]), warmup_epochs)
    set_parrad_stage2_phase(module, phase)
    module.text_weight = _parrad_stage2_text_weight(
        int(payload["epoch"]),
        warmup_epochs,
        int(config.stage.get("text_weight_ramp_epochs", 0)),
        float(config.stage.get("text_weight_final", 1.0)),
    )
    trainable = _checkpoint_probe_trainable(module)
    radar_ids = {id(parameter) for parameter in module.radar_encoder.parameters()}
    learning_rate = float(config.training.get("learning_rate", 1e-4))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [parameter for parameter in trainable if id(parameter) in radar_ids],
                "lr": float(config.training.get("radar_learning_rate", learning_rate * 0.1)),
            },
            {"params": [parameter for parameter in trainable if id(parameter) not in radar_ids], "lr": learning_rate},
        ],
        weight_decay=float(config.training.get("weight_decay", 1e-2)),
    )
    scaler = _checkpoint_probe_scaler(config.training, amp)
    load_training_checkpoint(checkpoint, module, optimizer, True, scaler)
    clip = float(config.training.get("grad_clip_norm", 1.0))
    alpha_parameters = [layer.alpha for layer in module.event_former.cross_attention]
    radar_parameters = list(module.radar_encoder.parameters())
    joint_adapter_ids = getattr(module, "_joint_adapter_parameter_ids", frozenset())
    adapter_parameters = [
        parameter for parameter in module.event_former.adapter.parameters() if id(parameter) in joint_adapter_ids
    ]
    radar_before = _parameter_snapshots(radar_parameters)
    adapter_before = _parameter_snapshots(adapter_parameters)
    records: list[dict[str, Any]] = []
    module.train()
    for index in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(batch.rd.device.type, enabled=amp):
            output = module.training_step(batch, ids, mask)
            if output.stream_class_logits is None or output.commit_class_logits is None:
                raise RuntimeError("PARrad Stage 2 requires auxiliary class logits")
            targets = parrad_event_targets(batch.labels, batch.frame_mask)
            auxiliary = (
                functional.cross_entropy(output.stream_class_logits, targets)
                + functional.cross_entropy(output.commit_class_logits, targets)
            ) / 2
            if phase == "semantic_warmup":
                loss = auxiliary + module.warmup_consistency_weight * output.losses.consistency
            else:
                loss = (
                    module.text_weight * output.losses.text
                    + module.align_weight * output.losses.align
                    + module.consistency_weight * output.losses.consistency
                    + module.auxiliary_weight * auxiliary
                )
        scale_before = None if scaler is None else float(scaler.get_scale())
        if scaler is None:
            loss.backward()
        else:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        alpha_gradient_norms = [
            0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm())
            for parameter in alpha_parameters
        ]
        cross_attention_gradient_norms = {
            name: 0.0 if parameter.grad is None else float(parameter.grad.detach().float().norm())
            for name, parameter in module.named_parameters()
            if name.startswith("event_former.cross_attention.")
        }
        radar_gradients = [
            parameter.grad.detach().float()
            for parameter in radar_parameters
            if parameter.grad is not None
        ]
        radar_gradient_norm = math.sqrt(sum(float(gradient.square().sum()) for gradient in radar_gradients))
        clip_grad_norm_(trainable, clip)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        scale_after = None if scaler is None else float(scaler.get_scale())
        records.append(
            {
                "probe_step": index + 1,
                "phase": phase,
                "loss": float(loss.detach()),
                "auxiliary_classification_loss": float(auxiliary.detach()),
                "scale_before": scale_before,
                "scale_after": scale_after,
                "optimizer_step_skipped": bool(
                    scaler is not None and scale_after is not None and scale_before is not None and scale_after < scale_before
                ),
                "alpha_gradient_norms": alpha_gradient_norms,
                "alpha_values_after_step": [float(parameter.detach()) for parameter in alpha_parameters],
                "cross_attention_gradient_norms": cross_attention_gradient_norms,
                "radar_gradient_norm": radar_gradient_norm,
                "radar_nonzero_gradient_elements": sum(int(gradient.ne(0).sum()) for gradient in radar_gradients),
                "radar_max_abs_change": _max_parameter_change(radar_parameters, radar_before),
                "adapter_max_abs_change": _max_parameter_change(adapter_parameters, adapter_before),
            }
        )
    optimizer.zero_grad(set_to_none=True)
    return records


def _state_and_key_diagnostics(
    module: nn.Module,
    tokenizer: Any,
    batch: PARradBatch,
    device: torch.device,
    amp: bool,
) -> tuple[dict[str, Any], Tensor, Tensor]:
    batch = _move_parrad(batch, device)
    captions = select_event_captions(batch.labels, batch.frame_mask, default_parrad_event_templates(), epoch=0)
    ids, mask = _tokenize_captions(tokenizer, captions)
    ids, mask = ids.to(device), mask.to(device)
    gate_outputs: list[Tensor] = []

    def capture_gate(_module: nn.Module, _inputs: tuple[Any, ...], output: tuple[Tensor, Tensor]) -> None:
        gate_outputs.append(output[1].detach())

    handle = module.event_former.state_gate.register_forward_hook(capture_gate)
    module.eval()
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=amp):
        radar_output = module.radar_encoder.forward_sequence(
            batch.rd,
            batch.md,
            sensor_index=batch.sensor_index,
            frame_mask=batch.frame_mask,
        )
        radar_key = torch.nn.functional.normalize(
            masked_token_mean(radar_output.radar_tokens, radar_output.radar_token_mask).float(),
            dim=-1,
        )
        output = module.training_step(batch, ids, mask)
        if output.stream_class_logits is None or output.commit_class_logits is None:
            raise RuntimeError("PARrad Stage 2 diagnostics require action class logits")
        action_diagnostics = action_classification_diagnostics(
            output.stream_class_logits,
            output.commit_class_logits,
            parrad_event_targets(batch.labels, batch.frame_mask),
        )
        caption_ids = ids[:, :-1]
        caption_mask = mask[:, :-1]
        zero_logits = module.event_former.adapter.caption_logits(
            torch.zeros_like(output.commit_event_tokens), caption_ids, caption_mask
        )
        shuffled_logits = module.event_former.adapter.caption_logits(
            output.commit_event_tokens.roll(1, 0), caption_ids, caption_mask
        )
        zero_text = causal_caption_cross_entropy(
            zero_logits, module.event_former.config.event_tokens, ids, mask
        )
        shuffled_text = causal_caption_cross_entropy(
            shuffled_logits, module.event_former.config.event_tokens, ids, mask
        )
    handle.remove()

    updates = output.stream_state_norms.shape[1]
    frame_step = module.radar_encoder.config.frames_per_update
    update_mask = batch.frame_mask.reshape(batch.frame_mask.shape[0], updates, frame_step).any(-1)
    norm_by_update = []
    gate_by_update = []
    for index in range(updates):
        valid = update_mask[:, index]
        if not valid.any():
            continue
        norms = output.stream_state_norms[valid, index].detach().float()
        gates = gate_outputs[index][valid].float()
        norm_by_update.append(
            {"update": index + 1, "samples": int(valid.sum()), "mean": float(norms.mean()), "max": float(norms.max())}
        )
        gate_by_update.append({"update": index + 1, **summarize_gate_values(gates)})
    cross_attention = [
        {"layer": index, "alpha": float(layer.alpha.detach()), "tanh_alpha": float(torch.tanh(layer.alpha.detach()))}
        for index, layer in enumerate(module.event_former.cross_attention)
    ]
    diagnostics = {
        "representations": {
            "radar": representation_diagnostics(radar_key),
            "stream": representation_diagnostics(output.stream_key),
            "commit": representation_diagnostics(output.commit_key),
            "text": representation_diagnostics(output.text_key),
        },
        "alignment": {
            "stream_to_text": alignment_diagnostics(output.stream_key, output.text_key),
            "commit_to_text": alignment_diagnostics(output.commit_key, output.text_key),
        },
        "caption_counterfactual": {
            "actual_event_tokens": float(output.losses.text.detach()),
            "zero_event_tokens": float(zero_text.detach()),
            "shuffled_event_tokens": float(shuffled_text.detach()),
        },
        "action_classification": action_diagnostics,
        "state_dynamics": {
            "cross_attention_alpha": cross_attention,
            "norm_by_update": norm_by_update,
            "gate_by_update": gate_by_update,
            "commit_gate_mean": float(gate_outputs[-1].float().mean()),
            "commit_gate": summarize_gate_values(gate_outputs[-1]),
        },
    }
    return diagnostics, ids, mask


def run_diagnostics(
    config_path: Path,
    output_path: Path,
    *,
    checkpoint: Path | None,
    metrics_path: Path | None,
    split: str,
) -> Path:
    config = load_experiment_config(config_path)
    seed = int(config.training.get("seed", 42))
    seed_everything(seed)
    dataset = _dataset(config, split)
    num_classes = int(config.stage.get("num_classes", 14))
    selected = _balanced_samples(dataset, num_classes, per_class=1)
    diagnostic_batch = parrad_collate(selected, frames_per_update=config.model.frames_per_update)
    gradient_batch = parrad_collate(selected[:4], frames_per_update=config.model.frames_per_update)
    labels = [_label(dataset[index]) for index in range(len(dataset))]

    module, tokenizer = _build_parrad_stage2(config)
    if checkpoint is not None:
        load_training_checkpoint(checkpoint, module)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module.to(device)
    amp = device.type == "cuda"
    diagnostics, _, _ = _state_and_key_diagnostics(module, tokenizer, diagnostic_batch, device, amp)

    gradient_batch = _move_parrad(gradient_batch, device)
    captions = select_event_captions(
        gradient_batch.labels,
        gradient_batch.frame_mask,
        default_parrad_event_templates(),
        epoch=0,
    )
    ids, mask = _tokenize_captions(tokenizer, captions)
    diagnostics["gradient_norms"] = _gradient_norms(
        module, gradient_batch, ids.to(device), mask.to(device), amp
    )
    if checkpoint is not None:
        diagnostics["optimizer_bootstrap_probe"] = _optimizer_bootstrap_probe(
            module,
            gradient_batch,
            ids.to(device),
            mask.to(device),
            config,
            checkpoint,
            amp,
        )
    diagnostics["false_negatives"] = false_negative_diagnostics(
        labels, int(config.training.get("batch_size", 4)), seed
    )
    diagnostics["dataset"] = {
        "split": split,
        "event_windows": len(dataset),
        "class_counts": {str(label): labels.count(label) for label in range(num_classes)},
        "balanced_probe_samples": len(selected),
    }
    if metrics_path is not None:
        diagnostics["training_history"] = _history(metrics_path)
    diagnostics["checkpoint"] = None if checkpoint is None else str(checkpoint)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--metrics", type=Path)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="validation")
    args = parser.parse_args()
    print(
        run_diagnostics(
            args.config,
            args.output,
            checkpoint=args.checkpoint,
            metrics_path=args.metrics,
            split=args.split,
        )
    )


if __name__ == "__main__":
    main()
