"""PARrad Stage P1: temporary closed-set frame supervision."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional

from continuous_mmwave.data_parrad import PARradBatch
from continuous_mmwave.models.parrad_encoder import (
    PARradAnyStreamState,
    PARradCausalMDGRUEncoder,
    PARradCausalRadMambaEncoder,
    PARradEncoderOutput,
    PARradRadarEncoder,
)


def frame_classification_loss(logits: Tensor, labels: Tensor, class_weights: Tensor | None = None) -> Tensor:
    """Cross entropy over real frames; padded label -100 is ignored."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits/labels must have shapes [B,T,C]/[B,T]")
    if not labels.ne(-100).any():
        raise ValueError("PARrad Stage P1 requires at least one supervised frame")
    return functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), weight=class_weights, ignore_index=-100)


def token_classification_targets(labels: Tensor, frames_per_update: int) -> Tensor:
    """Return one label per update, masking mixed or incompletely supervised updates."""
    if labels.ndim != 2:
        raise ValueError("labels must have shape [B,T]")
    if frames_per_update <= 0 or labels.shape[1] % frames_per_update:
        raise ValueError("frame count must be divisible by a positive frames_per_update")
    grouped = labels.reshape(labels.shape[0], -1, frames_per_update)
    first = grouped[..., 0]
    supervised = grouped.ne(-100).all(-1) & grouped.eq(first.unsqueeze(-1)).all(-1)
    return torch.where(supervised, first, torch.full_like(first, -100))


def boundary_aware_temporal_smoothness_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Jensen-Shannon consistency inside a true label run, never across boundaries."""
    if logits.ndim != 3 or labels.shape != logits.shape[:2]:
        raise ValueError("logits/labels must have shapes [B,T,C]/[B,T]")
    if logits.shape[1] < 2:
        return logits.sum() * 0
    pair_mask = (
        labels[:, 1:].ne(-100)
        & labels[:, :-1].ne(-100)
        & labels[:, 1:].eq(labels[:, :-1])
    )
    if not pair_mask.any():
        return logits.sum() * 0
    log_probabilities = functional.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    previous_probability = probabilities[:, :-1]
    current_probability = probabilities[:, 1:]
    mixture = 0.5 * (previous_probability + current_probability)
    log_mixture = mixture.clamp_min(torch.finfo(mixture.dtype).tiny).log()
    previous_kl = (previous_probability * (log_probabilities[:, :-1] - log_mixture)).sum(-1)
    current_kl = (current_probability * (log_probabilities[:, 1:] - log_mixture)).sum(-1)
    return (0.5 * (previous_kl + current_kl))[pair_mask].mean()


@dataclass(frozen=True)
class PARradStage1Loss:
    loss: Tensor
    accuracy: Tensor
    classification_loss: Tensor
    temporal_smoothness_loss: Tensor
    token_classification_loss: Tensor
    token_accuracy: Tensor
    supervised_tokens: Tensor


class PARradStage1Module(nn.Module):
    """Warm up the PARrad encoder with a removable activity classifier."""

    def __init__(
        self,
        encoder: PARradRadarEncoder | PARradCausalRadMambaEncoder | PARradCausalMDGRUEncoder,
        num_classes: int,
        class_weights: Tensor | None = None,
        temporal_smoothness_weight: float = 0.0,
        frame_classification_weight: float = 1.0,
        token_classification_weight: float = 0.0,
    ) -> None:
        super().__init__()
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")
        if class_weights is not None and class_weights.shape != (num_classes,):
            raise ValueError("class_weights must have one value per class")
        if temporal_smoothness_weight < 0:
            raise ValueError("temporal_smoothness_weight must be non-negative")
        if frame_classification_weight < 0 or token_classification_weight < 0:
            raise ValueError("classification weights must be non-negative")
        if frame_classification_weight == 0 and token_classification_weight == 0:
            raise ValueError("at least one classification weight must be positive")
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.config.radar_dim, num_classes)
        self.token_classifier = nn.Linear(encoder.config.radar_dim, num_classes)
        self.register_buffer("class_weights", class_weights)
        self.temporal_smoothness_weight = float(temporal_smoothness_weight)
        self.frame_classification_weight = float(frame_classification_weight)
        self.token_classification_weight = float(token_classification_weight)

    def forward(
        self,
        batch: PARradBatch,
        state: PARradAnyStreamState | None = None,
    ) -> tuple[Tensor, PARradEncoderOutput]:
        output = self.encoder.forward_sequence(
            batch.rd,
            batch.md,
            sensor_index=batch.sensor_index,
            frame_mask=batch.frame_mask,
            state=state,
        )
        return self.classifier(output.frame_hidden), output

    def token_logits(self, output: PARradEncoderOutput) -> Tensor:
        """Classify the masked mean of the four Radar Tokens produced per update."""
        size = self.encoder.config.radar_tokens_per_update
        tokens = output.radar_tokens
        mask = output.radar_token_mask
        if tokens.shape[:2] != mask.shape or tokens.shape[1] % size:
            raise ValueError("radar tokens must contain complete masked updates")
        grouped_tokens = tokens.reshape(tokens.shape[0], -1, size, tokens.shape[-1])
        grouped_mask = mask.reshape(mask.shape[0], -1, size)
        denominator = grouped_mask.sum(-1, keepdim=True).clamp_min(1)
        pooled = (grouped_tokens * grouped_mask.unsqueeze(-1)).sum(2) / denominator
        return self.token_classifier(pooled)

    def loss_from_logits(
        self,
        logits: Tensor,
        labels: Tensor,
        token_logits: Tensor | None = None,
        token_labels: Tensor | None = None,
    ) -> PARradStage1Loss:
        classification = frame_classification_loss(logits, labels, self.class_weights)
        smoothness = boundary_aware_temporal_smoothness_loss(logits, labels)
        valid = labels.ne(-100)
        accuracy = (logits.argmax(-1)[valid] == labels[valid]).float().mean()
        if token_logits is None or token_labels is None:
            if self.token_classification_weight:
                raise ValueError("token supervision requires token logits and labels")
            token_loss = logits.sum() * 0
            token_accuracy = logits.new_zeros(())
            supervised_tokens = logits.new_zeros((), dtype=torch.long)
        else:
            token_valid = token_labels.ne(-100)
            supervised_tokens = token_valid.sum()
            if token_valid.any():
                token_loss = functional.cross_entropy(
                    token_logits.flatten(0, 1),
                    token_labels.flatten(),
                    weight=self.class_weights,
                    ignore_index=-100,
                )
                token_accuracy = (
                    token_logits.argmax(-1)[token_valid] == token_labels[token_valid]
                ).float().mean()
            else:
                token_loss = token_logits.sum() * 0
                token_accuracy = token_logits.new_zeros(())
        total = (
            self.frame_classification_weight * classification
            + self.token_classification_weight * token_loss
            + self.temporal_smoothness_weight * smoothness
        )
        return PARradStage1Loss(
            total,
            accuracy,
            classification,
            smoothness,
            token_loss,
            token_accuracy,
            supervised_tokens,
        )

    def training_step(self, batch: PARradBatch) -> PARradStage1Loss:
        logits, output = self(batch)
        token_logits = self.token_logits(output)
        token_labels = token_classification_targets(
            batch.labels, self.encoder.config.frames_per_update
        )
        return self.loss_from_logits(logits, batch.labels, token_logits, token_labels)

    def export_encoder_state(self) -> dict[str, Tensor]:
        """Return only deployable encoder weights, excluding the P1 classifier."""
        return dict(self.encoder.state_dict())
