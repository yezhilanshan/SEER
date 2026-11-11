"""PARrad Stage P2: streaming/commit Event Token semantic alignment."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from continuous_mmwave.data_parrad import PARradBatch
from continuous_mmwave.models.event_former import StreamingSemanticEventFormer
from continuous_mmwave.models.parrad_encoder import PARradRadarEncoder
from continuous_mmwave.models.semantic import SemanticProjection
from continuous_mmwave.stages.stage2 import (
    Stage2Loss,
    Stage2Output,
    causal_caption_cross_entropy,
    cosine_consistency_loss,
    supervised_symmetric_alignment_loss,
)


_DEFAULT_EVENT_TEMPLATES: dict[int, tuple[str, str, str]] = {
    0: (
        "a person falls onto the room floor",
        "the person falls down on the floor",
        "someone has a fall in the room",
    ),
    1: (
        "a person gets into bed",
        "the person moves into the bed",
        "someone climbs into bed",
    ),
    2: (
        "a person gets out of bed",
        "the person leaves the bed",
        "someone rises and moves out of bed",
    ),
    3: (
        "a person lies in bed",
        "the person is lying on the bed",
        "someone rests in a lying position on the bed",
    ),
    4: (
        "a person rolls over in bed",
        "the person turns while lying in bed",
        "someone changes their lying position in bed",
    ),
    5: (
        "a person sits down on the bed",
        "the person lowers into a seated position on the bed",
        "someone takes a seat on the bed",
    ),
    6: (
        "a person sits down on a chair",
        "the person lowers into a seated position on the chair",
        "someone takes a seat on the chair",
    ),
    7: (
        "a person sits in bed",
        "the person remains seated on the bed",
        "someone is sitting upright in bed",
    ),
    8: (
        "a person stands up from the bed",
        "the person rises from the bed",
        "someone gets to a standing position from the bed",
    ),
    9: (
        "a person stands up from the chair",
        "the person rises from the chair",
        "someone gets to a standing position from the chair",
    ),
    10: (
        "a person stands up from the room floor",
        "the person rises from the floor",
        "someone gets to a standing position from the floor",
    ),
    11: (
        "a person walks toward the bed",
        "the person is moving to the bed",
        "someone approaches the bed",
    ),
    12: (
        "a person walks toward the chair",
        "the person is moving to the chair",
        "someone approaches the chair",
    ),
    13: (
        "a person walks into the room",
        "the person is moving through the room",
        "someone walks toward the room area",
    ),
}


def default_parrad_event_templates() -> dict[int, tuple[str, str, str]]:
    """Return a copy of the versioned closed-set PARrad caption templates."""
    return dict(_DEFAULT_EVENT_TEMPLATES)


def select_event_captions(
    labels: Tensor,
    frame_mask: Tensor,
    templates: Mapping[int, tuple[str, str, str]],
    *,
    epoch: int,
) -> tuple[str, ...]:
    """Select deterministic rotating templates for single-label event segments."""
    if labels.shape != frame_mask.shape or labels.ndim != 2:
        raise ValueError("labels and frame_mask must share [B,T]")
    captions = []
    for index in range(labels.shape[0]):
        valid = labels[index, frame_mask[index].bool()]
        valid = valid[valid.ne(-100)]
        unique = torch.unique(valid)
        if unique.numel() != 1:
            raise ValueError("each P2 event segment must contain exactly one activity label")
        label = int(unique.item())
        if label not in templates:
            raise KeyError(f"no semantic templates for PARrad label {label}")
        choices = templates[label]
        if len(choices) != 3 or any(not choice.strip() for choice in choices):
            raise ValueError("every PARrad label requires exactly three non-empty templates")
        captions.append(choices[(epoch + index) % len(choices)])
    return tuple(captions)


def parrad_event_targets(labels: Tensor, frame_mask: Tensor) -> Tensor:
    """Return one action target per label-run sample and reject mixed runs."""
    if labels.shape != frame_mask.shape or labels.ndim != 2:
        raise ValueError("labels and frame_mask must share [B,T]")
    targets: list[Tensor] = []
    for row_labels, row_mask in zip(labels, frame_mask, strict=True):
        valid = row_labels[row_mask.bool() & row_labels.ne(-100)].unique()
        if valid.numel() != 1:
            raise ValueError("each Stage 2 sample must contain a single action label")
        targets.append(valid[0])
    return torch.stack(targets).long()


def set_parrad_stage2_phase(module: PARradStage2Module, phase: str) -> None:
    """Apply the trainable-parameter contract for semantic warmup or joint training."""
    if phase not in {"semantic_warmup", "joint"}:
        raise ValueError("Stage 2 phase must be semantic_warmup or joint")
    module.phase = phase
    for parameter in module.radar_encoder.parameters():
        parameter.requires_grad = False
    for parameter in module.event_former.adapter.parameters():
        parameter.requires_grad = phase == "joint" and id(parameter) in module._joint_adapter_parameter_ids
    for parameter in module.event_former.cross_attention.parameters():
        parameter.requires_grad = True
    for parameter in module.event_former.state_gate.parameters():
        parameter.requires_grad = True
    for parameter in module.event_former.output_projection.parameters():
        parameter.requires_grad = True
    module.event_former.base_event_tokens.requires_grad = True
    for parameter in module.semantic_projection.parameters():
        parameter.requires_grad = True
    for parameter in module.auxiliary_classifier.parameters():
        parameter.requires_grad = True


class PARradStage2Module(nn.Module):
    """Reuse the existing Event Former losses with PARrad Radar Tokens."""

    def __init__(
        self,
        radar_encoder: PARradRadarEncoder,
        event_former: StreamingSemanticEventFormer,
        semantic_projection: SemanticProjection,
        align_weight: float = 1,
        consistency_weight: float = 1,
        contrastive_temperature: float = 0.07,
        checkpoint_streaming: bool = True,
        num_classes: int = 14,
        auxiliary_weight: float = 0.5,
        text_weight: float = 1.0,
        warmup_consistency_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if radar_encoder.config.radar_dim != event_former.config.radar_dim:
            raise ValueError("PARrad encoder and Event Former radar dimensions must match")
        if radar_encoder.config.radar_tokens_per_update != event_former.config.radar_tokens_per_update:
            raise ValueError("PARrad encoder and Event Former token counts must match")
        self.radar_encoder = radar_encoder
        self.event_former = event_former
        self.semantic_projection = semantic_projection
        self.align_weight = align_weight
        self.consistency_weight = consistency_weight
        self.contrastive_temperature = contrastive_temperature
        self.checkpoint_streaming = checkpoint_streaming
        self.auxiliary_classifier = nn.Linear(semantic_projection.projection[-1].out_features, num_classes)
        self.auxiliary_weight = auxiliary_weight
        self.text_weight = text_weight
        self.warmup_consistency_weight = warmup_consistency_weight
        self.phase = "joint"
        self._joint_adapter_parameter_ids = frozenset(
            id(parameter) for parameter in event_former.adapter.parameters() if parameter.requires_grad
        )

    def _stream_view(self, tokens: Tensor, token_mask: Tensor, prompt_cache: Any) -> tuple[Tensor, Tensor]:
        size = self.event_former.config.radar_tokens_per_update
        if tokens.shape[1] == 0 or tokens.shape[1] % size:
            raise ValueError("PARrad token sequence must contain complete updates")
        updates = tokens.shape[1] // size
        state = self.event_former.initial_state(tokens.shape[0])
        update_mask = token_mask.reshape(tokens.shape[0], updates, size).any(-1)
        total = update_mask.sum(1).clamp_min(1)
        completed = torch.zeros_like(total)
        norms = []
        for index in range(updates):
            valid = update_mask[:, index]
            completed += valid
            radar = tokens[:, index * size : (index + 1) * size]
            mask = token_mask[:, index * size : (index + 1) * size]
            maturity = completed.to(tokens.dtype) / total.to(tokens.dtype)
            if self.training and self.checkpoint_streaming:

                def update(
                    current: Tensor,
                    current_radar: Tensor,
                    current_maturity: Tensor,
                    current_mask: Tensor,
                ) -> Tensor:
                    return self.event_former.stream_update(
                        current, current_radar, current_maturity, current_mask, prompt_cache
                    ).event_tokens

                proposal = checkpoint(update, state, radar, maturity, mask, use_reentrant=False)
            else:
                proposal = self.event_former.stream_update(
                    state, radar, maturity, mask, prompt_cache
                ).event_tokens
            state = torch.where(valid[:, None, None], proposal, state)
            norms.append(torch.linalg.vector_norm(state, dim=(-2, -1)))
        return state, torch.stack(norms, 1)

    def training_step(self, batch: PARradBatch, caption_ids: Tensor, caption_mask: Tensor) -> Stage2Output:
        radar = self.radar_encoder.forward_sequence(
            batch.rd,
            batch.md,
            sensor_index=batch.sensor_index,
            frame_mask=batch.frame_mask,
        )
        cache = self.event_former.build_prompt_cache(radar.radar_tokens.shape[0])
        stream, norms = self._stream_view(radar.radar_tokens, radar.radar_token_mask, cache)
        commit = self.event_former.commit_refine(
            radar.radar_tokens, radar.radar_token_mask, cache
        ).event_tokens
        stream_key = self.semantic_projection(stream)
        commit_key = self.semantic_projection(commit)
        targets = parrad_event_targets(batch.labels, batch.frame_mask)
        consistency = cosine_consistency_loss(stream_key, commit_key)
        if self.phase == "semantic_warmup":
            text_key = torch.zeros_like(stream_key)
            align = (stream_key.sum() + commit_key.sum()) * 0
            text = align
        else:
            text_hidden = self.event_former.adapter.encode_text(caption_ids, caption_mask)
            text_key = self.semantic_projection(text_hidden, caption_mask)
            align = (
                supervised_symmetric_alignment_loss(stream_key, text_key, targets, self.contrastive_temperature)
                + supervised_symmetric_alignment_loss(commit_key, text_key, targets, self.contrastive_temperature)
            ) / 2
            caption_logits = self.event_former.adapter.caption_logits(
                commit, caption_ids[:, :-1], caption_mask[:, :-1]
            )
            text = causal_caption_cross_entropy(
                caption_logits, self.event_former.config.event_tokens, caption_ids, caption_mask
            )
        return Stage2Output(
            Stage2Loss(text, align, consistency),
            stream,
            commit,
            stream_key,
            commit_key,
            text_key,
            norms,
            self.auxiliary_classifier(stream_key),
            self.auxiliary_classifier(commit_key),
        )
