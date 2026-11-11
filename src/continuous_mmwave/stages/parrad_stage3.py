"""Trainable Stage P3 transition heads over frozen Stage P2 features."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from continuous_mmwave.models.transition import (
    LocalChangeHead,
    MonotonicVerifyHead,
    RadarTransitionHead,
    PrototypeState,
    SemanticDifferenceHead,
    candidate_stability,
)
from continuous_mmwave.parrad_stage3_data import (
    PhysicalBatch,
    PhysicalSequenceBatch,
    SemanticPairBatch,
    VerifyBatch,
    VerifyFeatureBatch,
)


class PARradStage3Module(nn.Module):
    """Four small heads trained in isolated physical, semantic, and Verify phases."""

    def __init__(
        self,
        radar_dim: int = 256,
        semantic_dim: int = 128,
        prototype_dim: int = 128,
        update_period_s: float = 0.18,
        tau_fast_s: float = 2.0,
        tau_stability: float = 0.1,
        physical_tbptt_steps: int = 32,
    ) -> None:
        super().__init__()
        if update_period_s <= 0 or tau_stability <= 0 or physical_tbptt_steps <= 0:
            raise ValueError("Stage P3 timing constants must be positive")
        self.local_change = LocalChangeHead(radar_dim)
        self.radar_transition = RadarTransitionHead(radar_dim, prototype_dim, tau_fast_s)
        self.semantic_difference = SemanticDifferenceHead(semantic_dim)
        self.verify = MonotonicVerifyHead()
        self.update_period_s = update_period_s
        self.tau_stability = tau_stability
        self.physical_tbptt_steps = physical_tbptt_steps
        self.phase: Literal["physical", "semantic", "verify"] = "physical"
        self.set_phase("physical")

    def set_phase(self, phase: Literal["physical", "semantic", "verify"]) -> None:
        if phase not in {"physical", "semantic", "verify"}:
            raise ValueError("Stage P3 phase must be physical, semantic, or verify")
        self.phase = phase
        prefixes = {
            "physical": ("local_change", "radar_transition"),
            "semantic": ("semantic_difference",),
            "verify": ("verify",),
        }[phase]
        for name, parameter in self.named_parameters():
            parameter.requires_grad = name.startswith(prefixes)

    @staticmethod
    def _bce(logits: Tensor, target: Tensor, weight: Tensor | None = None) -> Tensor:
        target = target.to(device=logits.device, dtype=logits.dtype)
        losses = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if weight is None:
            return losses.mean()
        weight = weight.to(device=logits.device, dtype=logits.dtype)
        if weight.shape != losses.shape or not torch.isfinite(weight).all() or (weight < 0).any():
            raise ValueError("sample weights must be finite, non-negative, and match targets")
        denominator = weight.sum()
        if denominator <= 0:
            raise ValueError("sample weights must have positive sum")
        return (losses * weight).sum() / denominator

    @staticmethod
    def _detach_prototype(state: PrototypeState) -> PrototypeState:
        return PrototypeState(
            fast=state.fast.detach(),
            mean=state.mean.detach(),
            m2=state.m2.detach(),
            count=state.count,
            age_seconds=state.age_seconds,
            alpha=state.alpha,
        )

    def _physical_sequence_step(self, batch: PhysicalSequenceBatch) -> dict[str, Tensor]:
        if batch.tokens.ndim != 4 or batch.tokens.shape[1] < 2:
            raise ValueError("physical sequences must have shape [B,U,R,D] with U >= 2")
        batch_size, update_count, radar_tokens, radar_dim = batch.tokens.shape
        pair_mask = batch.mask[:, :-1] & batch.mask[:, 1:]
        local_logits = self.local_change(
            batch.tokens[:, :-1].reshape(-1, radar_tokens, radar_dim),
            batch.tokens[:, 1:].reshape(-1, radar_tokens, radar_dim),
            pair_mask.reshape(-1, radar_tokens),
        ).reshape(batch_size, update_count - 1)
        _, state = self.radar_transition(
            batch.tokens[:, 0], batch.mask[:, 0], None, self.update_period_s
        )
        radar_logits: list[Tensor] = []
        for update_index in range(1, update_count):
            if update_index % self.physical_tbptt_steps == 0:
                state = self._detach_prototype(state)
            logits, state = self.radar_transition(
                batch.tokens[:, update_index],
                batch.mask[:, update_index],
                state,
                self.update_period_s,
            )
            radar_logits.append(logits)
        stacked_radar = torch.stack(radar_logits, dim=1)
        valid = batch.valid.bool()
        if valid.shape != local_logits.shape or not valid.any():
            raise ValueError("physical sequence validity mask must select at least one transition")
        local_loss = self._bce(
            local_logits[valid], batch.local_target[valid], batch.weight[valid]
        )
        radar_loss = self._bce(
            stacked_radar[valid], batch.radar_target[valid], batch.weight[valid]
        )
        return {
            "loss": local_loss + radar_loss,
            "local_loss": local_loss,
            "radar_loss": radar_loss,
            "local_logits": local_logits,
            "radar_logits": stacked_radar,
            "valid": valid,
            "radar_state_count": state.count,
        }

    def physical_step(self, batch: PhysicalBatch | PhysicalSequenceBatch) -> dict[str, Tensor]:
        if isinstance(batch, PhysicalSequenceBatch):
            return self._physical_sequence_step(batch)
        local_logits = self.local_change(batch.previous, batch.current, batch.mask)
        _, previous_state = self.radar_transition(
            batch.previous,
            batch.mask,
            None,
            self.update_period_s,
        )
        radar_logits, _ = self.radar_transition(
            batch.current,
            batch.mask,
            previous_state,
            self.update_period_s,
        )
        local_loss = self._bce(local_logits, batch.local_target)
        radar_loss = self._bce(radar_logits, batch.radar_target)
        return {
            "loss": local_loss + radar_loss,
            "local_loss": local_loss,
            "radar_loss": radar_loss,
            "local_logits": local_logits,
            "radar_logits": radar_logits,
        }

    def semantic_step(self, batch: SemanticPairBatch) -> dict[str, Tensor]:
        logits = self.semantic_difference(batch.active, batch.candidate, batch.maturity)
        return {"loss": self._bce(logits, batch.target), "logits": logits}

    def verify_step(self, batch: VerifyBatch | VerifyFeatureBatch) -> dict[str, Tensor]:
        if isinstance(batch, VerifyFeatureBatch):
            inputs = batch.inputs
            if inputs.ndim != 2 or inputs.shape[-1] != 3:
                raise ValueError("closed-loop Verify inputs must have shape [B,3]")
            logits = self.verify(inputs)
            return {
                "loss": self._bce(logits, batch.target, batch.weight),
                "logits": logits,
                "verify_inputs": inputs,
                "stability": inputs[:, 1],
                "difference_probability": inputs[:, 0],
            }
        stability = candidate_stability(batch.key_history, self.tau_stability).detach()
        semantic_was_training = self.semantic_difference.training
        self.semantic_difference.eval()
        try:
            with torch.no_grad():
                difference_probability = torch.sigmoid(
                    self.semantic_difference(
                        batch.active,
                        batch.key_history[:, -1],
                        batch.maturity,
                    )
                )
        finally:
            self.semantic_difference.train(semantic_was_training)
        inputs = torch.stack(
            (
                difference_probability.to(stability.dtype),
                stability,
                batch.maturity.to(stability.dtype),
            ),
            dim=-1,
        )
        logits = self.verify(inputs)
        return {
            "loss": self._bce(logits, batch.target, batch.weight),
            "logits": logits,
            "stability": stability,
            "difference_probability": difference_probability,
            "verify_inputs": inputs,
        }
