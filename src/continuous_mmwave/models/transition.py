"""Lightweight physical and semantic transition heads for PARrad Stage P3."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional


def _require_token_batch(tokens: Tensor, mask: Tensor, name: str) -> Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"{name} must have shape [B,R,D]")
    if mask.shape != tokens.shape[:2]:
        raise ValueError("radar mask must match the token batch")
    valid = mask.bool()
    if (~valid.any(dim=1)).any():
        raise ValueError("each sample requires at least one valid radar token")
    return valid


class LocalChangeHead(nn.Module):
    """Score change between consecutive four-token radar updates."""

    def __init__(self, radar_dim: int = 256, hidden_dim: int = 128, dropout: float = 0.1) -> None:
        super().__init__()
        if radar_dim <= 0 or hidden_dim <= 0:
            raise ValueError("radar_dim and hidden_dim must be positive")
        self.norm = nn.LayerNorm(radar_dim)
        self.input = nn.Sequential(nn.Linear(radar_dim * 3, hidden_dim), nn.GELU())
        self.attention = nn.Linear(hidden_dim, 1)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, max(1, hidden_dim // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(1, hidden_dim // 2), 1),
        )

    def forward(self, previous: Tensor, current: Tensor, mask: Tensor) -> Tensor:
        if previous.shape != current.shape:
            raise ValueError("previous and current radar tokens must share shape")
        valid = _require_token_batch(current, mask, "radar tokens")
        previous_normalized = self.norm(previous)
        current_normalized = self.norm(current)
        delta = current_normalized - previous_normalized
        hidden = self.input(torch.cat((delta, delta.abs(), current_normalized * previous_normalized), dim=-1))
        scores = self.attention(hidden).squeeze(-1)
        scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        pooled = torch.sum(hidden * scores.softmax(dim=1).unsqueeze(-1), dim=1)
        return self.output(pooled).squeeze(-1)


@dataclass(frozen=True, eq=False)
class PrototypeState:
    """Event-local fast EMA and slow Welford statistics."""

    fast: Tensor
    mean: Tensor
    m2: Tensor
    count: int
    age_seconds: float
    alpha: float

    @classmethod
    def initialize(cls, value: Tensor) -> PrototypeState:
        if value.ndim != 2 or value.shape[-1] == 0:
            raise ValueError("prototype value must have shape [B,D]")
        if not torch.isfinite(value).all():
            raise ValueError("prototype value must be finite")
        return cls(
            fast=value.clone(),
            mean=value.clone(),
            m2=torch.zeros_like(value),
            count=1,
            age_seconds=0.0,
            alpha=0.0,
        )

    @property
    def variance(self) -> Tensor:
        return self.m2 / max(self.count, 1)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PrototypeState):
            return NotImplemented
        return (
            torch.equal(self.fast, other.fast)
            and torch.equal(self.mean, other.mean)
            and torch.equal(self.m2, other.m2)
            and self.count == other.count
            and self.age_seconds == other.age_seconds
            and self.alpha == other.alpha
        )


def update_prototype(
    state: PrototypeState,
    value: Tensor,
    update_period_s: float,
    tau_fast_s: float,
) -> PrototypeState:
    """Advance physical prototype statistics by one fixed-duration update."""

    if value.shape != state.mean.shape:
        raise ValueError("prototype update must match state shape")
    if not torch.isfinite(value).all():
        raise ValueError("prototype update must be finite")
    if update_period_s <= 0 or tau_fast_s <= 0:
        raise ValueError("update period and EMA time constant must be positive")
    alpha = math.exp(-update_period_s / tau_fast_s)
    count = state.count + 1
    delta = value - state.mean
    mean = state.mean + delta / count
    m2 = state.m2 + delta * (value - mean)
    return PrototypeState(
        fast=alpha * state.fast + (1 - alpha) * value,
        mean=mean,
        m2=m2,
        count=count,
        age_seconds=state.age_seconds + update_period_s,
        alpha=alpha,
    )


class RadarTransitionHead(nn.Module):
    """Compare fast and slow event-local radar prototypes."""

    def __init__(
        self,
        radar_dim: int = 256,
        prototype_dim: int = 128,
        tau_fast_s: float = 2.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if radar_dim <= 0 or prototype_dim <= 0 or tau_fast_s <= 0 or eps <= 0:
            raise ValueError("RadarTransitionHead dimensions and constants must be positive")
        self.project = nn.Sequential(
            nn.Linear(radar_dim, prototype_dim),
            nn.GELU(),
            nn.Linear(prototype_dim, prototype_dim),
        )
        self.classifier = nn.Sequential(
            nn.Linear(prototype_dim * 4 + 2, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        self.tau_fast_s = tau_fast_s
        self.eps = eps

    def forward(
        self,
        radar_tokens: Tensor,
        token_mask: Tensor,
        state: PrototypeState | None,
        update_period_s: float,
    ) -> tuple[Tensor, PrototypeState]:
        valid = _require_token_batch(radar_tokens, token_mask, "radar_tokens")
        weights = valid.to(radar_tokens.dtype).unsqueeze(-1)
        pooled = (radar_tokens * weights).sum(dim=1) / weights.sum(dim=1)
        value = functional.normalize(self.project(pooled), dim=-1, eps=self.eps)
        updated = (
            PrototypeState.initialize(value)
            if state is None
            else update_prototype(state, value, update_period_s, self.tau_fast_s)
        )
        standardized = (updated.fast - updated.mean).abs() / torch.sqrt(updated.variance + self.eps)
        cosine_distance = 1 - functional.cosine_similarity(updated.fast, updated.mean, dim=-1)
        normalized_age = updated.age_seconds / (updated.age_seconds + self.tau_fast_s)
        age = torch.full_like(cosine_distance, normalized_age)
        feature = torch.cat(
            (
                updated.fast,
                updated.mean,
                standardized,
                updated.fast * updated.mean,
                cosine_distance[:, None],
                age[:, None],
            ),
            dim=-1,
        )
        return self.classifier(feature).squeeze(-1), updated


class SemanticDifferenceHead(nn.Module):
    """Score whether Candidate semantics differ from the frozen Active reference."""

    def __init__(self, semantic_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        if semantic_dim <= 0 or hidden_dim <= 0:
            raise ValueError("semantic_dim and hidden_dim must be positive")
        self.network = nn.Sequential(
            nn.Linear(semantic_dim * 4 + 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, max(1, hidden_dim // 4)),
            nn.GELU(),
            nn.Linear(max(1, hidden_dim // 4), 1),
        )

    def forward(self, active: Tensor, candidate: Tensor, maturity: Tensor) -> Tensor:
        if active.shape != candidate.shape or active.ndim != 2:
            raise ValueError("active and candidate keys must share shape [B,D]")
        if maturity.shape != (active.shape[0],):
            raise ValueError("maturity must have shape [B]")
        if not torch.isfinite(active).all() or not torch.isfinite(candidate).all() or not torch.isfinite(maturity).all():
            raise ValueError("semantic difference inputs must be finite")
        cosine_distance = 1 - functional.cosine_similarity(active, candidate, dim=-1)
        feature = torch.cat(
            (
                active,
                candidate,
                (active - candidate).abs(),
                active * candidate,
                cosine_distance[:, None],
                maturity[:, None].to(active.dtype),
            ),
            dim=-1,
        )
        return self.network(feature).squeeze(-1)


def candidate_stability(keys: Tensor, tau: float) -> Tensor:
    """Measure semantic stability over the last three Candidate keys."""

    if keys.ndim not in {2, 3}:
        raise ValueError("keys must have shape [H,D] or [B,H,D]")
    if tau <= 0:
        raise ValueError("tau must be positive")
    squeeze = keys.ndim == 2
    batched = keys.unsqueeze(0) if squeeze else keys
    if batched.shape[1] < 3:
        zeros = batched.new_zeros((batched.shape[0],))
        return zeros[0] if squeeze else zeros
    recent = batched[:, -3:]
    current = recent[:, -1]
    c1 = functional.cosine_similarity(current, recent[:, -2], dim=-1)
    c2 = functional.cosine_similarity(current, recent[:, -3], dim=-1)
    centroid = recent.mean(dim=1, keepdim=True)
    variance = (recent - centroid).square().mean(dim=(1, 2))
    result = (((c1 + c2) / 2) * torch.exp(-variance / tau)).clamp(0, 1)
    return result[0] if squeeze else result


class MonotonicVerifyHead(nn.Module):
    """Positive-weight fusion of semantic difference, stability, and maturity."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_weights = nn.Parameter(torch.zeros(3))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != 3:
            raise ValueError("Verify inputs must end with three features")
        return (inputs * self.positive_weights()).sum(dim=-1) + self.bias

    def positive_weights(self) -> Tensor:
        return functional.softplus(self.raw_weights)
