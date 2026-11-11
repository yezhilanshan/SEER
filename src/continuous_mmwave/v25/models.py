"""Age-free trainable modules. Age is exclusively a controller diagnostic."""
from __future__ import annotations

import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from ..models.transition import LocalChangeHead, PrototypeState, update_prototype, _require_token_batch


class AgeFreeStateGate(nn.Module):
    architecture = 'v25-age-free-tanh'

    def __init__(self, dim: int, initial_update: float = .1):
        super().__init__()
        if dim <= 0 or not 0 < initial_update < 1:
            raise ValueError('invalid gate dimensions/initial update')
        self.norm = nn.LayerNorm(dim)
        self.network = nn.Sequential(nn.Linear(2*dim, dim), nn.GELU(), nn.Linear(dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, math.atanh(initial_update))

    def forward(self, previous: Tensor, proposal: Tensor, _legacy_maturity=None):
        # Compatibility with the shared EventFormer call signature; never read age.
        if previous.shape != proposal.shape or previous.ndim != 3:
            raise ValueError('gate needs matching [B,E,D] states')
        delta = proposal - previous
        alpha = self.network(torch.cat((self.norm(previous), self.norm(delta)), -1)).tanh()
        return previous + alpha * delta, alpha


class PositiveCoordinateGate(AgeFreeStateGate):
    """Development bridge retaining the fitted positive gate coordinate.

    Fixed maturity=1 is folded into initialization, never an input. This is
    intentionally not the signed/rollback gate of the final v2.5 architecture.
    """
    architecture = 'v25-age-free-positive-coordinate-v1'

    def forward(self, previous: Tensor, proposal: Tensor, _legacy_maturity=None):
        if previous.shape != proposal.shape or previous.ndim != 3:
            raise ValueError('gate needs matching [B,E,D] states')
        delta = proposal - previous
        z = self.network(torch.cat((self.norm(previous), self.norm(delta)), -1))
        alpha = (.5 * F.softplus(z + math.log(2.))).tanh()
        return previous + alpha * delta, alpha


def install_positive_coordinate_gate(event_former):
    """Copy a fitted legacy sigmoid gate; never overwrite a migrated gate."""
    legacy = event_former.state_gate
    dim = legacy.norm.normalized_shape[0]
    if legacy.network[0].in_features != 2*dim+1:
        raise ValueError('fitted legacy gate with maturity column required')
    gate = PositiveCoordinateGate(dim).to(legacy.network[0].weight)
    with torch.no_grad():
        gate.norm.load_state_dict(legacy.norm.state_dict())
        gate.network[0].weight.copy_(legacy.network[0].weight[:, :2*dim])
        gate.network[0].bias.copy_(legacy.network[0].bias + legacy.network[0].weight[:, -1])
        gate.network[2].load_state_dict(legacy.network[2].state_dict())
    event_former.state_gate = gate
    return event_former


def install_age_free_gate(event_former):
    """Explicit architecture migration; the gate is NEW and requires training.

    Do this before loading a versioned v2.5 checkpoint. It intentionally does
    not copy sigmoid/logit gate weights into a tanh architecture.
    """
    config = event_former.config
    gate = AgeFreeStateGate(config.llm_dim, config.event_state_initial_update)
    gate = gate.to(event_former.output_projection.weight)
    event_former.state_gate = gate
    return event_former


class SemanticDifference(nn.Module):
    def __init__(self, semantic_dim=128, hidden_dim=256):
        super().__init__()
        if semantic_dim <= 0 or hidden_dim < 4:
            raise ValueError('invalid semantic dimensions')
        self.network = nn.Sequential(nn.Linear(4*semantic_dim+1, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim//4), nn.GELU(), nn.Linear(hidden_dim//4,1))

    def forward(self, active, candidate):
        if active.shape != candidate.shape or active.ndim != 2:
            raise ValueError('semantic keys must match [B,D]')
        if not torch.isfinite(active).all() or not torch.isfinite(candidate).all():
            raise ValueError('nonfinite semantic keys')
        distance = 1-F.cosine_similarity(active, candidate, dim=-1)
        feature = torch.cat((active, candidate, (active-candidate).abs(), active*candidate, distance[:,None]),-1)
        return self.network(feature).squeeze(-1)


class RadarTransition(nn.Module):
    def __init__(self, radar_dim=256, prototype_dim=128, tau_fast_s=2., eps=1e-6):
        super().__init__()
        if min(radar_dim, prototype_dim, tau_fast_s, eps) <= 0:
            raise ValueError('invalid radar dimensions/constants')
        self.project = nn.Sequential(nn.Linear(radar_dim,prototype_dim),nn.GELU(),nn.Linear(prototype_dim,prototype_dim))
        self.classifier = nn.Sequential(nn.Linear(4*prototype_dim+1,128),nn.GELU(),nn.Linear(128,32),nn.GELU(),nn.Linear(32,1))
        self.tau_fast_s, self.eps = tau_fast_s, eps

    def forward(self, tokens, mask, state, update_period_s):
        valid = _require_token_batch(tokens, mask, 'radar tokens')
        weights = valid.to(tokens.dtype).unsqueeze(-1)
        value = F.normalize(self.project((tokens*weights).sum(1)/weights.sum(1)),dim=-1,eps=self.eps)
        updated = (PrototypeState.initialize(value) if state is None else
                   update_prototype(state,value,update_period_s,self.tau_fast_s))
        standardized = (updated.fast-updated.mean).abs()/torch.sqrt(updated.variance+self.eps)
        distance = 1-F.cosine_similarity(updated.fast,updated.mean,dim=-1)
        feature = torch.cat((updated.fast,updated.mean,standardized,updated.fast*updated.mean,distance[:,None]),-1)
        return self.classifier(feature).squeeze(-1), updated


class EvidenceProjection(nn.Module):
    """Normalized physical attributes; target ranges are defined by the dataset."""
    def __init__(self, event_dim=896, evidence_dim=4):
        super().__init__()
        if event_dim <= 0 or evidence_dim <= 0:
            raise ValueError('invalid evidence dimensions')
        self.network = nn.Sequential(nn.LayerNorm(event_dim),nn.Linear(event_dim,128),nn.GELU(),
                                     nn.Linear(128,evidence_dim),nn.Sigmoid())

    def forward(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1] == 0:
            raise ValueError('event tokens must be [B,E,D]')
        # Native Qwen returns FP16/BF16 events while new heads stay FP32.
        return self.network(tokens.to(self.network[1].weight.dtype).mean(1))


class V25Heads(nn.Module):
    architecture = 'v25-sequential-evidence'

    def __init__(self, radar_dim=256, event_dim=896, semantic_dim=128, evidence_dim=4):
        super().__init__()
        self.settings = dict(radar_dim=radar_dim,event_dim=event_dim,semantic_dim=semantic_dim,evidence_dim=evidence_dim)
        self.local = LocalChangeHead(radar_dim)
        self.radar = RadarTransition(radar_dim)
        self.semantic = SemanticDifference(semantic_dim)
        self.evidence = EvidenceProjection(event_dim, evidence_dim)


def physical_difference(active, candidate, weights=None):
    if active.shape != candidate.shape or active.ndim != 2 or active.shape[-1] == 0:
        raise ValueError('physical attributes must match [B,M]')
    if not torch.isfinite(active).all() or not torch.isfinite(candidate).all():
        raise ValueError('physical attributes must be finite')
    weights = torch.ones_like(active[0]) if weights is None else torch.as_tensor(weights,device=active.device,dtype=active.dtype)
    if weights.shape != (active.shape[-1],) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError('invalid physical evidence weights')
    return ((active-candidate).abs()*weights).mean(-1)
