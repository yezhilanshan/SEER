"""Frozen-feature SAEF MVP heads and strictly causal event formation.

All intervals are half-open. A fire at frame i closes [start, i), and
frame i initializes the new event. Voltage diagnostics are pre-reset.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MVPConfig:
    feature_dim: int = 128
    hidden_dim: int = 128
    num_classes: int = 12
    frame_period_s: float = 1 / 15
    state_tau_s: float = 2.
    current_max: float = 3.
    minimum_duration_s: float | None = None
    unknown_confidence: float | None = None

    def __post_init__(self):
        for name in ('feature_dim', 'hidden_dim', 'num_classes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('frame_period_s', 'state_tau_s', 'current_max'):
            _positive(getattr(self, name), name)
        if self.minimum_duration_s is not None:
            _positive(self.minimum_duration_s, 'minimum_duration_s')
        if self.unknown_confidence is not None and not 0 <= self.unknown_confidence <= 1:
            raise ValueError('unknown_confidence must be in [0, 1]')


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be positive and finite')


class ActionHead(nn.Module):
    """The single shared per-frame semantic readout, H -> K logits."""
    def __init__(self, config):
        super().__init__()
        self.linear = nn.Linear(config.feature_dim, config.num_classes)

    def forward(self, h):
        return self.linear(h)


class EvidenceNet(nn.Module):
    """Matched-capacity local/state heads returning unscaled BCE logits.

    Local ablation retains current h and delta-h but zeros the two state
    slots. Multiplying inputs preserves the architecture/parameter budget.
    """
    def __init__(self, config, use_state: bool):
        super().__init__()
        self.feature_dim = config.feature_dim
        self.use_state = bool(use_state)
        self.net = nn.Sequential(
            nn.Linear(4 * config.feature_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, 1))

    def input_logits(self, inputs):
        if inputs.shape[-1] != 4 * self.feature_dim:
            raise ValueError('evidence inputs must have width 4H')
        if not self.use_state:
            inputs = torch.cat((inputs[..., :2*self.feature_dim],
                                torch.zeros_like(inputs[..., 2*self.feature_dim:])), dim=-1)
        return self.net(inputs).squeeze(-1)

    def forward(self, h, previous_h, previous_state):
        return self.input_logits(torch.cat(
            (h, h-previous_h, previous_state, h-previous_state), dim=-1))


def _features(features, config, *, device=None, dtype=torch.float32):
    # as_tensor keeps tensor autograd history; numpy/memmap inputs remain frozen.
    h = torch.as_tensor(features, dtype=dtype, device=device)
    if h.ndim != 2 or h.shape[1] != config.feature_dim:
        raise ValueError('features must have shape [T, feature_dim]')
    if not bool(torch.isfinite(h).all()):
        raise ValueError('features must be finite')
    return h


def _boundaries(boundaries, count):
    if boundaries is None:
        return set()
    items = list(boundaries)
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or not 0 < i < count for i in items):
        raise ValueError('teacher boundaries must be internal integer frame indices')
    if len(set(items)) != len(items):
        raise ValueError('teacher boundaries must be unique')
    return set(items)


def training_feature_inputs(features, config, boundaries=None, fire=None):
    """Build differentiable [T,4H] inputs with explicitly chosen reset history.

    ``boundaries`` is training-only oracle initialization. ``fire`` supplies
    a detached autonomous rollout's bool[T] decisions for closed-loop training.
    Both reset *after* the corresponding frame's evidence inputs are read.
    Neither path changes adjacent-frame delta-h or uses future features.
    """
    h = _features(features, config, device=features.device if torch.is_tensor(features) else None)
    reset = _boundaries(boundaries, len(h))
    if boundaries is not None and fire is not None:
        raise ValueError('choose teacher boundaries or autonomous fires')
    if fire is not None:
        decisions = torch.as_tensor(fire)
        if decisions.shape != (len(h),) or decisions.dtype != torch.bool:
            raise ValueError('fire must be bool[T]')
        if len(h) and bool(decisions[0]):
            raise ValueError('initialization cannot be a fire')
        reset = set(torch.nonzero(decisions, as_tuple=False).flatten().tolist())
    if not len(h):
        return h.new_empty((0, 4*config.feature_dim))
    state, previous, rows = h[0], h[0], []
    rho = math.exp(-config.frame_period_s / config.state_tau_s)
    for i, current in enumerate(h):
        rows.append(torch.cat((current, current-previous, state, current-state)))
        if i in reset:
            state = current
        elif i:
            state = rho*state + (1-rho)*current
        previous = current
    return torch.stack(rows)


@torch.no_grad()
def rollout(features, action_head, evidence, config, mechanism='lif', threshold=1.,
            tau=1., teacher_boundaries=None, *, dt_s=None):
    """Run a complete recording without annotations or future-frame lookahead.

    ``teacher_boundaries`` is an explicit training-only escape hatch for
    oracle initialization diagnostics. Evaluation callers must leave it None.
    dt_s optionally gives elapsed seconds since the preceding frame (first
    value unused by dynamics); the default is the native nominal frame period.
    The final frame owns a final dt_s[-1] interval for EOF timing.
    """
    if mechanism not in ('direct', 'if', 'lif'):
        raise ValueError('mechanism must be direct, if, or lif')
    _positive(threshold, 'threshold')
    _positive(tau, 'tau')
    parameter = next(action_head.parameters())
    h = _features(features, config, device=parameter.device, dtype=parameter.dtype)
    reset = _boundaries(teacher_boundaries, len(h))
    if dt_s is None:
        dt = np.full(len(h), config.frame_period_s, dtype=np.float64)
    else:
        dt = np.asarray(dt_s, dtype=np.float64)
        if dt.ndim == 0:
            dt = np.full(len(h), float(dt), dtype=np.float64)
        if dt.shape != (len(h),) or not np.isfinite(dt).all() or np.any(dt <= 0):
            raise ValueError('dt_s must be positive finite scalar or [T]')
    count = len(h)
    trace = dict(frame=np.arange(count, dtype=np.int64),
                 evidence=np.zeros(count), voltage=np.zeros(count),
                 fire=np.zeros(count, dtype=bool), state_distance=np.zeros(count),
                 probabilities=np.empty((count, config.num_classes), dtype=np.float32))
    if not count:
        return dict(events=[], trace=trace)
    # Per-frame readout deliberately avoids classifier(mean(features)).
    probabilities = action_head(h).softmax(-1)
    if probabilities.shape != (count, config.num_classes) or not torch.isfinite(probabilities).all():
        raise ValueError('action head must produce finite [T, num_classes] logits')
    trace['probabilities'] = probabilities.cpu().numpy()
    edges = np.concatenate(([0.], np.cumsum(dt[1:]), [dt[1:].sum()+dt[-1]]))
    events, start, state, previous, voltage, was_high = [], 0, h[0], h[0], 0., False

    def close(stop, closure):
        mean = probabilities[start:stop].mean(0)
        confidence, label = float(mean.max()), int(mean.argmax())
        if config.unknown_confidence is not None and confidence < config.unknown_confidence:
            label = -100
        events.append(dict(event_id=len(events), start_frame=start, stop_frame=stop,
                           start_s=float(edges[start]), stop_s=float(edges[stop]),
                           duration_s=float(edges[stop]-edges[start]),
                           confirmed_at_frame=stop, confirmed_at_s=float(edges[stop]),
                           class_id=label, confidence=confidence, probabilities=mean.cpu().tolist(),
                           closure=closure, time_basis='nominal' if dt_s is None else 'elapsed'))

    for i, current in enumerate(h):
        logit = evidence(current, previous, state)
        if logit.numel() != 1 or not bool(torch.isfinite(logit).all()):
            raise ValueError('evidence must return a finite scalar logit')
        drive = config.current_max * float(logit.sigmoid())
        trace['evidence'][i] = drive
        trace['state_distance'][i] = float(torch.linalg.vector_norm(current-state))
        high = drive >= threshold
        if i == 0:
            # Start initializes state; an initially high direct input must
            # first drop below threshold before producing an upward crossing.
            was_high = high
            continue
        if mechanism == 'lif':
            decay = math.exp(-float(dt[i])/tau)
            candidate = decay*voltage + (1-decay)*drive
        elif mechanism == 'if':
            candidate = voltage + float(dt[i])*drive  # reference tau_0=1 second
        else:
            candidate = drive
        trigger = high and not was_high if mechanism == 'direct' else candidate >= threshold
        if config.minimum_duration_s is not None:
            trigger = trigger and edges[i]-edges[start] >= config.minimum_duration_s
        trace['voltage'][i], trace['fire'][i] = candidate, trigger
        if trigger:
            close(i, 'fire')
            start, voltage = i, 0.
        else:
            voltage = candidate
        # Oracle diagnostics must reproduce the warm-start state history,
        # independently of predicted fires; normal inference uses fires only.
        if i in reset or (teacher_boundaries is None and trigger):
            state = current
        else:
            rho = math.exp(-float(dt[i])/config.state_tau_s)
            state = rho*state + (1-rho)*current
        previous, was_high = current, high
    close(count, 'eof')
    return dict(events=events, trace=trace)
