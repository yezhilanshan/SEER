"""Shared causal evidence state, event heads and explicit training surrogates."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
from .cif import cif_integrate
from .config import GateConfig
from .lif import LIFController, LIFState


@dataclass
class GateState:
    reference: torch.Tensor
    recent: torch.Tensor
    content_sum: torch.Tensor
    content_count: torch.Tensor
    dynamics: LIFState | None = None
    cif_mass: float = 0.
    recent_variance: torch.Tensor | None = None


class EventGate(nn.Module):
    def __init__(self, config: GateConfig, mode='cif'):
        super().__init__()
        if mode not in ('cif', 'lif', 'frame-baseline'):
            raise ValueError('mode must be cif, lif or frame-baseline')
        self.config, self.mode = config, mode
        h = config.feature_dim
        self.driver = nn.Sequential(nn.Linear(4*h, config.hidden_dim), nn.GELU(), nn.Linear(config.hidden_dim, 1))
        nn.init.constant_(self.driver[-1].bias, -2.)
        self.metric = nn.Linear(h, h, bias=False)
        nn.init.eye_(self.metric.weight)
        fraction = min(max(config.radius / 2, 1e-6), 1-1e-6)
        self.radius_logit = nn.Parameter(torch.tensor(math.log(fraction/(1-fraction))))
        self.classifier = nn.Sequential(nn.Linear(h, config.hidden_dim), nn.GELU(),
                                        nn.Linear(config.hidden_dim, config.num_classes))
        self.controller = LIFController(config.tau, config.theta, config.beta, config.tau_r,
                                        config.temperature, config.reset)

    def validate_features(self, features, length=None):
        n = len(features) if length is None else int(length)
        if (features.ndim != 2 or features.shape[1] != self.config.feature_dim or not 0 < n <= len(features)
                or not torch.isfinite(features[:n]).all()):
            raise ValueError('expected nonempty finite valid features [T,H]')
        return features[:n].float()

    def _initial(self, h):
        return GateState(h, h, h.new_zeros(h.shape), h.new_zeros(()), recent_variance=torch.zeros_like(h))

    def _evidence(self, h, state, dt):
        rate = -math.expm1(-dt/self.config.recent_tau_s)
        variance = (1-rate)*(state.recent_variance+rate*(h-state.recent).square())
        recent = state.recent + rate*(h-state.recent)
        distance = (self.metric(recent-state.reference).square().mean()+1e-8).sqrt()-1e-4
        radius = 2*torch.sigmoid(self.radius_logit)
        score = self.driver(torch.cat((h, state.reference, h-state.reference, (variance+1e-8).sqrt()))).squeeze(-1)
        drive = F.relu(distance-radius)*F.softplus(score)
        low = state.dynamics is None or float(state.dynamics.voltage.detach()) < .2*self.config.theta
        reference = state.reference
        if low:
            reference = reference + (-math.expm1(-dt/self.config.reference_tau_s))*(recent-reference)
        return score, drive, reference, recent, variance

    def cif_step(self, h, state=None):
        state = self._initial(h) if state is None else state
        score, _, reference, recent, variance = self._evidence(h, state, self.config.frame_period_s)
        alpha = torch.sigmoid(score)
        mass = state.cif_mass + float(alpha.detach())
        if mass >= 1:
            reference, recent, mass = h, h, mass-1
            variance = torch.zeros_like(variance)
        next_state = GateState(reference, recent, state.content_sum, state.content_count,
                               cif_mass=mass,recent_variance=variance)
        return alpha, next_state

    def lif_step(self, h, state=None, dt=None):
        first = state is None
        state = self._initial(h) if first else state
        dt = self.config.frame_period_s if dt is None else float(dt)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('dt must be finite and positive')
        _, drive, reference, recent, variance = self._evidence(h, state, dt)
        step = self.controller.step(drive, dt, state.dynamics)
        old_mean = h if first else state.content_sum/state.content_count.clamp_min(1)
        logits = self.classifier(old_mean)
        # A boundary before any observed content cannot close a nonempty event.
        spike = step.spike if not first else h.new_zeros(())
        hard = step.hard_spike if not first else h.new_zeros(())
        soft = step.soft_spike if not first else h.new_zeros(())
        total = (1-spike)*(state.content_sum+h) + spike*h
        count = (1-spike)*(state.content_count+1) + spike
        reference = (1-spike)*reference + spike*h
        recent = (1-spike)*recent + spike*h
        next_state = GateState(reference, recent, total, count, step.state, recent_variance=(1-spike)*variance)
        score = (step.pre_voltage-step.threshold)/self.controller.temperature
        return next_state, dict(logits=logits, hard=hard, soft=soft, score=score,
                                pre=step.pre_voltage, threshold=step.threshold, drive=drive)

    def drive_trace(self, features):
        features = self.validate_features(features)
        state, values = None, []
        for h in features:
            if self.mode == 'cif':
                value, state = self.cif_step(h, state)
            else:
                state, info = self.lif_step(h, state)
                value = info['drive']
            values.append(value)
        return torch.stack(values)

    def training_loss(self, features, targets, *, length=None, teacher=None, distill_weight=0.):
        features = self.validate_features(features, length)
        target = torch.as_tensor(targets, dtype=torch.long, device=features.device)
        if (target.ndim != 1 or not len(target) or (target < 0).any()
                or (target >= self.config.num_classes).any()):
            raise ValueError('nonempty valid ordered class targets required')
        if self.mode == 'frame-baseline':
            raise ValueError('frame baseline uses explicitly masked frame labels')
        if self.mode == 'cif':
            alpha = self.drive_trace(features)
            output = cif_integrate(features, alpha, len(target))
            quantity = (output.raw_mass-len(target)).abs()
            classification = F.cross_entropy(self.classifier(output.tokens), target)
            return dict(loss=quantity+classification, quantity=quantity, classification=classification,
                        distillation=quantity.new_zeros(()), hard_count=len(output.tokens), soft_count=alpha.sum())
        repeats = int((target[1:] == target[:-1]).sum())
        if len(target)+repeats > len(features):
            raise ValueError('CTC has too few valid time steps for target and adjacent repetitions')
        state, probabilities, soft, hard, scores = None, [], [], [], []
        for index, h in enumerate(features):
            state, info = self.lif_step(h, state)
            soft.append(info['soft']); hard.append(info['hard']); scores.append(info['score'])
            if index:
                # log-space sigmoid avoids clamping away the gradient when hard firing is absent.
                class_log = F.log_softmax(info['logits'], -1) + F.logsigmoid(info['score'])
                probabilities.append(torch.cat((F.logsigmoid(-info['score']).view(1), class_log)))
        final_logits = self.classifier(state.content_sum/state.content_count.clamp_min(1))
        probabilities.append(torch.cat((features.new_full((1,), -torch.inf), F.log_softmax(final_logits, -1))))
        log_prob = torch.stack(probabilities).unsqueeze(1)
        classification = F.ctc_loss(log_prob, (target+1), torch.tensor([len(probabilities)]),
                                     torch.tensor([len(target)]), blank=0, reduction='sum', zero_infinity=False)/len(target)
        soft_count = torch.stack(soft).sum()
        quantity = (soft_count-(len(target)-1)).abs()
        distillation = quantity.new_zeros(())
        if teacher is not None:
            teacher = torch.as_tensor(teacher, device=features.device, dtype=features.dtype)
            if teacher.shape != (len(features),) or not torch.isfinite(teacher).all() or ((teacher<0)|(teacher>1)).any():
                raise ValueError('invalid teacher probabilities')
            if len(features)>1:
                distillation = F.binary_cross_entropy_with_logits(torch.stack(scores)[1:], teacher[1:])
        loss = classification + quantity + float(distill_weight)*distillation
        if not torch.isfinite(loss):
            raise FloatingPointError('nonfinite event-gate loss; refusing to hide invalid alignments')
        return dict(loss=loss, quantity=quantity, classification=classification, distillation=distillation,
                    hard_count=int(torch.stack(hard).sum().detach()), soft_count=soft_count)
