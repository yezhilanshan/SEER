"""Exact zero-order-hold leaky integration with hard forward firing."""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class LIFState:
    voltage: torch.Tensor
    adaptation: torch.Tensor


@dataclass
class LIFStep:
    state: LIFState
    pre_voltage: torch.Tensor
    threshold: torch.Tensor
    hard_spike: torch.Tensor
    soft_spike: torch.Tensor
    spike: torch.Tensor


def _inverse_softplus(x):
    return math.log(math.expm1(x)) if x < 20 else x


class LIFController(nn.Module):
    def __init__(self, tau=1.5, theta=1., beta=1., tau_r=3., temperature=.5, reset='soft'):
        super().__init__()
        if (not all(math.isfinite(v) and v > 0 for v in (tau, theta, tau_r, temperature))
                or not math.isfinite(beta) or beta < 0 or reset not in ('soft', 'hard')):
            raise ValueError('invalid LIF parameters')
        self.log_tau = nn.Parameter(torch.tensor(_inverse_softplus(tau)))
        self.log_tau_r = nn.Parameter(torch.tensor(_inverse_softplus(tau_r)))
        # An exact zero is retained for the no-adaptation ablation.
        self.log_beta = nn.Parameter(torch.tensor(_inverse_softplus(max(beta, 1e-6))))
        self.adaptive = beta > 0
        self.theta, self.temperature, self.reset = theta, temperature, reset

    @property
    def tau(self):
        return F.softplus(self.log_tau).clamp_min(1e-4)

    def step(self, drive, dt, state=None):
        if not math.isfinite(float(dt)) or float(dt) <= 0:
            raise ValueError('dt must be finite and positive')
        if not torch.isfinite(drive).all() or (drive < 0).any():
            raise ValueError('drive must be finite and nonnegative')
        if state is None:
            state = LIFState(torch.zeros_like(drive), torch.zeros_like(drive))
        if state.voltage.shape != drive.shape or state.adaptation.shape != drive.shape:
            raise ValueError('LIF state shape mismatch')
        decay = torch.exp(-float(dt) / self.tau)
        adaptation = state.adaptation * torch.exp(-float(dt) / F.softplus(self.log_tau_r).clamp_min(1e-4))
        beta = F.softplus(self.log_beta) if self.adaptive else drive.new_zeros(())
        threshold = self.theta + beta * adaptation
        pre = decay * state.voltage + self.tau * (1 - decay) * drive
        hard = (pre >= threshold).to(drive.dtype)
        soft = torch.sigmoid((pre - threshold) / self.temperature)
        spike = hard + soft - soft.detach() if self.training else hard
        voltage = pre - threshold * spike if self.reset == 'soft' else pre * (1-spike)
        return LIFStep(LIFState(voltage, adaptation + spike), pre, threshold, hard, soft, spike)
