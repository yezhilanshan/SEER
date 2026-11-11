"""Serializable, validated configuration for the independent event gate."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class GateConfig:
    feature_dim: int = 128
    hidden_dim: int = 128
    num_classes: int = 12
    frame_period_s: float = 1 / 15
    tau: float = 1.5
    tau_r: float = 3.0
    beta: float = 1.0
    theta: float = 1.0
    temperature: float = .5
    radius: float = .1
    recent_tau_s: float = 3.0
    reference_tau_s: float = 20.0
    buffer_frames: int = 192
    retrospective: bool = False
    reset: str = 'soft'
    tail_threshold: float = .5

    def __post_init__(self):
        for name in ('feature_dim', 'hidden_dim', 'num_classes', 'buffer_frames'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('frame_period_s', 'tau', 'tau_r', 'theta', 'temperature',
                     'recent_tau_s', 'reference_tau_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not math.isfinite(self.beta) or self.beta < 0 or not 0 <= self.radius <= 2:
            raise ValueError('invalid beta/radius')
        if self.reset not in ('soft', 'hard') or not 0 <= self.tail_threshold <= 1:
            raise ValueError('invalid reset/tail threshold')

    def to_dict(self):
        return asdict(self)
