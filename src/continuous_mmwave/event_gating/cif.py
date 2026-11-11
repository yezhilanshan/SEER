"""Differentiable CIF mass splitting; target scaling is training-only."""
from dataclasses import dataclass
import torch


@dataclass
class CIFOutput:
    tokens: torch.Tensor
    end_frames: list[int]
    raw_mass: torch.Tensor
    residual_mass: torch.Tensor
    residual_sum: torch.Tensor


def cif_integrate(features, alpha, target_count=None):
    if (features.ndim != 2 or not len(features) or alpha.shape != (len(features),)
            or not torch.isfinite(features).all() or not torch.isfinite(alpha).all()
            or (alpha < 0).any()):
        raise ValueError('finite features [T,H] and nonnegative alpha [T] required')
    raw = alpha.sum()
    # Accumulate in FP64: hundreds of FP32 additions can miss the final scaled token.
    # Keep raw_mass in the original precision for the unscaled quantity objective.
    alpha = alpha.double()
    if target_count is not None:
        if type(target_count) is not int or target_count < 1 or raw.detach().item() <= 0:
            raise ValueError('positive target_count and positive mass required')
        alpha = alpha * (target_count / alpha.sum())
    mass = alpha.new_zeros(())
    total = features.new_zeros(features.shape[1],dtype=torch.float64)
    tokens, ends = [], []
    for index, (h, weight) in enumerate(zip(features, alpha)):
        remaining = weight
        while remaining.detach().item() > 0:
            available = 1 - mass
            take = torch.minimum(remaining, available)
            total = total + take * h
            mass = mass + take
            remaining = remaining - take
            if mass.detach().item() >= 1 - 1e-10:
                tokens.append((total / mass).to(features.dtype))
                ends.append(index + 1)
                mass = alpha.new_zeros(())
                total = features.new_zeros(features.shape[1],dtype=torch.float64)
    if target_count is not None and len(tokens) != target_count:
        raise RuntimeError('CIF scaled mass did not produce the requested number of tokens')
    stacked = torch.stack(tokens) if tokens else features.new_empty((0, features.shape[1]))
    return CIFOutput(stacked, ends, raw, mass, total)
