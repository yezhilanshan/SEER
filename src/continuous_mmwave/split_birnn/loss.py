"""Equation (8-10) literal global Dice and exact off-target smoothing."""
from __future__ import annotations
import torch
from torch.nn import functional as F


def combo_loss(logits: torch.Tensor, labels: torch.Tensor, *, dice_mode: str = 'global', epsilon: float = .2):
    if dice_mode not in ('global', 'per_class', 'none') or not 0 <= epsilon < 1:
        raise ValueError('unknown dice_mode or smoothing epsilon')
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != 9:
        raise ValueError('nine-class logits and matching labels required')
    mask = labels != -100
    if not mask.any():
        zero = logits.sum() * 0
        return zero, {'cce': zero, 'dice': zero}
    x, y = logits[mask], labels[mask]
    logp = F.log_softmax(x, dim=-1)
    target = F.one_hot(y, 9).to(x.dtype)
    smooth = target*(1-epsilon) + (1-target)*(epsilon/8)
    cce = -(smooth*logp).sum(-1).mean()
    p = logp.exp()
    if dice_mode == 'global':
        dice = 1-(2*(target*p).sum()+1)/(target.sum()+p.sum()+1)
    elif dice_mode == 'per_class':
        dice = (1-(2*(target*p).sum(0)+1)/(target.sum(0)+p.sum(0)+1)).mean()
    else:
        dice = cce*0
    return cce+dice, {'cce': cce, 'dice': dice}
