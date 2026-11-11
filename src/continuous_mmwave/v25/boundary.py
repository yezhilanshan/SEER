"""Post-confirm localization on a feature grid, without GT online inputs."""
from __future__ import annotations

import math
from dataclasses import dataclass
from torch.nn import functional as F


@dataclass(frozen=True)
class BoundaryResult:
    timestamp: float
    sequence: int
    score: float
    scores: tuple[dict, ...]


def localize_boundary(frames, *, proposal_time, confirm_time, active_start, delta_pre_s=.5, weights=(1.,1.,1.)):
    if (not all(math.isfinite(x) for x in (proposal_time,confirm_time,active_start,delta_pre_s,*weights))
        or confirm_time < proposal_time or delta_pre_s < 0 or len(weights) != 3
        or min(weights) < 0 or sum(weights) <= 0):
        raise ValueError('invalid boundary localization configuration')
    rows = []
    previous = None
    for frame in frames:
        if previous is None:
            semantic = 0.
        else:
            semantic = float((1-F.cosine_similarity(frame.semantic_key,previous.semantic_key,-1))[0])/2
        previous = frame
        if not (max(active_start,proposal_time-delta_pre_s) < frame.timestamp+1e-9
                and frame.timestamp > active_start+1e-9 and frame.timestamp <= confirm_time+1e-9):
            continue
        # Probabilities already have a fixed [0,1] scale; cosine distance / 2 does too.
        score = weights[0]*frame.local_probability+weights[1]*frame.radar_probability+weights[2]*max(0.,min(1.,semantic))
        rows.append({'timestamp':frame.timestamp,'sequence':frame.sequence,'score':score,
                     'local':frame.local_probability,'radar':frame.radar_probability,'semantic':semantic})
    if not rows:
        raise RuntimeError('no valid nonempty boundary grid in retained features')
    # Stable deterministic tie breaking: earliest maximum.
    best = max(rows,key=lambda r:r['score'])
    return BoundaryResult(best['timestamp'],best['sequence'],best['score'],tuple(rows))
