"""Causal inputs, partial-label targets, and online event boundary evaluation."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def continuous_inputs(hidden):
    hidden = np.asarray(hidden, dtype=np.float32)
    if hidden.ndim != 2 or hidden.shape[1] != 128 or not len(hidden) or not np.isfinite(hidden).all():
        raise ValueError('expected finite continuous hidden [T,128]')
    cutoffs = np.arange(2, len(hidden) + 1, 2, dtype=np.int64)
    if len(hidden) % 2:
        cutoffs = np.append(cutoffs, len(hidden))
    current = hidden[cutoffs - 1]
    delta = np.concatenate((np.zeros_like(current[:1]), current[1:] - current[:-1]))
    return np.concatenate((current, delta), axis=1), cutoffs


def boundary_targets(frames, instances):
    if frames < 1:
        raise ValueError('empty recording')
    cutoffs = np.arange(2, frames + 1, 2, dtype=np.int64)
    if frames % 2:
        cutoffs = np.append(cutoffs, frames)
    known = np.zeros(frames, dtype=bool)
    starts, ends = [], []
    previous = 0
    for row in sorted(instances, key=lambda r: r['start']):
        s, e = row['start'], row['stop']
        if not previous <= s < e <= frames:
            raise ValueError('invalid or overlapping instances')
        known[s:e] = True; starts.append(s); ends.append(e); previous = e
    target = np.zeros((len(cutoffs), 2), dtype=np.float32)
    begin = np.concatenate(([0], cutoffs[:-1]))
    all_known = np.array([known[s:e].all() for s, e in zip(begin, cutoffs)])
    mask = np.repeat(all_known[:, None], 2, axis=1)
    truth = {'start': np.array(starts, dtype=np.int64), 'end': np.array(ends, dtype=np.int64)}
    for head, points in enumerate((truth['start'] + 1, truth['end'])):
        positions = np.searchsorted(cutoffs, points)
        if len(np.unique(positions)) != len(positions):
            raise ValueError('boundary resolution: multiple same-head boundaries within one Update')
        for index in positions:
            mask[max(0, index - 1):index + 2, head] = False
        target[positions, head] = 1
        mask[positions, head] = True
    return {'cutoffs': cutoffs, 'targets': target, 'mask': mask, 'known_frames': known, 'truth': truth}


class BoundaryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(256, 128), nn.GELU(), nn.Dropout(.1), nn.Linear(128, 2))

    def forward(self, x):
        return self.network(x)


def masked_boundary_loss(logits, targets, mask, pos_weight):
    values = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight, reduction='none')
    weights = mask.to(values.dtype)
    return ((values * weights).sum(0) / weights.sum(0).clamp_min(1)).mean()


def decode_boundaries(probability, cutoffs, threshold, minimum_gap=4):
    probability, cutoffs = np.asarray(probability), np.asarray(cutoffs)
    if probability.shape != cutoffs.shape or not np.isfinite(probability).all():
        raise ValueError('invalid online score sequence')
    above = probability >= threshold
    rises = np.flatnonzero(above & np.concatenate(([True], ~above[:-1])))
    emitted = []
    for index in rises:
        now = int(cutoffs[index])
        if not emitted or now - emitted[-1] >= minimum_gap:
            emitted.append(now)
    return np.array(emitted, dtype=np.int64)


def _summary(tp, fp, fn, unknown, lags, unknown_frames):
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    lags = np.array(lags, dtype=float)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'unscored_unknown': unknown,
            'precision': precision, 'recall': recall,
            'f1': 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
            'mean_signed_delay_s': float(lags.mean() / 15) if len(lags) else None,
            'median_abs_delay_s': float(np.median(np.abs(lags)) / 15) if len(lags) else None,
            'p90_abs_delay_s': float(np.quantile(np.abs(lags), .9) / 15) if len(lags) else None,
            'unmatched_unknown_per_minute': unknown / (unknown_frames / 15 / 60) if unknown_frames else None,
            'unknown_frames': unknown_frames, 'lags_frames': lags.astype(int).tolist()}


def score_events(predicted, truth, known_frames, tolerance):
    predicted = np.asarray(predicted, dtype=np.int64)
    truth = np.asarray(truth, dtype=np.int64)
    known_frames = np.asarray(known_frames, dtype=bool)
    if (np.any(np.diff(predicted) <= 0) or np.any(np.diff(truth) < 0)
            or np.any(predicted < 1) or np.any(predicted > len(known_frames))):
        raise ValueError('invalid event timestamps')
    i = j = 0; pairs = []; false = []; missed = []
    while i < len(predicted) and j < len(truth):
        p, t = int(predicted[i]), int(truth[j])
        if p < t - tolerance:
            false.append(p); i += 1
        elif p > t + tolerance:
            missed.append(t); j += 1
        else:
            pairs.append({'predicted': p, 'truth': t, 'lag_frames': p - t}); i += 1; j += 1
    false.extend(predicted[i:].tolist()); missed.extend(truth[j:].tolist())
    fp = [p for p in false if known_frames[p - 1]]
    unknown = [p for p in false if not known_frames[p - 1]]
    result = _summary(len(pairs), len(fp), len(missed), len(unknown),
                      [r['lag_frames'] for r in pairs], int((~known_frames).sum()))
    return {**result, 'pairs': pairs, 'unmatched_known': fp, 'unmatched_unknown': unknown, 'missed_truth': missed}


def combine_scores(rows):
    return _summary(sum(r['tp'] for r in rows), sum(r['fp'] for r in rows), sum(r['fn'] for r in rows),
                    sum(r['unscored_unknown'] for r in rows), [lag for r in rows for lag in r['lags_frames']],
                    sum(r['unknown_frames'] for r in rows))


def independent_event_counts(predicted, truth, known_frames, tolerance):
    """Dynamic-programming match cardinality, independent of the two-pointer scorer."""
    p, t = list(predicted), list(truth)
    dp = np.zeros((len(p) + 1, len(t) + 1), dtype=np.int32)
    for i in range(len(p) - 1, -1, -1):
        for j in range(len(t) - 1, -1, -1):
            match = 1 + dp[i + 1, j + 1] if abs(p[i] - t[j]) <= tolerance else 0
            dp[i, j] = max(dp[i + 1, j], dp[i, j + 1], match)
    i = j = 0; matched = set()
    while i < len(p) and j < len(t):
        if abs(p[i] - t[j]) <= tolerance and 1 + dp[i + 1, j + 1] == dp[i, j]:
            matched.add(i); i += 1; j += 1
        elif p[i] < t[j] - tolerance:
            assert dp[i + 1, j] == dp[i, j]
            i += 1
        else:
            assert p[i] > t[j] + tolerance and dp[i, j + 1] == dp[i, j]
            j += 1
    assert len(matched) == dp[0, 0]
    fp = sum(bool(known_frames[value - 1]) for i, value in enumerate(p) if i not in matched)
    return len(matched), fp, len(t) - len(matched), len(p) - len(matched) - fp
