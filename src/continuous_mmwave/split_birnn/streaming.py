"""Causal replay of Algorithm 2 with explicit correction availability and delay."""
from __future__ import annotations
import numpy as np
import torch


def correction_windows(frames: int, window: int, hop: int) -> list[tuple[int, int]]:
    if frames < 0 or window < 1 or not 1 <= hop <= window:
        raise ValueError('require nonnegative frames and 1 <= hop <= window')
    return [(end-window, end) for end in range(window, frames+1, hop)]


def _views(forward: torch.Tensor):
    frames = forward.shape[1]
    return {'forward_logits': forward, 'first_logits': forward.clone(), 'final_logits': forward.clone(),
            'correction_counts': np.zeros(frames, dtype=np.int64),
            'first_end_frame': np.full(frames, -1, dtype=np.int64),
            'final_end_frame': np.full(frames, -1, dtype=np.int64), 'events': []}


def _update(result: dict, logits: torch.Tensor, start: int, end: int):
    unseen = result['correction_counts'][start:end] == 0
    indices = np.flatnonzero(unseen) + start
    result['first_logits'][:, indices] = logits[:, np.flatnonzero(unseen)]
    result['first_end_frame'][indices] = end-1
    result['final_logits'][:, start:end] = logits
    result['final_end_frame'][start:end] = end-1
    result['correction_counts'][start:end] += 1
    result['events'].append({'start_frame': start, 'end_frame_exclusive': end, 'emission_frame': end-1})


def _delays(result: dict, period: float):
    if not np.isfinite(period) or period <= 0:
        raise ValueError('positive finite frame period required')
    frames = np.arange(len(result['correction_counts']))
    for kind in ('first', 'final'):
        endpoints = result[kind+'_end_frame']
        result[kind+'_delay_s'] = np.where(endpoints >= 0, (endpoints-frames)*period, np.nan)
    result['available_mask'] = result['correction_counts'] > 0
    result['delay_clock'] = 'approximate_frame_arrival_clock_not_physical_action_onset'
    return result


@torch.no_grad()
def replay_split(model, md: torch.Tensor, *, window: int = 512, hop: int = 171,
                 chunk_frames: int = 128, frame_period_s: float = .09):
    if model.training or md.ndim != 3 or md.shape[0] != 1 or md.shape[1] == 0 or chunk_frames < 1:
        raise ValueError('eval model and a single nonempty recording required')
    schedule = correction_windows(md.shape[1], window, hop)
    # This offline replay batches a causal forward computation. Each correction
    # accesses only cache[start:end], so no future context leaks into an emission.
    state, ys, caches = None, [], []
    for start in range(0, md.shape[1], chunk_frames):
        y, cache, state = model.forward_chunk(md[:, start:start+chunk_frames], state)
        ys.append(y); caches.append(cache)
    result = _views(torch.cat(ys, 1))
    cache = torch.cat(caches, 1)
    for start, end in schedule:
        _update(result, model.correct(cache[:,start:end]), start, end)
    return _delays(result, frame_period_s)


@torch.no_grad()
def replay_baseline(model, md: torch.Tensor, *, window: int = 512, hop: int = 171, frame_period_s: float = .09):
    if model.training or md.ndim != 3 or md.shape[0] != 1 or md.shape[1] == 0:
        raise ValueError('eval model and a single nonempty recording required')
    result = _views(torch.full((1, md.shape[1], 9), float('nan'), device=md.device, dtype=md.dtype))
    for start, end in correction_windows(md.shape[1], window, hop):
        _update(result, model(md[:,start:end]), start, end)
    # There is no trained causal head in the ordinary BiGRU. Missing tail outputs
    # remain explicit; metric denominators must not silently remove these frames.
    return _delays(result, frame_period_s)


def classification_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    if logits.shape[:-1] != labels.shape or logits.shape[-1] != 9:
        raise ValueError('nine-class logits and matching labels required')
    y = labels.detach().cpu().numpy().reshape(-1)
    x = logits.detach().cpu().numpy().reshape(-1,9)
    valid = y != -100
    y, x = y[valid], x[valid]
    if np.any((y < 0) | (y >= 9)):
        raise ValueError('invalid class IDs')
    available = np.isfinite(x).all(1)
    pred = np.where(available, np.nan_to_num(x).argmax(1), 9)
    # Last column is abstention/unavailable; its truth rows still count in recall.
    confusion = np.zeros((9,10), dtype=np.int64)
    np.add.at(confusion, (y,pred), 1)
    support = confusion.sum(1); tp = np.diag(confusion[:,:9]); predicted = confusion[:,:9].sum(0)
    precision = np.divide(tp,predicted,out=np.zeros(9),where=predicted>0)
    recall = np.divide(tp,support,out=np.zeros(9),where=support>0)
    f1 = np.divide(2*precision*recall,precision+recall,out=np.zeros(9),where=(precision+recall)>0)
    n = len(y)
    return {'valid_frames': n, 'unavailable_frames': int((~available).sum()),
            'accuracy': float(tp.sum()/n) if n else None,
            'weighted_f1': float((f1*support).sum()/n) if n else None,
            'macro_f1_all_9': float(f1.mean()) if n else None,
            'class_precision': precision.tolist(), 'class_recall': recall.tolist(),
            'class_f1': f1.tolist(), 'class_support': support.tolist(),
            'confusion_matrix': confusion.tolist(), 'unavailable_column': 9}
