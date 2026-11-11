"""Fixed motion targets and observable command-episode alignment."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


def build_command_episodes(row, frame_timestamps):
    if row.get('recording_start_rule') != 'first_command_as_legacy_converter':
        raise ValueError('unverified recording origin')
    with Path(row['source_commands']).open(encoding='utf-8-sig', newline='') as handle:
        commands = list(csv.DictReader(handle))
    times = [datetime.fromisoformat(r['timestamp'].strip()) for r in commands]
    if not times or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError('command clock must increase')
    starts = [(t - times[0]).total_seconds() for t in times]
    end = float(frame_timestamps[-1]) + .09
    return [dict(episode_id=i, start=start, end=starts[i+1] if i+1 < len(starts) else end,
                 command=' '.join(commands[i]['command'].split()), supervision_source='command_boundary_proxy')
            for i, start in enumerate(starts) if start < end]


def update_grid(timestamps, command_starts):
    stamps = np.asarray(timestamps, dtype=np.float64)
    if stamps.ndim != 1 or len(stamps) < 2 or not np.isfinite(stamps).all() or (np.diff(stamps) <= 0).any():
        raise ValueError('invalid recording timestamps')
    frames = len(stamps)//2*2
    pairs = stamps[:frames].reshape(-1,2)
    intervals = np.stack((pairs[:,0], pairs[:,1]+.09),1)
    starts = np.asarray(command_starts, dtype=np.float64)
    if not len(starts) or (np.diff(starts) <= 0).any():
        raise ValueError('command starts must increase')
    episode = np.searchsorted(starts, pairs[:,0]+1e-8, side='right')-1
    last_episode = np.searchsorted(starts, intervals[:,1]-1e-8, side='right')-1
    valid = (episode >= 0) & (episode == last_episode) & np.isclose(pairs[:,1]-pairs[:,0], .09, atol=1e-7)
    return dict(intervals=intervals, episode=episode, valid=valid, trailing_frames=len(stamps)-frames)


def future_indices(update_intervals, valid, horizons):
    intervals, valid = np.asarray(update_intervals), np.asarray(valid, dtype=bool)
    if intervals.ndim != 2 or intervals.shape != (len(valid),2) or not np.isfinite(intervals).all():
        raise ValueError('invalid recording intervals')
    if (np.diff(intervals[:,0]) <= 0).any() or (intervals[:,1] <= intervals[:,0]).any():
        raise ValueError('one increasing recording required')
    contiguous = np.isclose(intervals[:-1,1], intervals[1:,0], atol=1e-7, rtol=0)
    result = {}
    for h in horizons:
        if not isinstance(h,int) or h <= 0:
            raise ValueError('positive integer horizon required')
        indices = np.full(len(valid),-1,dtype=np.int64)
        for t in range(len(valid)-h):
            if valid[t:t+h+1].all() and contiguous[t:t+h].all():
                indices[t] = t+h
        result[h] = indices
    return result


def prediction_contract(update_intervals, observation_valid, command_starts, horizons=(1,3,5)):
    """P1 physical targets and offline proxy metadata; never use proxy masks for futures.

    ``proxy_boundary_cross`` counts boundaries in (available_at[t], available_at[t+h]].
    Mixed endpoints are reported independently and get no single episode ID.
    This function operates on exactly one increasing recording.
    """
    intervals = np.asarray(update_intervals,dtype=np.float64)
    observed = np.asarray(observation_valid,dtype=bool)
    starts = np.asarray(command_starts,dtype=np.float64)
    if intervals.ndim != 2 or intervals.shape != (len(observed),2):
        raise ValueError('one recording with matching observation mask required')
    if not len(starts) or starts.ndim != 1 or not np.isfinite(starts).all() or (np.diff(starts) <= 0).any():
        raise ValueError('increasing command starts required')
    observed = observed & np.isclose(intervals[:,1]-intervals[:,0],.18,rtol=0,atol=1e-7)
    mapping = future_indices(intervals,observed,horizons)
    first = np.searchsorted(starts,intervals[:,0]+1e-8,side='right')-1
    last = np.searchsorted(starts,intervals[:,1]-1e-8,side='right')-1
    assigned = (first >= 0) & (first == last)
    mixed = (first != last)
    episode = np.where(assigned,first,-1)
    boundaries = starts[1:]
    reached = np.searchsorted(boundaries,intervals[:,1]+1e-8,side='right')
    cross, mixed_current, mixed_future = {}, {}, {}
    for h,future in mapping.items():
        good = future >= 0
        cross[h] = np.zeros(len(observed),dtype=bool)
        mixed_current[h] = np.zeros(len(observed),dtype=bool)
        mixed_future[h] = np.zeros(len(observed),dtype=bool)
        cross[h][good] = reached[future[good]] > reached[good]
        mixed_current[h][good] = mixed[good]
        mixed_future[h][good] = mixed[future[good]]
    common = np.flatnonzero(np.logical_and.reduce([future >= 0 for future in mapping.values()]))
    return dict(observation_valid=observed,event_assignment_valid=assigned,proxy_episode_id=episode,
                mixed_update=mixed,future=mapping,proxy_boundary_cross=cross,
                mixed_current=mixed_current,mixed_future=mixed_future,common_start=common)


def fixed_motion_target(tokens, mask):
    mask = mask.bool()
    if mask.shape != tokens.shape[:-1] or not mask.any(-1).all():
        raise ValueError('motion target needs valid tokens')
    return ((tokens.detach().float()*mask[...,None]).sum(-2)/mask.sum(-1,keepdim=True)).detach()


def fit_normalization(records):
    if not records or any(r['role'] != 'train' for r in records):
        raise ValueError('normalization is fit-only')
    # Equal recording mass rather than length-weighted statistics.
    values = [r['motion'][r['valid'].bool()].double() for r in records]
    if any(len(v) == 0 or not torch.isfinite(v).all() for v in values):
        raise ValueError('normalization needs finite fit observations')
    mean = torch.stack([v.mean(0) for v in values]).mean(0)
    second = torch.stack([(v*v).mean(0) for v in values]).mean(0)
    return dict(mean=mean.float(), std=(second-mean.square()).clamp_min(0).sqrt().clamp_min(1e-4).float())
