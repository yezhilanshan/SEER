"""Physical targets and current proxy labels have independent validity contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
import random

import numpy as np
import torch
from torch import Tensor

from ..v26.source import require_sha
from ..v26.losses import fit_persistence_variance

HORIZONS = (1, 3, 5)


def with_previous_observation(physical: dict,intervals) -> dict:
    """Keep P1-C's common population, which also supports the trend baseline."""
    intervals=np.asarray(intervals)
    observed=np.asarray(physical['observation_valid'],dtype=bool)
    previous=np.zeros(len(observed),dtype=bool)
    previous[1:]=observed[1:] & observed[:-1] & np.isclose(intervals[:-1,1],intervals[1:,0],atol=1e-7,rtol=0)
    result=dict(physical)
    common=np.asarray(physical['common_start'])
    result['common_start']=common[previous[common]].copy()
    return result


@dataclass
class JointRecord:
    uid: str
    role: str
    subject: str
    radar: Tensor
    motion_ref: Tensor
    intervals: Tensor
    observation_valid: Tensor
    semantic_target: Tensor
    common_start: Tensor
    source_lock: dict = field(default_factory=dict)
    groups: dict = field(default_factory=dict)


def verify_source_lock(lock: dict) -> None:
    for path, digest in lock.items():
        require_sha(path, digest)


def validate_class_maps(old: dict, new: dict) -> None:
    def ordered(value):
        rows = sorted(value['classes'], key=lambda row: row['id'])
        if [row['id'] for row in rows] != list(range(14)):
            raise ValueError('class mapping must contain fourteen contiguous IDs')
        return [(row['id'], row['command']) for row in rows]
    if ordered(old) != ordered(new):
        raise ValueError('class ID mapping differs from fitted semantic reader')


def build_joint_record(cache: dict, labels: Tensor, physical: dict, role: str) -> JointRecord:
    if role not in ('train', 'validation') or cache['role'] != role:
        raise ValueError('role mismatch or forbidden test role')
    radar = torch.as_tensor(cache['radar']).detach().float().cpu()
    motion = torch.as_tensor(cache['motion']).detach().float().cpu()
    intervals = torch.as_tensor(cache['intervals']).double().cpu()
    n = len(motion)
    if radar.shape != (n,4,256) or motion.shape != (n,256) or intervals.shape != (n,2):
        raise ValueError('expected F[T,4,256], motion[T,256], intervals[T,2]')
    if not all(torch.isfinite(x).all() for x in (radar,motion,intervals)):
        raise ValueError('finite frozen observations required')
    if (intervals[:,1] <= intervals[:,0]).any() or (intervals[1:,0] <= intervals[:-1,0]).any():
        raise ValueError('increasing observation clock required')
    torch.testing.assert_close(motion,radar.mean(1),rtol=0,atol=0)
    valid = torch.as_tensor(physical['observation_valid']).bool().clone()
    mixed = torch.as_tensor(physical['mixed_update']).bool()
    common = torch.as_tensor(physical['common_start']).long().clone()
    if valid.shape != (n,) or mixed.shape != (n,) or common.ndim != 1:
        raise ValueError('matching physical masks required')
    if not len(common) or (common < 0).any() or (common+5 >= n).any() or (common[1:] <= common[:-1]).any():
        raise ValueError('invalid common future indices')
    for t in common.tolist():
        if not valid[t:t+6].all() or not torch.isclose(intervals[t:t+5,1],intervals[t+1:t+6,0],atol=1e-7,rtol=0).all():
            raise ValueError('common future crosses missing observations')
    labels = torch.as_tensor(labels).long().cpu()
    if len(labels) < n*2:
        raise ValueError('two frame labels required for each Update')
    pairs = labels[:n*2].reshape(n,2)
    eligible = valid & ~mixed & pairs[:,0].eq(pairs[:,1]) & pairs[:,0].ge(0) & pairs[:,0].lt(14)
    target = torch.where(eligible,pairs[:,0],torch.full((n,),-100))
    return JointRecord(cache['uid'],role,cache['metadata']['subject_id'],radar,motion,intervals,
                       valid,target,common,dict(cache.get('source_lock',{})))


def prefix_bounds(record: JointRecord, t: int, max_updates: int = 32) -> tuple[int,int]:
    if not 0 <= t < len(record.radar) or not record.observation_valid[t] or not 1 <= max_updates <= 32:
        raise ValueError('legal observed current index and maximum<=32 required')
    start = t
    while start > max(0,t-max_updates+1):
        if not record.observation_valid[start-1] or not torch.isclose(record.intervals[start-1,1],record.intervals[start,0],atol=1e-7,rtol=0):
            break
        start -= 1
    return start,t+1


def joint_schedule(records: list[JointRecord], seed: int, steps: int = 1000) -> list:
    if not records or any(r.role != 'train' for r in records):
        raise ValueError('fit/train records only')
    clean = [r.common_start[r.semantic_target[r.common_start].ge(0)].tolist() for r in records]
    if any(not x for x in clean) or steps <= 0:
        raise ValueError('nonempty semantic and physical populations required')
    rng = random.Random(seed)
    schedule = []
    for _ in range(steps):
        a,b = rng.randrange(len(records)),rng.randrange(len(records))
        schedule.append([(a,rng.choice(clean[a])),(b,rng.choice(records[b].common_start.tolist()))])
    return schedule


def fit_motion_contract(records: list[JointRecord], normalization: dict) -> dict:
    if not records or any(r.role != 'train' for r in records):
        raise ValueError('fit/train normalization and variance only')
    mean,std = (normalization[k].detach().float().cpu() for k in ('mean','std'))
    if mean.shape != (256,) or std.shape != (256,) or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError('finite fixed normalization required')
    currents, targets, ids = [],[],[]
    for i,r in enumerate(records):
        x = (r.motion_ref-mean)/std
        currents.append(x[r.common_start])
        targets.append(torch.stack([x[r.common_start+h] for h in HORIZONS],1))
        ids.append(torch.full((len(r.common_start),),i))
    current,target,identity = torch.cat(currents),torch.cat(targets),torch.cat(ids)
    R = torch.stack([torch.stack([((target[identity==i,j]-current[identity==i])**2).mean() for i in range(len(records))]).mean() for j in range(3)])
    if not torch.isfinite(R).all() or (R <= 0).any():
        raise ValueError('positive fit persistence MSE required')
    logvar = torch.stack([fit_persistence_variance(current,target[:,j],identity) for j in range(3)])
    return dict(mean=mean,std=std,R=R,logvar=logvar)


def make_joint_batch(records: list[JointRecord], stats: dict, samples: list, device: str) -> dict:
    if not samples:
        raise ValueError('nonempty samples required')
    prefixes, current, future, semantic, ids = [],[],[],[],[]
    for ri,t in samples:
        record = records[ri]
        if t not in record.common_start:
            raise ValueError('current index not in common population')
        start,end = prefix_bounds(record,t)
        prefixes.append(record.radar[start:end])
        current.append((record.motion_ref[t]-stats['mean'])/stats['std'])
        future.append(torch.stack([(record.motion_ref[t+h]-stats['mean'])/stats['std'] for h in HORIZONS]))
        semantic.append(record.semantic_target[t]);ids.append(ri)
    length = max(map(len,prefixes))
    F = torch.zeros(len(samples),length,4,256)
    mask = torch.zeros(len(samples),length,dtype=torch.bool)
    for i,prefix in enumerate(prefixes):
        F[i,:len(prefix)] = prefix;mask[i,:len(prefix)] = True
    return {k:v.to(device) for k,v in dict(F=F,mask=mask,m_ref=torch.stack(current),
             target=torch.stack(future),semantic=torch.stack(semantic).long(),recording_ids=torch.tensor(ids)).items()}
