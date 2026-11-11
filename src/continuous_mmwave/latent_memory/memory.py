"""Bounded time-based memory; annotations never determine writes or promotion."""
from dataclasses import dataclass, replace
import torch


@dataclass(eq=False)
class MemoryItem:
    values: torch.Tensor
    weights: torch.Tensor
    start_frame: int
    end_frame: int
    level: int = 0

    @property
    def mask(self):
        return self.weights > 0


class TemporalMemory:
    def __init__(self, capacities=(16,16,16), group_size=4):
        if len(capacities)!=3 or group_size<2 or any(c<group_size for c in capacities):
            raise ValueError('three capacities must each fit a compression group >= 2')
        self.capacities=tuple(capacities); self.group_size=group_size
        self.levels=[[],[],[]]; self.observed_frames=0

    def append(self, item):
        if (item.level!=0 or item.values.shape!=(2,4,64) or item.weights.shape!=(2,)
            or item.start_frame!=self.observed_frames or item.end_frame<=item.start_frame
            or not torch.isfinite(item.values).all() or not torch.isfinite(item.weights).all()
            or (item.weights<0).any() or float(item.weights.sum())!=item.end_frame-item.start_frame):
            raise ValueError('invalid/noncontiguous time unit')
        self.levels[0].append(replace(item,values=item.values.detach().cpu().clone(),
                                      weights=item.weights.detach().cpu().clone()))
        self.observed_frames=item.end_frame
        self._bound(0)

    def _bound(self, level):
        while len(self.levels[level])>self.capacities[level]:
            group=self.levels[level][:self.group_size]
            weights=torch.stack([x.weights for x in group]).sum(0)
            values=sum(x.values*x.weights[:,None,None] for x in group)/weights.clamp_min(1)[:,None,None]
            target=min(level+1,2)
            summary=MemoryItem(values,weights,group[0].start_frame,group[-1].end_frame,target)
            del self.levels[level][:self.group_size]
            if target==level:
                self.levels[level].insert(0,summary)
            else:
                self.levels[target].append(summary); self._bound(target)

    def candidates(self, group):
        if group not in (0,1,2): raise ValueError('invalid LMT group')
        return sorted((x for level in self.levels[:group+1] for x in level),key=lambda x:x.start_frame)

    def export(self):
        return dict(schema=1,capacities=self.capacities,group_size=self.group_size,observed_frames=self.observed_frames,
                    levels=[[dict(values=x.values.clone(),weights=x.weights.clone(),start_frame=x.start_frame,
                                  end_frame=x.end_frame,level=x.level) for x in level] for level in self.levels])

    @classmethod
    def restore(cls, saved):
        if saved['schema']!=1: raise ValueError('unknown memory schema')
        result=cls(saved['capacities'],saved['group_size'])
        result.levels=[[MemoryItem(**{**x,'values':x['values'].clone(),'weights':x['weights'].clone()})
                        for x in level] for level in saved['levels']]
        result.observed_frames=saved['observed_frames']
        items=result.candidates(2)
        cursor=0
        for x in items:
            if (x.start_frame!=cursor or x.end_frame<=cursor or x.values.shape!=(2,4,64)
                or x.weights.shape!=(2,) or (x.weights<0).any() or not torch.isfinite(x.values).all()
                or not torch.isfinite(x.weights).all() or float(x.weights.sum())!=x.end_frame-x.start_frame):
                raise ValueError('corrupt memory snapshot')
            cursor=x.end_frame
        if (cursor!=result.observed_frames or len(result.levels)!=3
            or any(len(v)>c or any(x.level!=i for x in v) for i,(v,c) in enumerate(zip(result.levels,result.capacities)))):
            raise ValueError('corrupt memory coverage/capacity')
        return result
