"""Bounded rewind features plus temporary episode pins and pooled Commit input."""
from __future__ import annotations

import math
from dataclasses import dataclass
import torch
from torch import Tensor


@dataclass(frozen=True)
class FeatureFrame:
    sequence: int
    timestamp: float
    tokens: Tensor
    mask: Tensor
    local_probability: float
    radar_probability: float
    semantic_key: Tensor
    evidence: Tensor | None = None


class FeatureBuffer:
    def __init__(self, retention_s):
        if not math.isfinite(retention_s) or retention_s <= 0:
            raise ValueError('buffer retention must be positive')
        self.retention_s = retention_s
        self.frames = []
        self.pins = {}

    def append(self, frame):
        if self.frames and (frame.sequence <= self.frames[-1].sequence or frame.timestamp <= self.frames[-1].timestamp):
            raise ValueError('feature sequence/timestamps must increase')
        self.frames.append(frame)
        self._evict()

    def pin(self, owner, start_time):
        if owner in self.pins or not math.isfinite(start_time):
            raise ValueError('duplicate/invalid buffer pin')
        self.pins[owner] = start_time

    def release(self, owner):
        self.pins.pop(owner,None)
        self._evict()

    def _evict(self):
        if self.frames:
            cutoff = self.frames[-1].timestamp-self.retention_s
            if self.pins:
                cutoff = min(cutoff,min(self.pins.values()))
            self.frames = [f for f in self.frames if f.timestamp >= cutoff-1e-9]

    def since(self, start_time, *, after_sequence=-1):
        return tuple(f for f in self.frames if f.timestamp >= start_time-1e-9 and f.sequence > after_sequence)


@dataclass(frozen=True)
class FeatureChunk:
    start: float
    end: float
    tokens: Tensor
    mask: Tensor
    valid_counts: Tensor


class CommitArchive:
    """Pool only the old prefix. Recent rewind features remain exact.

    This prevents a long Active event from retaining unbounded raw features.
    Pooling is explicit and approximate, as is the existing Commit token cap.
    A chunk's start/end records all observations it represents.
    """
    def __init__(self, max_tokens=256, preserve_s=2.5):
        if max_tokens < 4 or not math.isfinite(preserve_s) or preserve_s <= 0:
            raise ValueError('invalid Commit archive bound')
        self.max_tokens, self.preserve_s = max_tokens, preserve_s
        self.chunks = []
        self.pooled_updates = 0
        self.pin_start = None

    def append(self, frame):
        self.chunks.append(FeatureChunk(frame.timestamp,frame.timestamp,frame.tokens,frame.mask,
                                        frame.mask.to(frame.tokens.dtype).unsqueeze(-1)))
        self._compact(frame.timestamp)

    def _compact(self, now):
        # Never merge across a possible localization boundary in the rewind window.
        while sum(c.tokens.shape[1] for c in self.chunks) > self.max_tokens and len(self.chunks) >= 2:
            a,b = self.chunks[:2]
            cutoff = now-self.preserve_s
            if self.pin_start is not None:
                cutoff = min(cutoff,self.pin_start)
            if b.end >= cutoff-1e-9:
                break
            counts = a.valid_counts+b.valid_counts
            tokens = (a.tokens*a.valid_counts+b.tokens*b.valid_counts)/counts.clamp_min(1)
            merged = FeatureChunk(a.start,b.end,tokens,counts.squeeze(-1)>0,counts)
            self.chunks[:2] = [merged]
            self.pooled_updates += 1

    def partition(self, boundary):
        if any(c.start < boundary-1e-9 and c.end >= boundary-1e-9 for c in self.chunks):
            raise RuntimeError('boundary crosses a pooled Commit chunk; insufficient rewind preservation')
        return ([c for c in self.chunks if c.end < boundary-1e-9],
                [c for c in self.chunks if c.start >= boundary-1e-9])

    @staticmethod
    def tensors(chunks):
        if not chunks:
            raise ValueError('empty Commit interval')
        return torch.cat([c.tokens for c in chunks],1),torch.cat([c.mask for c in chunks],1)
