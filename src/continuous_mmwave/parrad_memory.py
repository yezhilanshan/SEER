"""Bounded event-token memory. Captions never control its structure."""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass
class MemoryUnit:
    uid: str
    tokens: Tensor
    start: float
    end: float
    confirmed_at: float
    confidence: float
    key: Tensor
    caption: str | None = None
    status: str = 'refined'
    level: int = 0
    source_ids: tuple[str, ...] = ()
    count: int = 1


class EventMemory:
    def __init__(self, compressor, projection, *, l0_capacity=8, l1_capacity=8,
                 l2_capacity=32, group_size=4, merge_cosine=None, merge_gap_s=.18):
        if group_size < 2 or any(n < group_size for n in (l0_capacity, l1_capacity, l2_capacity)):
            raise ValueError('capacity must fit a compression group of at least two')
        self.compressor, self.projection = compressor, projection
        self.capacities = (l0_capacity, l1_capacity, l2_capacity)
        self.group_size, self.merge_cosine, self.merge_gap_s = group_size, merge_cosine, merge_gap_s
        self.levels: list[list[MemoryUnit]] = [[], [], []]
        self.promotions = [0, 0, 0]

    @torch.no_grad()
    def _compress(self, units, level):
        if any(u.status == 'provisional' for u in units):
            raise ValueError('refine provisional events before compression')
        device = next(self.compressor.parameters()).device
        lengths = torch.tensor([len(u.tokens) for u in units], device=device)
        tokens = torch.nn.utils.rnn.pad_sequence([u.tokens.to(device) for u in units], batch_first=True)[None]
        mask = torch.arange(tokens.shape[2],device=device)[None,None] < lengths[None,:,None]
        times = torch.tensor([[u.start for u in units]], device=device)
        result = self.compressor(tokens, mask,
                                 times, level=level)[0].detach()
        key = self.projection(result[None])[0].detach()
        return MemoryUnit('summary:' + ':'.join(u.uid for u in units), result, units[0].start,
                          units[-1].end, max(u.confirmed_at for u in units), min(u.confidence for u in units),
                          key, level=level, source_ids=tuple(s for u in units for s in (u.source_ids or (u.uid,))),
                          count=sum(u.count for u in units))

    def _bound(self, level):
        while len(self.levels[level]) > self.capacities[level]:
            group = self.levels[level][:self.group_size]
            summary = self._compress(group, min(level + 1, 2))
            del self.levels[level][:self.group_size]
            self.promotions[level] += 1
            if level == 2:
                self.levels[2].insert(0, summary)
            else:
                self.levels[level+1].append(summary)
                self._bound(level+1)

    def add(self, unit: MemoryUnit):
        if unit.level != 0 or unit.tokens.ndim != 2 or unit.tokens.shape[0] != 8:
            raise ValueError('L0 requires eight event tokens')
        if not unit.start < unit.end or not 0 <= unit.confidence <= 1 or unit.confirmed_at < unit.end:
            raise ValueError('invalid event interval/confidence/confirmation')
        if unit.status not in ('provisional', 'refined', 'captioned'):
            raise ValueError('unknown memory status')
        all_units = [u for level in self.levels for u in level]
        if any(unit.uid == u.uid or unit.uid in u.source_ids for u in all_units):
            raise ValueError('duplicate event id')
        if all_units and unit.start < max(u.end for u in all_units) - 1e-6:
            raise ValueError('events must be chronological and nonoverlapping')
        unit = replace(unit, tokens=unit.tokens.detach().clone(), key=unit.key.detach().clone())
        recent = self.levels[0]
        if (self.merge_cosine is not None and recent and recent[-1].status != 'provisional'
                and unit.status != 'provisional' and unit.start - recent[-1].end <= self.merge_gap_s
                and float(F.cosine_similarity(recent[-1].key[None], unit.key[None])) > self.merge_cosine):
            # A merged event stays in the newest L0 position. Padding/masks
            # allow its four tokens to share a future group with eight-token events.
            summary = self._compress([recent[-1], unit], 1)
            recent[-1] = replace(summary, level=0)
        else:
            recent.append(unit)
            self._bound(0)

    def refine(self, uid, tokens, key, *, caption=None):
        for unit in self.levels[0]:
            if unit.uid == uid:
                if tokens.shape != unit.tokens.shape:
                    raise ValueError('refinement must retain event dimensions')
                unit.tokens, unit.key = tokens.detach().clone(), key.detach().clone()
                unit.caption, unit.status = caption, 'captioned' if caption is not None else 'refined'
                return
        raise KeyError('event not in L0; provisional units must be refined before promotion')

    def context(self, query_key: Tensor, *, now: float, top_l1=2, top_l2=1,
                beta=.1, gamma=.1, tau=60., current: Tensor | None = None):
        if tau <= 0 or top_l1 < 0 or top_l2 < 0:
            raise ValueError('invalid retrieval settings')
        def select(level, top):
            def score(unit):
                cosine = float(F.cosine_similarity(unit.key[None], query_key.to(unit.key)[None]))
                return cosine + beta * math.exp(-max(0., now - unit.end)/tau) + gamma * unit.confidence
            ranked = sorted(self.levels[level], key=lambda u: (-score(u), u.start))[:top]
            return sorted(ranked, key=lambda u: u.start)
        selected = select(2, top_l2) + select(1, top_l1) + list(self.levels[0])
        tokens = [u.tokens for u in selected]
        if current is not None:
            tokens.append(current.squeeze(0) if current.ndim == 3 else current)
        if not tokens:
            raise ValueError('no memory tokens available')
        device = next(self.compressor.parameters()).device
        return torch.cat([t.to(device) for t in tokens])[None], selected

    def snapshot(self):
        from dataclasses import asdict
        return {'schema_version': 1, 'capacities': self.capacities, 'group_size': self.group_size,
                'promotions': list(self.promotions),
                'levels': [[{k: v.detach().cpu() if isinstance(v, Tensor) else v for k, v in asdict(u).items()}
                            for u in level] for level in self.levels]}

    def restore(self, snapshot):
        if snapshot['schema_version'] != 1 or tuple(snapshot['capacities']) != self.capacities or snapshot['group_size'] != self.group_size:
            raise ValueError('incompatible memory snapshot')
        device = next(self.compressor.parameters()).device
        self.levels = [[MemoryUnit(**{k: v.to(device) if isinstance(v, Tensor) else v for k, v in row.items()})
                        for row in level] for level in snapshot['levels']]
        self.promotions = list(snapshot['promotions'])
