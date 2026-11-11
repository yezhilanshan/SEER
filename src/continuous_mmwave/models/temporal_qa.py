"""Generation-only temporal residual; original QA and retrieval stay frozen."""
import math

import torch
from torch import nn


class TemporalQAFusion(nn.Module):
    def __init__(self, base, bottleneck=256, encoding_dim=16, tokens_per_event=8, dim=None):
        super().__init__()
        if encoding_dim < 2 or encoding_dim % 2 or tokens_per_event != 8:
            raise ValueError('expected an even encoding width and eight-token events')
        if dim is not None and dim!=base.settings['dim']:
            raise ValueError('export dimension differs from frozen QA')
        self.base=base.eval().requires_grad_(False)
        dim=base.settings['dim']
        self.settings=dict(dim=dim,bottleneck=bottleneck,encoding_dim=encoding_dim,tokens_per_event=tokens_per_event)
        self.norm=nn.LayerNorm(dim)
        self.input=nn.Linear(dim+3*encoding_dim,bottleneck)
        self.output=nn.Linear(bottleneck,dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        self.register_buffer('frequencies',torch.exp(-math.log(10000)*torch.arange(0,encoding_dim,2,dtype=torch.float64)/encoding_dim))

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def _encode(self, values):
        phase=values[...,None]*self.frequencies
        return torch.stack((phase.sin(),phase.cos()),dim=-1).flatten(-2)

    def features(self, starts, ends, *, arm):
        if arm not in ('none','order','time') or starts.ndim!=2 or starts.shape!=ends.shape or starts.shape[1] not in (1,2,4):
            raise ValueError('invalid temporal arm or event metadata shape')
        starts=starts.to(device=self.frequencies.device,dtype=torch.float64)
        ends=ends.to(device=self.frequencies.device,dtype=torch.float64)
        if not torch.isfinite(starts).all() or not torch.isfinite(ends).all() or not (ends>starts).all():
            raise ValueError('event timestamps must be finite with positive duration')
        if starts.shape[1]>1 and (not (starts[:,1:]>starts[:,:-1]).all() or (ends[:,:-1]>starts[:,1:]+1e-6).any()):
            raise ValueError('events must be chronological and nonoverlapping')
        b,n=starts.shape
        rank=torch.arange(n,device=starts.device,dtype=torch.float64)[None].expand(b,-1)
        relative=torch.log1p(starts-starts[:,:1])
        duration=torch.log1p(ends-starts)
        features=torch.cat((self._encode(rank),self._encode(relative),self._encode(duration)),dim=-1)
        if arm=='none': features=torch.zeros_like(features)
        elif arm=='order': features[:,:,self.settings['encoding_dim']:]=0
        return features.repeat_interleave(self.settings['tokens_per_event'],dim=1)

    def forward(self, tokens, starts, ends, *, arm):
        if tokens.ndim!=3 or tokens.shape[-1]!=self.settings['dim'] or tokens.shape[:2]!=(starts.shape[0],8*starts.shape[1]):
            raise ValueError('expected complete [B,8*events,D] event tokens')
        features=self.features(starts,ends,arm=arm)
        with torch.no_grad(): content=self.base(tokens)
        if starts.shape[1]==1: return content
        hidden=torch.cat((self.norm(content),features.to(content.dtype)),dim=-1)
        return content+self.output(torch.nn.functional.gelu(self.input(hidden)))
