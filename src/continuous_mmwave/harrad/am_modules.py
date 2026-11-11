"""Bounded event conditioning and fixed-anchor contrastive supervision."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class EventModulation(nn.Module):
    def __init__(self,radar_dim=256,event_dim=896,attention_dim=64):
        super().__init__(); self.radar_norm=nn.LayerNorm(radar_dim); self.event_norm=nn.LayerNorm(event_dim)
        self.query=nn.Linear(radar_dim,attention_dim); self.key=nn.Linear(event_dim,attention_dim)
        self.value=nn.Linear(event_dim,attention_dim)
        self.affine=nn.Sequential(nn.Linear(attention_dim,128),nn.GELU(),nn.Linear(128,2*radar_dim))
        nn.init.zeros_(self.affine[-1].weight); nn.init.zeros_(self.affine[-1].bias)

    def forward(self,z,previous):
        # Frozen Qwen emits FP16 states even when evaluation autocast is off.
        normalized=self.radar_norm(z.float()); event=self.event_norm(previous.detach().float())
        q=self.query(normalized); k=self.key(event); v=self.value(event)
        weights=(q.float()@k.float().transpose(-1,-2)/math.sqrt(q.shape[-1])).softmax(-1)
        context=weights@v.float(); gamma,beta=self.affine(context).chunk(2,-1)
        return z+.1*(gamma.tanh()*normalized+beta.tanh())


class AnchoredContrastive(nn.Module):
    def __init__(self,temperature=.1):
        super().__init__()
        if temperature<=0: raise ValueError('positive contrastive temperature required')
        self.temperature=temperature

    def forward(self,query,positive,labels,bank,bank_labels):
        if query.shape!=positive.shape or query.ndim!=2 or bank.ndim!=2 or bank.shape[1]!=query.shape[1]:
            raise ValueError('incompatible contrastive vectors')
        if bank_labels.shape!=(len(bank),) or labels.shape!=(len(query),) or not len(bank) or (bank_labels<0).any():
            raise ValueError('invalid fixed negative bank')
        q=F.normalize(query.float(),dim=-1); p=F.normalize(positive.detach().float(),dim=-1)
        keys=F.normalize(bank.detach().float(),dim=-1)
        allowed=labels[:,None]!=bank_labels[None]; valid=labels>=0
        counts=allowed.sum(1)
        if (valid&(counts==0)).any(): raise ValueError('no different-class negative for a valid query')
        positive_score=(q*p).sum(-1)/self.temperature
        negatives=(q@keys.T/self.temperature).masked_fill(~allowed,-torch.inf)
        loss=(torch.logsumexp(torch.cat([positive_score[:,None],negatives],1),1)-positive_score)
        loss=loss/torch.log1p(counts.clamp(min=1).float())
        return torch.where(valid,loss,torch.zeros_like(loss))
