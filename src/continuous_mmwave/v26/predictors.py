"""Capacity-controlled predictors with identical observable radar prefixes."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Predictor(nn.Module):
    def __init__(self, kind, config):
        super().__init__()
        self.kind, self.config = kind, dict(config)
        d, radar, event = config['motion_dim'], config['radar_dim'], config['event_dim']
        hidden, width = config['hidden'], config['head_width']
        self.logvar_min, self.logvar_max = config.get('logvar_min',-6.), config.get('logvar_max',2.)
        if not self.logvar_min < 0 < self.logvar_max: raise ValueError('variance bounds must enclose zero')
        if kind == 'event':
            self.event_norm = nn.LayerNorm(event)
            self.query = nn.Parameter(torch.randn(event)*.02)
            self.project = nn.Linear(event,d)
        elif kind in ('gru','tcn'):
            self.input = nn.Linear(4*radar,hidden)
            if kind == 'gru': self.gru = nn.GRU(hidden,hidden,batch_first=True)
            else:
                self.convs = nn.ModuleList([nn.Conv1d(hidden,hidden,2,padding=dilation,dilation=dilation)
                                           for dilation in (1,2,4,8,16)])
            self.project = nn.Linear(hidden,d)
        elif kind != 'event_free': raise ValueError('unknown prediction arm')
        self.head = nn.Sequential(nn.Linear(d if kind=='event_free' else 2*d,width),nn.GELU(),nn.Linear(width,2*d))
        with torch.no_grad():
            self.head[-1].weight[d:].zero_()
            probability = -self.logvar_min/(self.logvar_max-self.logvar_min)
            self.head[-1].bias[d:].fill_(math.log(probability/(1-probability)))

    def forward(self,motion,radar_prefix,event_tokens,prefix_mask):
        motion = motion.float()
        if self.kind == 'event_free': features = motion
        elif self.kind == 'event':
            value = self.event_norm(event_tokens.float())
            weight = (value @ self.query/math.sqrt(value.shape[-1])).softmax(1)
            context = self.project((value*weight[...,None]).sum(1))
            features = torch.cat((motion,context),-1)
        else:
            valid = prefix_mask.bool()
            lengths = valid.sum(1)
            if (lengths == 0).any() or (valid[:,1:] & ~valid[:,:-1]).any():
                raise ValueError('right-padded nonempty causal prefix required')
            value = torch.nn.functional.gelu(self.input(radar_prefix.float().flatten(2)))
            value = value*valid[...,None]
            if self.kind == 'gru':
                packed = pack_padded_sequence(value,lengths.cpu(),batch_first=True,enforce_sorted=False)
                encoded,_ = self.gru(packed)
                encoded,_ = pad_packed_sequence(encoded,batch_first=True,total_length=value.shape[1])
            else:
                encoded = value.transpose(1,2)
                for conv in self.convs:
                    encoded = encoded + torch.nn.functional.gelu(conv(encoded)[...,:value.shape[1]])
                encoded = encoded.transpose(1,2)
            context = self.project(encoded[torch.arange(len(motion),device=motion.device),lengths-1])
            features = torch.cat((motion,context),-1)
        mean,raw = self.head(features).chunk(2,-1)
        return mean, self.logvar_min+(self.logvar_max-self.logvar_min)*raw.sigmoid()


def make_predictor(kind,config):
    return Predictor(kind,config)


class ResidualPredictor(Predictor):
    """Same capacity and variance learning as Predictor, with a zero mean residual."""
    def __init__(self,kind,config):
        super().__init__(kind,config)
        with torch.no_grad():
            self.head[-1].weight[:config['motion_dim']].zero_()
            self.head[-1].bias[:config['motion_dim']].zero_()

    def forward(self,motion,radar_prefix,event_tokens,prefix_mask):
        residual,logvar = super().forward(motion,radar_prefix,event_tokens,prefix_mask)
        return motion.float()+residual,logvar


def make_residual_predictor(kind,config):
    return ResidualPredictor(kind,config)


def select_architecture(kind,config):
    best = None
    # Only shapes/parameter count drive selection, never data or scores.
    for hidden in ((64,) if kind in ('event','event_free') else (32,64,96,128,160,192,256)):
        for width in (64,96,128,160,192,256,384,512):
            candidate = dict(config,hidden=hidden,head_width=width)
            count = sum(p.numel() for p in make_predictor(kind,candidate).parameters())
            key = (abs(count-config['parameter_budget']),hidden,width)
            if best is None or key < best[0]: best = key,dict(candidate,parameter_count=count)
    chosen = best[1]
    chosen['capacity_matched'] = abs(chosen['parameter_count']-config['parameter_budget'])/config['parameter_budget'] <= .1
    return chosen
