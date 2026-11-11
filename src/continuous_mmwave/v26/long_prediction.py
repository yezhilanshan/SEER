"""Full-history causal residual heads and frozen-state diagnostic statistics."""
from __future__ import annotations
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from .predictors import Predictor


class MatchedResidualPredictor(Predictor):
    def __init__(self,kind,config):
        super().__init__(kind,config)
        if kind=='tcn':
            dilations=config['dilations']
            if not dilations or any(d!=2**i for i,d in enumerate(dilations)):
                raise ValueError('complete causal dilation sequence required')
            self.convs=nn.ModuleList([nn.Conv1d(config['hidden'],config['hidden'],2,dilation=d) for d in dilations])
        with torch.no_grad():
            self.head[-1].weight[:config['motion_dim']].zero_();self.head[-1].bias[:config['motion_dim']].zero_()

    def history_context(self,radar,mask):
        valid=mask.bool();lengths=valid.sum(1)
        if valid.shape!=radar.shape[:2] or (lengths==0).any() or (valid[:,1:]&~valid[:,:-1]).any():
            raise ValueError('right-padded nonempty complete history required')
        value=F.gelu(self.input(radar.float().flatten(2)))*valid[...,None]
        if self.kind=='gru':
            encoded,_=self.gru(pack_padded_sequence(value,lengths.cpu(),batch_first=True,enforce_sorted=False))
            encoded,_=pad_packed_sequence(encoded,batch_first=True,total_length=value.shape[1])
        elif self.kind=='tcn':
            encoded=value.transpose(1,2)
            for conv in self.convs:encoded=encoded+F.gelu(conv(F.pad(encoded,(conv.dilation[0],0))))
            encoded=encoded.transpose(1,2)
        else:raise ValueError('history context requires GRU/TCN')
        return self.project(encoded)

    def _output(self,motion,context):
        features=motion.float() if context is None else torch.cat((motion.float(),context),-1)
        residual,raw=self.head(features).chunk(2,-1)
        return motion.float()+residual,self.logvar_min+(self.logvar_max-self.logvar_min)*raw.sigmoid()

    def forward(self,motion,radar_prefix,event_tokens,prefix_mask):
        if self.kind in ('event','event_free'):
            residual,lv=super().forward(motion,radar_prefix,event_tokens,prefix_mask)
            return motion.float()+residual,lv
        context=self.history_context(radar_prefix,prefix_mask)
        lengths=prefix_mask.sum(1)
        return self._output(motion,context[torch.arange(len(motion),device=motion.device),lengths-1])

    def forward_sequence(self,motion,radar,event,mask):
        if self.kind in ('gru','tcn'):return self._output(motion,self.history_context(radar,mask))
        b,t=motion.shape[:2]
        value,lv=self.forward(motion.reshape(b*t,-1),None,event.reshape(b*t,*event.shape[2:]),None)
        return value.reshape(b,t,-1),lv.reshape(b,t,-1)


def matched_architecture(kind,max_history):
    base=dict(motion_dim=256,radar_dim=256,event_dim=896,hidden=128,head_width=128,logvar_min=-6.,logvar_max=2.)
    if kind=='event_free':base.update(hidden=64,head_width=512)
    if kind=='event':base.update(hidden=64,head_width=160)
    if kind=='tcn':
        dilations=[2**i for i in range(max(1,math.ceil(math.log2(max_history))))]
        best=None
        # Closed-form count: input, conv stack, projection, 2*d -> width -> 2*d head.
        for hidden in (48,64,80,96,112,128):
            for width in range(16,513,16):
                count=1025*hidden+len(dilations)*(2*hidden*hidden+hidden)+256*hidden+256+1025*width+512
                candidate=(abs(count-400000),hidden,width)
                if best is None or candidate<best:best=candidate
        base.update(hidden=best[1],head_width=best[2],dilations=dilations,receptive_field=1+sum(dilations))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0);base['parameter_count']=sum(p.numel() for p in MatchedResidualPredictor(kind,base).parameters())
    return base


def make_long_batch(records,stats,samples,horizon,arm,device):
    motions=[];targets=[];events=[];prefixes=[]
    for ri,si in samples:
        record=records[ri];t=int(record['selected'][si]);start=int(record['segment_start'][t]);future=t+horizon
        if not 0<=start<=t<future<len(record['motion']):raise ValueError('invalid complete causal sample')
        motion=record['motion'];motions.append((motion[t]-stats['mean'])/stats['std'])
        targets.append((motion[future]-stats['mean'])/stats['std'])
        source=int(record['shuffle_donor'][t]) if arm=='event_shuffle' else t
        if source<0 or (arm=='event_shuffle' and source>=t):raise ValueError('strictly past donor required')
        e=record['events'][source]
        events.append(torch.zeros_like(e) if arm=='event_zero' else e)
        prefixes.append((record['radar'][start:t+1]-stats['mean'])/stats['std'])
    length=max(map(len,prefixes));radar=torch.zeros(len(samples),length,*prefixes[0].shape[1:])
    mask=torch.zeros(len(samples),length,dtype=torch.bool)
    for i,prefix in enumerate(prefixes):radar[i,:len(prefix)]=prefix;mask[i,:len(prefix)]=True
    batch=dict(motion=torch.stack(motions),target=torch.stack(targets),event_tokens=torch.stack(events),
               radar_prefix=radar,prefix_mask=mask,recording_ids=torch.tensor([ri for ri,_ in samples]))
    return {key:value.detach().to(device) for key,value in batch.items()}


def state_dynamics(events):
    value=events.double();flat=value.flatten(1);energy=flat.square().sum(-1)
    rms=(energy/flat.shape[1]).sqrt();delta=np.full(len(flat),np.nan);cosine=np.full(len(flat),np.nan)
    delta[1:]=(flat[1:]-flat[:-1]).square().mean(-1).sqrt().numpy()
    cosine[1:]=((flat[1:]*flat[:-1]).sum(-1)/(energy[1:]*energy[:-1]).sqrt().clamp_min(1e-30)).numpy()
    dominance=(value.square().amax(-1).sum(-1)/energy.clamp_min(1e-30)).numpy()
    high=rms.numpy()>3
    candidates=np.flatnonzero(np.convolve(high.astype(int),np.ones(32,dtype=int),'valid')==32) if len(high)>=32 else []
    return dict(rms=rms.numpy(),step_delta_rms=delta,adjacent_cosine=cosine,top_coordinate_energy=dominance,
                token_rms=value.square().mean(-1).sqrt().numpy(),first_sustained_high=int(candidates[0]) if len(candidates) else None)
