"""Frozen pretrained RD front end with explicit state and chunk-independent outputs."""
import torch
from torch import nn
from torch.nn import functional as F
from .model import SegmentClassifier


class FrozenRDStream(nn.Module):
    def __init__(self, backbone: SegmentClassifier):
        super().__init__()
        if backbone.arm!='rd': raise ValueError('RD checkpoint required')
        self.backbone=backbone.requires_grad_(False).eval()

    def train(self,mode=True):
        super().train(False); return self

    @torch.no_grad()
    def forward_chunk(self,x,mask=None,state=None):
        if x.ndim!=4 or x.shape[-2:]!=(40,64): raise ValueError('RD input must be [B,T,40,64]')
        if mask is None: mask=torch.ones(x.shape[:2],dtype=torch.bool,device=x.device)
        if mask.shape!=x.shape[:2] or (mask[:,1:]&~mask[:,:-1]).any(): raise ValueError('right-padded mask required')
        frames=x.new_zeros(*x.shape[:2],64); cells=x.new_zeros(*x.shape[:2],4,64)
        if mask.any():
            maps=[]
            for v in x[mask].unsqueeze(1).split(128):
                for layer in list(self.backbone.spatial.children())[:-2]: v=layer(v)
                maps.append(v)
            maps=torch.cat(maps,0)
            frames[mask]=maps.mean((-2,-1))
            cells[mask]=F.adaptive_avg_pool2d(maps,(2,2)).flatten(2).transpose(1,2)
        temporal=self.backbone.temporal(frames,frame_mask=mask,state=state)
        return {'frame_features':frames,'cells':cells,'hidden':temporal.frame_hidden,'state':temporal.state}


class PairBuffer:
    def __init__(self): self.pending=None

    def push(self,frames):
        if frames.ndim!=3 or frames.shape[1:]!=(4,64): raise ValueError('cells must be [T,4,64]')
        if self.pending is not None: frames=torch.cat((self.pending,frames),0)
        paired=len(frames)//2*2
        result=[(frames[i:i+2],torch.ones(2,dtype=torch.bool,device=frames.device)) for i in range(0,paired,2)]
        self.pending=frames[paired:] if paired<len(frames) else None
        return result

    def finish(self):
        if self.pending is None: return []
        values=torch.cat((self.pending,torch.zeros_like(self.pending)),0)
        result=[(values,torch.tensor([True,False],device=values.device))]; self.pending=None
        return result
