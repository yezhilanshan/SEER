"""Causal frame encoders with an offline, full-instance classification readout."""
import torch
from torch import nn
from continuous_mmwave.models.md_gru_encoder import CausalMDGRUEncoder


class SegmentClassifier(nn.Module):
    def __init__(self, arm: str, dropout: float=.1):
        super().__init__()
        if arm not in ('md','rd'): raise ValueError('arm must be md or rd')
        self.arm=arm
        if arm=='rd':
            layers=[]; cin=1
            for cout in (16,32,64):
                layers.extend([nn.Conv2d(cin,cout,3,stride=2,padding=1,bias=False),nn.GroupNorm(4,cout),nn.GELU()])
                cin=cout
            self.spatial=nn.Sequential(*layers,nn.AdaptiveAvgPool2d(1),nn.Flatten(1))
        self.temporal=CausalMDGRUEncoder(256 if arm=='md' else 64,128,dropout)
        self.classifier=nn.Linear(128,12)

    def forward(self,x,mask):
        if x.ndim!=(3 if self.arm=='md' else 4) or mask.shape!=x.shape[:2]: raise ValueError('invalid clip shape')
        if not mask.any(1).all() or (mask[:,1:]&~mask[:,:-1]).any(): raise ValueError('nonempty right-padded mask required')
        if self.arm=='rd':
            valid=x[mask].unsqueeze(1)
            encoded=torch.cat([self.spatial(v) for v in valid.split(128)],dim=0)
            frames=x.new_zeros(x.shape[0],x.shape[1],64); frames[mask]=encoded
        else: frames=x
        hidden=self.temporal(frames,frame_mask=mask).frame_hidden
        pooled=(hidden*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
        return self.classifier(pooled)
