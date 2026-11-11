"""Generation-only residual adapter; retrieval always uses original tokens."""
import torch
from torch import nn


class EventQAAdapter(nn.Module):
    def __init__(self, dim=896, bottleneck=256):
        super().__init__()
        self.settings=dict(dim=dim,bottleneck=bottleneck)
        self.norm=nn.LayerNorm(dim)
        self.input=nn.Linear(dim,bottleneck)
        self.output=nn.Linear(bottleneck,dim)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, tokens):
        if tokens.ndim!=3 or tokens.shape[-1]!=self.settings['dim']:
            raise ValueError('QA adapter expects [B,T,D]')
        return tokens+self.output(torch.nn.functional.gelu(self.input(self.norm(tokens))))
