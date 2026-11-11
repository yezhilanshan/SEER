"""Shared Query Compressor for event and higher-level memory tokens."""
from __future__ import annotations

import torch
from torch import Tensor, nn


class QueryCompressor(nn.Module):
    def __init__(self, llm_dim: int, internal_dim: int = 1024, heads: int = 8,
                 layers: int = 2, summary_tokens: int = 4, max_events: int = 32,
                 max_tokens: int = 8, dropout: float = 0.1):
        super().__init__()
        self.llm_dim, self.max_events, self.max_tokens = llm_dim, max_events, max_tokens
        self.settings = dict(llm_dim=llm_dim, internal_dim=internal_dim, heads=heads, layers=layers,
                             summary_tokens=summary_tokens, max_events=max_events,
                             max_tokens=max_tokens, dropout=dropout)
        self.input_projection = nn.Linear(llm_dim, internal_dim)
        self.event_position = nn.Embedding(max_events, internal_dim)
        self.token_position = nn.Embedding(max_tokens, internal_dim)
        self.level_embedding = nn.Embedding(3, internal_dim)
        self.time_projection = nn.Linear(1, internal_dim)
        self.queries = nn.Parameter(torch.randn(summary_tokens, internal_dim) * .02)
        layer = nn.TransformerDecoderLayer(internal_dim, heads, internal_dim * 4,
                                           dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(internal_dim))
        self.output_projection = nn.Linear(internal_dim, llm_dim)

    def forward(self, tokens: Tensor, mask: Tensor, times: Tensor, *, level: int) -> Tensor:
        if tokens.ndim != 4 or mask.shape != tokens.shape[:3] or times.shape != tokens.shape[:2]:
            raise ValueError('expected tokens[B,G,T,D], mask[B,G,T], times[B,G]')
        b, g, t, d = tokens.shape
        if d != self.llm_dim or g > self.max_events or t > self.max_tokens or level not in (1, 2):
            raise ValueError('unsupported memory dimensions or level')
        if not mask.flatten(1).bool().any(1).all():
            raise ValueError('each sample needs valid memory tokens')
        valid_events = mask.bool().any(-1)
        if not torch.isfinite(times[valid_events]).all() or not torch.isfinite(tokens[mask.bool()]).all():
            raise ValueError('valid memory input must be finite')
        tokens = tokens.masked_fill(~mask.bool()[..., None], 0).to(self.queries.dtype)
        times = times.to(tokens.dtype).masked_fill(~valid_events, 0)
        origin = times.masked_fill(~valid_events, float('inf')).amin(1, keepdim=True)
        relative = (times - origin).clamp_min(0).log1p().to(tokens.dtype)
        hidden = self.input_projection(tokens)
        hidden = hidden + self.event_position(torch.arange(g, device=tokens.device))[None, :, None]
        hidden = hidden + self.token_position(torch.arange(t, device=tokens.device))[None, None]
        hidden = hidden + self.time_projection(relative[..., None])[:, :, None]
        hidden = hidden + self.level_embedding.weight[level]
        query = self.queries[None].expand(b, -1, -1)
        return self.output_projection(self.decoder(query, hidden.flatten(1, 2),
                                      memory_key_padding_mask=~mask.flatten(1).bool()))
