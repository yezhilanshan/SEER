"""Shared event/text semantic projection."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional


class SemanticProjection(nn.Module):
    """共享的事件/文本语义投影：注意力池化 -> MLP -> L2 归一化。

    事件 token 与文本隐层走完全相同的一套参数，因此两者落在同一单位球面上，
    可直接用 cosine/InfoNCE 比较。
    """

    def __init__(self, llm_dim: int = 896, semantic_dim: int = 128) -> None:
        super().__init__()
        # 单个 learnable query 用于对序列做注意力池化（event 与 text 共用）。
        self.semantic_query = nn.Parameter(torch.empty(llm_dim))
        self.projection = nn.Sequential(nn.Linear(llm_dim, 256), nn.GELU(), nn.Linear(256, semantic_dim))
        nn.init.normal_(self.semantic_query, std=0.02)

    def forward(self, hidden: Tensor, mask: Tensor | None = None) -> Tensor:
        """hidden [B,L,llm_dim] -> 归一化后的 [B,semantic_dim]。"""
        if hidden.ndim != 3:
            raise ValueError("hidden must be [B,L,D]")
        hidden = hidden.to(self.semantic_query.dtype)
        # 用 query 与每个位置的点积作注意力权重（已除 sqrt(d) 稳定数值）。
        scores = torch.einsum("bld,d->bl", hidden, self.semantic_query) / math.sqrt(hidden.shape[-1])
        if mask is not None:
            # 要求每个样本至少有一个有效位置，否则 softmax 无意义。
            if mask.shape != hidden.shape[:2] or not mask.bool().any(1).all():
                raise ValueError("mask must be valid [B,L]")
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        pooled = torch.einsum("bl,bld->bd", scores.softmax(-1), hidden)
        return functional.normalize(self.projection(pooled), dim=-1)
