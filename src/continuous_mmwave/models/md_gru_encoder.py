"""Small causal GRU encoder for Micro-Doppler streams."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _right_padded(mask: Tensor) -> bool:
    return not bool((mask[:, 1:] & ~mask[:, :-1]).any())


@dataclass(frozen=True)
class CausalMDGRUState:
    hidden: Tensor


@dataclass(frozen=True)
class CausalMDGRUOutput:
    frame_hidden: Tensor
    state: CausalMDGRUState
    frame_mask: Tensor


class CausalMDGRUEncoder(nn.Module):
    """Encode ``[batch,time,doppler]`` MD streams with carried GRU state."""

    def __init__(self, md_bins: int, hidden_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        if md_bins <= 0 or hidden_dim <= 0:
            raise ValueError("MD-GRU dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.md_bins = md_bins
        self.hidden_dim = hidden_dim
        self.input_norm = nn.LayerNorm(md_bins)
        self.input_projection = nn.Linear(md_bins, hidden_dim)
        self.input_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output_dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        md: Tensor,
        frame_mask: Tensor | None = None,
        state: CausalMDGRUState | None = None,
    ) -> CausalMDGRUOutput:
        if md.ndim != 3 or md.shape[-1] != self.md_bins or md.shape[1] == 0:
            raise ValueError(f"MD input must be [B,T,{self.md_bins}] with non-zero time")
        if not torch.isfinite(md).all():
            raise ValueError("MD input must be finite")
        batch, frames = md.shape[:2]
        if frame_mask is None:
            frame_mask = torch.ones((batch, frames), dtype=torch.bool, device=md.device)
        if frame_mask.shape != (batch, frames):
            raise ValueError("frame_mask must be [B,T]")
        frame_mask = frame_mask.bool()
        if not _right_padded(frame_mask):
            raise ValueError("MD-GRU frame_mask must be right padded")

        projected = functional.gelu(self.input_projection(self.input_norm(md)))
        projected = self.input_dropout(projected) * frame_mask.unsqueeze(-1)
        initial_hidden = (
            projected.new_zeros((1, batch, self.hidden_dim))
            if state is None
            else state.hidden
        )
        if initial_hidden.shape != (1, batch, self.hidden_dim):
            raise ValueError("invalid MD-GRU hidden state")

        lengths = frame_mask.long().sum(1)
        packed = pack_padded_sequence(
            projected,
            lengths.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, next_hidden = self.gru(packed, initial_hidden)
        recurrent, _ = pad_packed_sequence(
            packed_output,
            batch_first=True,
            total_length=frames,
        )
        empty = lengths.eq(0)
        next_hidden = torch.where(empty.view(1, batch, 1), initial_hidden, next_hidden)
        frame_hidden = self.output_norm(projected + self.output_dropout(recurrent))
        frame_hidden = frame_hidden * frame_mask.unsqueeze(-1)
        return CausalMDGRUOutput(
            frame_hidden=frame_hidden,
            state=CausalMDGRUState(next_hidden),
            frame_mask=frame_mask,
        )


def detach_causal_md_gru_state(state: CausalMDGRUState) -> CausalMDGRUState:
    return CausalMDGRUState(state.hidden.detach())
