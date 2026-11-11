"""Causal, stateful RadMamba-style encoder for PARrad micro-Doppler streams."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional


def _right_padded(mask: Tensor) -> bool:
    return not bool((mask[:, 1:] & ~mask[:, :-1]).any())


def _advance_cache(cache: Tensor, values: Tensor, mask: Tensor) -> Tensor:
    """Append valid right-padded timesteps without allowing padding into state."""
    cache_width = cache.shape[-1]
    next_cache = []
    for batch_index, count in enumerate(mask.long().sum(1).tolist()):
        combined = torch.cat((cache[batch_index], values[batch_index, ..., :count]), dim=-1)
        next_cache.append(combined[..., -cache_width:])
    return torch.stack(next_cache, dim=0)


def _position_encoding(positions: Tensor, dim: int, reference: Tensor) -> Tensor:
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=reference.device, dtype=reference.dtype)
        * (-math.log(10_000.0) / max(dim, 1))
    )
    scaled = positions.to(reference.dtype).unsqueeze(-1) * frequencies
    output = reference.new_zeros((*positions.shape, dim))
    output[..., 0::2] = torch.sin(scaled)
    output[..., 1::2] = torch.cos(scaled[..., : output[..., 1::2].shape[-1]])
    return output


@dataclass(frozen=True)
class CPMambaState:
    conv_cache: Tensor
    ssm_state: Tensor


@dataclass(frozen=True)
class CausalRadMambaState:
    frontend_cache: Tensor
    layers: tuple[CPMambaState, ...]
    position: Tensor


@dataclass(frozen=True)
class CausalRadMambaOutput:
    frame_hidden: Tensor
    token_hidden: Tensor
    state: CausalRadMambaState
    frame_mask: Tensor


class SelectiveStateSpace(nn.Module):
    """Small selective SSM with an explicit recurrent state for online inference."""

    def __init__(self, dim: int, d_state: int, dt_rank: int) -> None:
        super().__init__()
        self.dim = dim
        self.d_state = d_state
        self.delta_bc = nn.Linear(dim, dt_rank + 2 * d_state, bias=False)
        self.dt_projection = nn.Linear(dt_rank, dim)
        self.a_log = nn.Parameter(
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32)).repeat(dim, 1)
        )
        self.skip = nn.Parameter(torch.ones(dim))

    def forward(self, sequence: Tensor, mask: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        batch, frames, dim = sequence.shape
        if state is None:
            state = sequence.new_zeros((batch, dim, self.d_state))
        if state.shape != (batch, dim, self.d_state):
            raise ValueError("invalid RadMamba SSM state")

        delta_bc = self.delta_bc(sequence)
        delta_rank = self.dt_projection.in_features
        delta, input_matrix, output_matrix = torch.split(
            delta_bc, (delta_rank, self.d_state, self.d_state), dim=-1
        )
        delta = functional.softplus(self.dt_projection(delta))
        transition = -torch.exp(self.a_log.float()).to(sequence.dtype)
        outputs = []
        hidden = state
        for index in range(frames):
            step = delta[:, index]
            decay = torch.exp(step.unsqueeze(-1) * transition)
            drive = step.unsqueeze(-1) * input_matrix[:, index].unsqueeze(1) * sequence[:, index].unsqueeze(-1)
            candidate = decay * hidden + drive
            valid = mask[:, index].view(batch, 1, 1)
            hidden = torch.where(valid, candidate, hidden)
            emitted = (hidden * output_matrix[:, index].unsqueeze(1)).sum(-1)
            emitted = emitted + self.skip.to(sequence.dtype) * sequence[:, index]
            outputs.append(torch.where(mask[:, index].unsqueeze(1), emitted, torch.zeros_like(emitted)))
        return torch.stack(outputs, dim=1), hidden


class CausalCPMambaBlock(nn.Module):
    """One forward-only CP-Mamba block with cached causal convolution context."""

    def __init__(self, dim: int, d_state: int, dt_rank: int, dropout: float) -> None:
        super().__init__()
        self.dim = dim
        self.cache_width = 2
        self.input_norm = nn.LayerNorm(dim)
        self.gate_projection = nn.Conv1d(dim, dim, kernel_size=3)
        self.input_projection = nn.Conv1d(dim, dim, kernel_size=3)
        self.direction_projection = nn.Conv1d(dim, dim, kernel_size=1)
        self.direction_norm = nn.LayerNorm(dim)
        self.ssm = SelectiveStateSpace(dim, d_state, dt_rank)
        self.output_projection = nn.Conv1d(dim, dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        sequence: Tensor,
        mask: Tensor,
        state: CPMambaState | None = None,
    ) -> tuple[Tensor, CPMambaState]:
        batch = sequence.shape[0]
        normalized = self.input_norm(sequence)
        channels = normalized.transpose(1, 2)
        cache = (
            channels.new_zeros((batch, self.dim, self.cache_width))
            if state is None
            else state.conv_cache
        )
        if cache.shape != (batch, self.dim, self.cache_width):
            raise ValueError("invalid RadMamba convolution state")
        context = torch.cat((cache, channels), dim=-1)
        gate = functional.silu(self.gate_projection(context).transpose(1, 2))
        projected = self.input_projection(context)
        projected = self.direction_projection(projected).transpose(1, 2)
        projected = self.direction_norm(projected)
        ssm_output, ssm_state = self.ssm(
            projected,
            mask,
            None if state is None else state.ssm_state,
        )
        mixed = self.output_projection((ssm_output * gate).transpose(1, 2)).transpose(1, 2)
        output = sequence + self.dropout(mixed)
        output = output * mask.unsqueeze(-1)
        next_cache = _advance_cache(cache, channels, mask)
        return output, CPMambaState(next_cache, ssm_state)


class CausalRadMambaEncoder(nn.Module):
    """Encode ``[batch,time,doppler]`` MD streams without future leakage."""

    def __init__(
        self,
        md_bins: int,
        model_dim: int,
        output_dim: int,
        d_state: int,
        dt_rank: int,
        depth: int = 1,
        channel_dim: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        values = (md_bins, model_dim, output_dim, d_state, dt_rank, depth, channel_dim)
        if any(value <= 0 for value in values):
            raise ValueError("RadMamba dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.md_bins = md_bins
        self.model_dim = model_dim
        self.channel_dim = channel_dim
        self.frontend_cache_width = 2
        self.input_norm = nn.LayerNorm(md_bins)
        self.frontend_conv = nn.Conv2d(1, channel_dim, kernel_size=(3, 3), padding=(1, 0), bias=False)
        self.frontend_norm = nn.Identity() if channel_dim == 1 else nn.LayerNorm(channel_dim)
        self.patch_norm = nn.LayerNorm(channel_dim * (md_bins // 2))
        self.patch_embedding = nn.Linear(channel_dim * (md_bins // 2), model_dim)
        self.embedding_norm = nn.LayerNorm(model_dim)
        self.embedding_dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            CausalCPMambaBlock(model_dim, d_state, dt_rank, dropout) for _ in range(depth)
        )
        self.output_projection = nn.Linear(model_dim, output_dim)

    def forward(
        self,
        md: Tensor,
        frame_mask: Tensor | None = None,
        state: CausalRadMambaState | None = None,
    ) -> CausalRadMambaOutput:
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
            raise ValueError("RadMamba frame_mask must be right padded")

        normalized_md = self.input_norm(md)
        frontend_input = normalized_md.masked_fill(~frame_mask.unsqueeze(-1), 0).transpose(1, 2).unsqueeze(1)
        frontend_cache = (
            frontend_input.new_zeros((batch, 1, self.md_bins, self.frontend_cache_width))
            if state is None
            else state.frontend_cache
        )
        if frontend_cache.shape != (batch, 1, self.md_bins, self.frontend_cache_width):
            raise ValueError("invalid RadMamba frontend state")
        frontend_context = torch.cat((frontend_cache, frontend_input), dim=-1)
        feature = self.frontend_conv(frontend_context)
        feature = self.frontend_norm(feature.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feature = functional.gelu(feature)
        feature = functional.max_pool2d(feature, kernel_size=(2, 1), stride=(2, 1))
        feature = feature * frame_mask[:, None, None, :]
        next_frontend_cache = _advance_cache(frontend_cache, frontend_input, frame_mask)

        tokens = feature.permute(0, 3, 1, 2).reshape(batch, frames, -1)
        tokens = self.embedding_norm(self.patch_embedding(self.patch_norm(tokens)))
        starting_position = (
            torch.zeros(batch, dtype=torch.long, device=md.device) if state is None else state.position
        )
        if starting_position.shape != (batch,):
            raise ValueError("invalid RadMamba position state")
        offsets = frame_mask.long().cumsum(1) - 1
        positions = starting_position.unsqueeze(1) + offsets.clamp_min(0)
        tokens = tokens + _position_encoding(positions, self.model_dim, tokens)
        tokens = self.embedding_dropout(tokens) * frame_mask.unsqueeze(-1)

        next_layers = []
        for index, layer in enumerate(self.layers):
            layer_state = None if state is None else state.layers[index]
            tokens, layer_state = layer(tokens, frame_mask, layer_state)
            next_layers.append(layer_state)
        frame_hidden = self.output_projection(tokens) * frame_mask.unsqueeze(-1)
        next_state = CausalRadMambaState(
            frontend_cache=next_frontend_cache,
            layers=tuple(next_layers),
            position=starting_position + frame_mask.long().sum(1),
        )
        return CausalRadMambaOutput(frame_hidden, tokens, next_state, frame_mask)


def detach_causal_radmamba_state(state: CausalRadMambaState) -> CausalRadMambaState:
    """Detach all recurrent tensors at a truncated-BPTT boundary."""
    return CausalRadMambaState(
        frontend_cache=state.frontend_cache.detach(),
        layers=tuple(
            CPMambaState(layer.conv_cache.detach(), layer.ssm_state.detach()) for layer in state.layers
        ),
        position=state.position,
    )
