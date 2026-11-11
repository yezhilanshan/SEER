"""MD-guided streaming RD encoder for the PARrad validation variant."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from continuous_mmwave.models.md_gru_encoder import (
    CausalMDGRUEncoder,
    CausalMDGRUState,
    detach_causal_md_gru_state,
)
from continuous_mmwave.models.radar_encoder import QueryPooling, StatefulCausalTCN, TCNState
from continuous_mmwave.models.radmamba_encoder import (
    CausalRadMambaEncoder,
    CausalRadMambaState,
    detach_causal_radmamba_state,
)
from continuous_mmwave.parrad_config import PARradConfig


class ResidualRDEncoderBlock(nn.Module):
    """Two-convolution residual block with possibly asymmetric downsampling."""

    def __init__(self, input_channels: int, output_channels: int, stride: tuple[int, int]) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
            nn.GELU(),
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(output_channels),
        )
        self.skip = (
            nn.Identity()
            if input_channels == output_channels and stride == (1, 1)
            else nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(output_channels),
            )
        )
        self.activation = nn.GELU()

    def forward(self, frames: Tensor) -> Tensor:
        return self.activation(self.main(frames) + self.skip(frames))


class RDSpatialEncoder(nn.Module):
    """Encode RD maps while delaying Doppler-axis downsampling."""

    def __init__(self, config: PARradConfig) -> None:
        super().__init__()
        first = max(config.radar_dim // 8, 8)
        second = max(config.radar_dim // 4, 8)
        third = max(config.radar_dim // 2, 8)
        self.stem = nn.Sequential(
            nn.Conv2d(config.rd_channels, first, 3, stride=(2, 1), padding=1, bias=False),
            nn.BatchNorm2d(first),
            nn.GELU(),
        )
        self.block1 = ResidualRDEncoderBlock(first, second, (2, 1))
        self.block2 = ResidualRDEncoderBlock(second, third, (2, 2))
        self.block3 = ResidualRDEncoderBlock(third, config.radar_dim, (1, 1))

    def forward(self, frames: Tensor) -> Tensor:
        if frames.ndim != 4:
            raise ValueError("RD frames must be [B,C,R,D]")
        return self.block3(self.block2(self.block1(self.stem(frames))))


class MDFrameEncoder(nn.Module):
    """Lightweight per-frame Micro-Doppler encoder."""

    def __init__(self, motion_dim: int) -> None:
        super().__init__()
        first = max(motion_dim // 4, 8)
        second = max(motion_dim // 2, 8)
        self.network = nn.Sequential(
            nn.Conv1d(1, first, 5, padding=2, bias=False),
            nn.BatchNorm1d(first),
            nn.GELU(),
            nn.Conv1d(first, second, 3, padding=1, bias=False),
            nn.BatchNorm1d(second),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.output = nn.Linear(second, motion_dim)

    def forward(self, signature: Tensor) -> Tensor:
        if signature.ndim != 2:
            raise ValueError("MD frames must be [B,D]")
        return self.output(self.network(signature.unsqueeze(1)).flatten(1))


class MotionFiLM(nn.Module):
    """Channel-wise identity-initialized motion modulation."""

    def __init__(self, motion_dim: int, channels: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(motion_dim, channels)
        self.beta = nn.Linear(motion_dim, channels)
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight)
        nn.init.zeros_(self.beta.bias)

    def forward(self, feature: Tensor, motion: Tensor) -> Tensor:
        if feature.ndim != 4 or motion.ndim != 2 or feature.shape[0] != motion.shape[0]:
            raise ValueError("FiLM expects feature[B,C,R,D] and motion[B,M]")
        gamma = self.gamma(motion).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(motion).unsqueeze(-1).unsqueeze(-1)
        return (1 + gamma) * feature + beta


@dataclass(frozen=True)
class PARradStreamState:
    rd_tcn: TCNState
    md_tcn: TCNState
    update_index: int


@dataclass(frozen=True)
class PARradRadMambaStreamState:
    radmamba: CausalRadMambaState
    update_index: int


@dataclass(frozen=True)
class PARradMDGRUStreamState:
    gru: CausalMDGRUState
    update_index: int


PARradAnyStreamState = PARradStreamState | PARradRadMambaStreamState | PARradMDGRUStreamState


def detach_parrad_stream_state(state: PARradAnyStreamState) -> PARradAnyStreamState:
    """Detach cached history at a TBPTT boundary without resetting stream time."""
    if isinstance(state, PARradRadMambaStreamState):
        return PARradRadMambaStreamState(
            radmamba=detach_causal_radmamba_state(state.radmamba),
            update_index=state.update_index,
        )
    if isinstance(state, PARradMDGRUStreamState):
        return PARradMDGRUStreamState(
            gru=detach_causal_md_gru_state(state.gru),
            update_index=state.update_index,
        )
    return PARradStreamState(
        rd_tcn=TCNState(tuple(cache.detach() for cache in state.rd_tcn.caches)),
        md_tcn=TCNState(tuple(cache.detach() for cache in state.md_tcn.caches)),
        update_index=state.update_index,
    )


@dataclass(frozen=True)
class PARradEncoderOutput:
    radar_tokens: Tensor
    semantic_state: Tensor
    motion_state: Tensor
    frame_hidden: Tensor
    state: PARradAnyStreamState
    radar_token_mask: Tensor
    frame_mask: Tensor


def _position_encoding(positions: Tensor, dim: int, reference: Tensor) -> Tensor:
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=reference.device, dtype=reference.dtype)
        * (-math.log(10_000.0) / max(dim, 1))
    )
    output = reference.new_zeros((positions.numel(), dim))
    scaled = positions.to(device=reference.device, dtype=reference.dtype).unsqueeze(1) * frequencies.unsqueeze(0)
    output[:, 0::2] = torch.sin(scaled)
    output[:, 1::2] = torch.cos(scaled[:, : output[:, 1::2].shape[1]])
    return output


def _last_valid(sequence: Tensor, mask: Tensor) -> Tensor:
    counts = mask.long().sum(1)
    indices = (counts - 1).clamp_min(0)
    gathered = sequence[torch.arange(sequence.shape[0], device=sequence.device), indices]
    return torch.where(counts.unsqueeze(1) > 0, gathered, torch.zeros_like(gathered))


class PARradRadarEncoder(nn.Module):
    """Map aligned PARrad RD/MD streams to the existing four-token contract."""

    def __init__(self, config: PARradConfig) -> None:
        super().__init__()
        self.config = config
        self.rd_spatial = RDSpatialEncoder(config)
        self.md_frame = MDFrameEncoder(config.motion_dim)
        self.md_tcn = StatefulCausalTCN(
            config.motion_dim, 3, config.md_tcn_dilations, config.dropout
        )
        self.film = MotionFiLM(config.motion_dim, config.radar_dim)
        self.doppler_projection = nn.Linear(config.radar_dim, config.radar_dim)
        self.frame_pool = QueryPooling(1, config.radar_dim, config.fusion_heads)
        self.rd_tcn = StatefulCausalTCN(
            config.radar_dim, 3, config.rd_tcn_dilations, config.dropout
        )
        self.temporal_projection = nn.Linear(config.radar_dim, config.radar_dim)
        self.temporal_gate = nn.Parameter(torch.zeros(()))
        self.token_resampler = QueryPooling(
            config.radar_tokens_per_update, config.radar_dim, config.fusion_heads
        )
        self.semantic_pool = QueryPooling(1, config.radar_dim, config.fusion_heads)
        self.sensor_embedding = nn.Embedding(config.sensor_vocab_size, config.radar_dim)
        nn.init.normal_(self.sensor_embedding.weight, std=0.02)

    def _validate_update(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None,
        frame_mask: Tensor | None,
    ) -> Tensor:
        expected_rd = (
            self.config.frames_per_update,
            self.config.rd_channels,
            self.config.rd_height,
            self.config.rd_width,
        )
        if rd.ndim != 5 or rd.shape[1:] != expected_rd:
            raise ValueError(f"RD update must be [B,{','.join(str(v) for v in expected_rd)}]")
        if md.shape != (rd.shape[0], self.config.frames_per_update, self.config.md_bins):
            raise ValueError("MD update shape does not match the PARrad configuration")
        if sensor_index is not None:
            if sensor_index.shape != (rd.shape[0],) or sensor_index.dtype != torch.long:
                raise ValueError("sensor_index must be int64 [B]")
            if (sensor_index < 0).any() or (sensor_index >= self.config.sensor_vocab_size).any():
                raise ValueError("sensor_index is outside the configured vocabulary")
        if frame_mask is None:
            frame_mask = torch.ones(rd.shape[:2], dtype=torch.bool, device=rd.device)
        if frame_mask.shape != rd.shape[:2]:
            raise ValueError("frame_mask must be [B,T]")
        if not torch.isfinite(rd).all() or not torch.isfinite(md).all():
            raise ValueError("RD and MD inputs must be finite")
        return frame_mask.bool()

    def forward_update(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None = None,
        state: PARradStreamState | None = None,
        frame_mask: Tensor | None = None,
    ) -> PARradEncoderOutput:
        frame_mask = self._validate_update(rd, md, sensor_index, frame_mask)
        batch, frames = rd.shape[:2]
        update_index = 0 if state is None else state.update_index
        flat_mask = frame_mask.flatten()

        if not flat_mask.any():
            md_input = md.new_zeros((batch, frames, self.config.motion_dim))
            md_hidden, md_state = self.md_tcn(md_input, None if state is None else state.md_tcn)
            frame_hidden = rd.new_zeros((batch, frames, self.config.radar_dim))
            _, rd_state = self.rd_tcn(frame_hidden, None if state is None else state.rd_tcn)
            radar_token_mask = torch.zeros(
                (batch, self.config.radar_tokens_per_update),
                dtype=torch.bool,
                device=rd.device,
            )
            radar_tokens = rd.new_zeros(
                (batch, self.config.radar_tokens_per_update, self.config.radar_dim)
            )
            return PARradEncoderOutput(
                radar_tokens=radar_tokens,
                semantic_state=rd.new_zeros((batch, self.config.radar_dim)),
                motion_state=rd.new_zeros((batch, self.config.motion_dim)),
                frame_hidden=frame_hidden,
                state=PARradStreamState(rd_state, md_state, update_index + 1),
                radar_token_mask=radar_token_mask,
                frame_mask=frame_mask,
            )

        valid_rd_maps = self.rd_spatial(rd.flatten(0, 1)[flat_mask])
        rd_maps = valid_rd_maps.new_zeros((batch * frames, *valid_rd_maps.shape[1:]))
        rd_maps[flat_mask] = valid_rd_maps
        valid_md_frames = self.md_frame(md.flatten(0, 1)[flat_mask])
        md_frames = valid_md_frames.new_zeros((batch * frames, valid_md_frames.shape[-1]))
        md_frames[flat_mask] = valid_md_frames
        md_frames = md_frames.reshape(batch, frames, -1)
        md_hidden, md_state = self.md_tcn(md_frames, None if state is None else state.md_tcn)
        md_hidden = md_hidden * frame_mask.unsqueeze(-1)

        enhanced = self.film(rd_maps, md_hidden.flatten(0, 1))
        enhanced = enhanced * flat_mask[:, None, None, None]
        range_pooled = enhanced.mean(dim=-2).reshape(batch, frames, self.config.radar_dim, -1)
        doppler_tokens = self.doppler_projection(range_pooled.permute(0, 1, 3, 2))
        doppler_bins = doppler_tokens.shape[2]
        doppler_position = _position_encoding(
            torch.arange(doppler_bins, device=rd.device), self.config.radar_dim, doppler_tokens
        )
        doppler_tokens = doppler_tokens + doppler_position.view(1, 1, doppler_bins, -1)

        doppler_mask = frame_mask.flatten().unsqueeze(1).expand(-1, doppler_bins)
        frame_hidden = self.frame_pool(
            doppler_tokens.flatten(0, 1), doppler_mask
        ).squeeze(1).reshape(batch, frames, -1)
        frame_hidden = frame_hidden * frame_mask.unsqueeze(-1)
        temporal, rd_state = self.rd_tcn(frame_hidden, None if state is None else state.rd_tcn)
        temporal = temporal * frame_mask.unsqueeze(-1)
        time_positions = torch.arange(
            update_index * frames, (update_index + 1) * frames, device=rd.device
        )
        temporal = temporal + _position_encoding(time_positions, self.config.radar_dim, temporal).unsqueeze(0)
        temporal = temporal * frame_mask.unsqueeze(-1)
        doppler_tokens = doppler_tokens + torch.tanh(self.temporal_gate) * self.temporal_projection(temporal).unsqueeze(2)
        if sensor_index is not None and self.config.use_sensor_embedding:
            doppler_tokens = doppler_tokens + self.sensor_embedding(sensor_index).view(batch, 1, 1, -1)

        source = doppler_tokens.reshape(batch, frames * doppler_bins, self.config.radar_dim)
        source_mask = frame_mask.unsqueeze(-1).expand(-1, -1, doppler_bins).reshape(batch, -1)
        radar_tokens = self.token_resampler(source, source_mask)
        update_valid = frame_mask.any(1)
        radar_token_mask = update_valid.unsqueeze(1).expand(-1, self.config.radar_tokens_per_update)
        radar_tokens = torch.where(radar_token_mask.unsqueeze(-1), radar_tokens, torch.zeros_like(radar_tokens))
        semantic_state = self.semantic_pool(radar_tokens, radar_token_mask).squeeze(1)
        motion_state = _last_valid(md_hidden, frame_mask)
        return PARradEncoderOutput(
            radar_tokens=radar_tokens,
            semantic_state=semantic_state,
            motion_state=motion_state,
            frame_hidden=frame_hidden,
            state=PARradStreamState(rd_state, md_state, update_index + 1),
            radar_token_mask=radar_token_mask,
            frame_mask=frame_mask,
        )

    def forward_sequence(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None = None,
        frame_mask: Tensor | None = None,
        state: PARradStreamState | None = None,
    ) -> PARradEncoderOutput:
        step = self.config.frames_per_update
        if rd.ndim != 5 or rd.shape[1] == 0 or rd.shape[1] % step:
            raise ValueError("PARrad sequence time must be a non-zero multiple of frames_per_update")
        if md.shape[:2] != rd.shape[:2] or md.ndim != 3:
            raise ValueError("RD and MD sequence times must match")
        if frame_mask is None:
            frame_mask = torch.ones(rd.shape[:2], dtype=torch.bool, device=rd.device)
        outputs = []
        semantic_state = rd.new_zeros((rd.shape[0], self.config.radar_dim))
        motion_state = rd.new_zeros((rd.shape[0], self.config.motion_dim))
        for start in range(0, rd.shape[1], step):
            output = self.forward_update(
                rd[:, start : start + step],
                md[:, start : start + step],
                sensor_index=sensor_index,
                state=state,
                frame_mask=frame_mask[:, start : start + step],
            )
            valid = output.frame_mask.any(1)
            semantic_state = torch.where(valid.unsqueeze(1), output.semantic_state, semantic_state)
            motion_state = torch.where(valid.unsqueeze(1), output.motion_state, motion_state)
            outputs.append(output)
            state = output.state
        return PARradEncoderOutput(
            radar_tokens=torch.cat([output.radar_tokens for output in outputs], 1),
            semantic_state=semantic_state,
            motion_state=motion_state,
            frame_hidden=torch.cat([output.frame_hidden for output in outputs], 1),
            state=outputs[-1].state,
            radar_token_mask=torch.cat([output.radar_token_mask for output in outputs], 1),
            frame_mask=torch.cat([output.frame_mask for output in outputs], 1),
        )


class PARradCausalRadMambaEncoder(nn.Module):
    """MD-only causal RadMamba backend preserving the PARrad four-token API."""

    def __init__(self, config: PARradConfig) -> None:
        super().__init__()
        self.config = config
        self.radmamba = CausalRadMambaEncoder(
            md_bins=config.md_bins,
            model_dim=config.radmamba_dim,
            output_dim=config.radmamba_dim,
            d_state=config.radmamba_d_state,
            dt_rank=config.radmamba_dt_rank,
            depth=config.radmamba_depth,
            channel_dim=config.radmamba_channels,
            dropout=config.dropout,
        )
        self.frame_projection = nn.Linear(config.radmamba_dim, config.radar_dim)
        self.motion_projection = nn.Linear(config.radmamba_dim, config.motion_dim)
        self.token_resampler = QueryPooling(
            config.radar_tokens_per_update, config.radmamba_dim, config.fusion_heads
        )
        self.token_projection = nn.Linear(config.radmamba_dim, config.radar_dim)
        self.sensor_embedding = nn.Embedding(config.sensor_vocab_size, config.radar_dim)
        nn.init.normal_(self.sensor_embedding.weight, std=0.02)

    def _validate_update(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None,
        frame_mask: Tensor | None,
    ) -> Tensor:
        expected_rd = (
            self.config.frames_per_update,
            self.config.rd_channels,
            self.config.rd_height,
            self.config.rd_width,
        )
        placeholder_rd = rd.ndim == 5 and rd.shape[1:] == (self.config.frames_per_update, 0, 0, 0)
        if rd.ndim != 5 or (rd.shape[1:] != expected_rd and not placeholder_rd):
            raise ValueError(f"RD update must be [B,{','.join(str(v) for v in expected_rd)}]")
        if md.shape != (rd.shape[0], self.config.frames_per_update, self.config.md_bins):
            raise ValueError("MD update shape does not match the PARrad configuration")
        if sensor_index is not None:
            if sensor_index.shape != (rd.shape[0],) or sensor_index.dtype != torch.long:
                raise ValueError("sensor_index must be int64 [B]")
            if (sensor_index < 0).any() or (sensor_index >= self.config.sensor_vocab_size).any():
                raise ValueError("sensor_index is outside the configured vocabulary")
        if frame_mask is None:
            frame_mask = torch.ones(rd.shape[:2], dtype=torch.bool, device=rd.device)
        if frame_mask.shape != rd.shape[:2]:
            raise ValueError("frame_mask must be [B,T]")
        if not torch.isfinite(md).all():
            raise ValueError("MD inputs must be finite")
        return frame_mask.bool()

    def forward_update(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None = None,
        state: PARradRadMambaStreamState | None = None,
        frame_mask: Tensor | None = None,
    ) -> PARradEncoderOutput:
        frame_mask = self._validate_update(rd, md, sensor_index, frame_mask)
        if state is not None and not isinstance(state, PARradRadMambaStreamState):
            raise TypeError("causal RadMamba requires a RadMamba stream state")
        encoded = self.radmamba(md, frame_mask, None if state is None else state.radmamba)
        frame_hidden = self.frame_projection(encoded.frame_hidden)
        if sensor_index is not None and self.config.use_sensor_embedding:
            frame_hidden = frame_hidden + self.sensor_embedding(sensor_index).unsqueeze(1)
        frame_hidden = frame_hidden * frame_mask.unsqueeze(-1)
        radar_tokens = self.token_projection(self.token_resampler(encoded.frame_hidden, frame_mask))
        if sensor_index is not None and self.config.use_sensor_embedding:
            radar_tokens = radar_tokens + self.sensor_embedding(sensor_index).unsqueeze(1)
        update_valid = frame_mask.any(1)
        radar_token_mask = update_valid.unsqueeze(1).expand(-1, self.config.radar_tokens_per_update)
        radar_tokens = torch.where(
            radar_token_mask.unsqueeze(-1), radar_tokens, torch.zeros_like(radar_tokens)
        )
        semantic_state = radar_tokens.sum(1) / radar_token_mask.sum(1, keepdim=True).clamp_min(1)
        semantic_state = semantic_state * update_valid.unsqueeze(1)
        motion_frames = self.motion_projection(encoded.token_hidden) * frame_mask.unsqueeze(-1)
        motion_state = _last_valid(motion_frames, frame_mask)
        update_index = 0 if state is None else state.update_index
        return PARradEncoderOutput(
            radar_tokens=radar_tokens,
            semantic_state=semantic_state,
            motion_state=motion_state,
            frame_hidden=frame_hidden,
            state=PARradRadMambaStreamState(encoded.state, update_index + 1),
            radar_token_mask=radar_token_mask,
            frame_mask=frame_mask,
        )

    def forward_sequence(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None = None,
        frame_mask: Tensor | None = None,
        state: PARradRadMambaStreamState | None = None,
    ) -> PARradEncoderOutput:
        step = self.config.frames_per_update
        if rd.ndim != 5 or rd.shape[1] == 0 or rd.shape[1] % step:
            raise ValueError("PARrad sequence time must be a non-zero multiple of frames_per_update")
        if md.shape[:2] != rd.shape[:2] or md.ndim != 3:
            raise ValueError("RD and MD sequence times must match")
        if frame_mask is None:
            frame_mask = torch.ones(rd.shape[:2], dtype=torch.bool, device=rd.device)
        outputs = []
        semantic_state = rd.new_zeros((rd.shape[0], self.config.radar_dim))
        motion_state = rd.new_zeros((rd.shape[0], self.config.motion_dim))
        for start in range(0, rd.shape[1], step):
            output = self.forward_update(
                rd[:, start : start + step],
                md[:, start : start + step],
                sensor_index=sensor_index,
                state=state,
                frame_mask=frame_mask[:, start : start + step],
            )
            valid = output.frame_mask.any(1)
            semantic_state = torch.where(valid.unsqueeze(1), output.semantic_state, semantic_state)
            motion_state = torch.where(valid.unsqueeze(1), output.motion_state, motion_state)
            outputs.append(output)
            state = output.state
        return PARradEncoderOutput(
            radar_tokens=torch.cat([output.radar_tokens for output in outputs], 1),
            semantic_state=semantic_state,
            motion_state=motion_state,
            frame_hidden=torch.cat([output.frame_hidden for output in outputs], 1),
            state=outputs[-1].state,
            radar_token_mask=torch.cat([output.radar_token_mask for output in outputs], 1),
            frame_mask=torch.cat([output.frame_mask for output in outputs], 1),
        )


class PARradCausalMDGRUEncoder(nn.Module):
    """MD-only causal GRU baseline preserving the PARrad four-token API."""

    def __init__(self, config: PARradConfig) -> None:
        super().__init__()
        self.config = config
        self.gru = CausalMDGRUEncoder(config.md_bins, config.md_gru_dim, config.dropout)
        self.frame_projection = nn.Linear(config.md_gru_dim, config.radar_dim)
        self.motion_projection = nn.Linear(config.md_gru_dim, config.motion_dim)
        self.token_resampler = QueryPooling(
            config.radar_tokens_per_update, config.md_gru_dim, config.fusion_heads
        )
        self.token_projection = nn.Linear(config.md_gru_dim, config.radar_dim)
        self.sensor_embedding = nn.Embedding(config.sensor_vocab_size, config.radar_dim)
        nn.init.normal_(self.sensor_embedding.weight, std=0.02)

    def _validate_update(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None,
        frame_mask: Tensor | None,
    ) -> Tensor:
        expected_rd = (
            self.config.frames_per_update,
            self.config.rd_channels,
            self.config.rd_height,
            self.config.rd_width,
        )
        placeholder_rd = rd.ndim == 5 and rd.shape[1:] == (self.config.frames_per_update, 0, 0, 0)
        if rd.ndim != 5 or (rd.shape[1:] != expected_rd and not placeholder_rd):
            raise ValueError(f"RD update must be [B,{','.join(str(v) for v in expected_rd)}]")
        if md.shape != (rd.shape[0], self.config.frames_per_update, self.config.md_bins):
            raise ValueError("MD update shape does not match the PARrad configuration")
        if sensor_index is not None:
            if sensor_index.shape != (rd.shape[0],) or sensor_index.dtype != torch.long:
                raise ValueError("sensor_index must be int64 [B]")
            if (sensor_index < 0).any() or (sensor_index >= self.config.sensor_vocab_size).any():
                raise ValueError("sensor_index is outside the configured vocabulary")
        if frame_mask is None:
            frame_mask = torch.ones(rd.shape[:2], dtype=torch.bool, device=rd.device)
        if frame_mask.shape != rd.shape[:2]:
            raise ValueError("frame_mask must be [B,T]")
        if not torch.isfinite(md).all():
            raise ValueError("MD inputs must be finite")
        return frame_mask.bool()

    def forward_update(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None = None,
        state: PARradMDGRUStreamState | None = None,
        frame_mask: Tensor | None = None,
    ) -> PARradEncoderOutput:
        frame_mask = self._validate_update(rd, md, sensor_index, frame_mask)
        if state is not None and not isinstance(state, PARradMDGRUStreamState):
            raise TypeError("causal MD-GRU requires an MD-GRU stream state")
        encoded = self.gru(md, frame_mask, None if state is None else state.gru)
        frame_hidden = self.frame_projection(encoded.frame_hidden)
        if sensor_index is not None and self.config.use_sensor_embedding:
            frame_hidden = frame_hidden + self.sensor_embedding(sensor_index).unsqueeze(1)
        frame_hidden = frame_hidden * frame_mask.unsqueeze(-1)
        radar_tokens = self.token_projection(self.token_resampler(encoded.frame_hidden, frame_mask))
        if sensor_index is not None and self.config.use_sensor_embedding:
            radar_tokens = radar_tokens + self.sensor_embedding(sensor_index).unsqueeze(1)
        update_valid = frame_mask.any(1)
        radar_token_mask = update_valid.unsqueeze(1).expand(-1, self.config.radar_tokens_per_update)
        radar_tokens = torch.where(
            radar_token_mask.unsqueeze(-1), radar_tokens, torch.zeros_like(radar_tokens)
        )
        semantic_state = radar_tokens.sum(1) / radar_token_mask.sum(1, keepdim=True).clamp_min(1)
        semantic_state = semantic_state * update_valid.unsqueeze(1)
        motion_frames = self.motion_projection(encoded.frame_hidden) * frame_mask.unsqueeze(-1)
        motion_state = _last_valid(motion_frames, frame_mask)
        update_index = 0 if state is None else state.update_index
        return PARradEncoderOutput(
            radar_tokens=radar_tokens,
            semantic_state=semantic_state,
            motion_state=motion_state,
            frame_hidden=frame_hidden,
            state=PARradMDGRUStreamState(encoded.state, update_index + 1),
            radar_token_mask=radar_token_mask,
            frame_mask=frame_mask,
        )

    def forward_sequence(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor | None = None,
        frame_mask: Tensor | None = None,
        state: PARradMDGRUStreamState | None = None,
    ) -> PARradEncoderOutput:
        step = self.config.frames_per_update
        if rd.ndim != 5 or rd.shape[1] == 0 or rd.shape[1] % step:
            raise ValueError("PARrad sequence time must be a non-zero multiple of frames_per_update")
        if md.shape[:2] != rd.shape[:2] or md.ndim != 3:
            raise ValueError("RD and MD sequence times must match")
        if frame_mask is None:
            frame_mask = torch.ones(rd.shape[:2], dtype=torch.bool, device=rd.device)
        outputs = []
        semantic_state = rd.new_zeros((rd.shape[0], self.config.radar_dim))
        motion_state = rd.new_zeros((rd.shape[0], self.config.motion_dim))
        for start in range(0, rd.shape[1], step):
            output = self.forward_update(
                rd[:, start : start + step],
                md[:, start : start + step],
                sensor_index=sensor_index,
                state=state,
                frame_mask=frame_mask[:, start : start + step],
            )
            valid = output.frame_mask.any(1)
            semantic_state = torch.where(valid.unsqueeze(1), output.semantic_state, semantic_state)
            motion_state = torch.where(valid.unsqueeze(1), output.motion_state, motion_state)
            outputs.append(output)
            state = output.state
        return PARradEncoderOutput(
            radar_tokens=torch.cat([output.radar_tokens for output in outputs], 1),
            semantic_state=semantic_state,
            motion_state=motion_state,
            frame_hidden=torch.cat([output.frame_hidden for output in outputs], 1),
            state=outputs[-1].state,
            radar_token_mask=torch.cat([output.radar_token_mask for output in outputs], 1),
            frame_mask=torch.cat([output.frame_mask for output in outputs], 1),
        )


def build_parrad_encoder(
    config: PARradConfig,
) -> PARradRadarEncoder | PARradCausalRadMambaEncoder | PARradCausalMDGRUEncoder:
    """Build the explicitly configured PARrad encoder backend."""
    if config.encoder_type == "causal_radmamba":
        return PARradCausalRadMambaEncoder(config)
    if config.encoder_type == "causal_md_gru":
        return PARradCausalMDGRUEncoder(config)
    return PARradRadarEncoder(config)
