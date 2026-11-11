"""Stateful 20-fps RD/point encoder producing four tokens at 5 Hz."""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor, nn

from continuous_mmwave.config import ModelConfig


class RDFrameEncoder(nn.Module):
    """单帧 RD 图编码器：四层 Conv-BN-GELU-AvgPool 后全局池化。

    64×64 输入经 4 次下采样后为 4×4，再由 AdaptiveAvgPool2d 归约为 [B, output_dim]。
    """

    def __init__(self, input_channels: int = 2, output_dim: int = 256) -> None:
        super().__init__()
        channels = (input_channels, 32, 64, 128, output_dim)
        blocks = []
        for cin, cout in pairwise(channels):
            blocks.extend((nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.GELU(), nn.AvgPool2d(2)))
        self.network = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1))

    def forward(self, frames: Tensor) -> Tensor:
        """[B,2,64,64] -> [B,output_dim]。"""
        if frames.ndim != 4:
            raise ValueError("RD frames must be [B,2,H,W]")
        return self.network(frames).flatten(1)


class PointNetFrameEncoder(nn.Module):
    """单帧点云编码器：逐点 MLP + 沿点数维的最大池化（PointNet 对称函数）。"""

    def __init__(self, point_dim: int = 6, output_dim: int = 256) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(point_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, output_dim),
            nn.GELU(),
        )

    def forward(self, points: Tensor, mask: Tensor) -> Tensor:
        """[B,N,6] + [B,N] -> [B,output_dim]。"""
        if points.ndim != 3 or mask.shape != points.shape[:2]:
            raise ValueError("points/mask must be [B,N,6]/[B,N]")
        # 填充点置为 -inf，max 后必然不会胜出；若整帧全为填充则单独归零，
        # 避免 -inf 传播到后续层产生 NaN。
        hidden = self.mlp(points).masked_fill(~mask.bool().unsqueeze(-1), -torch.inf)
        pooled = hidden.max(dim=1).values
        return torch.where(mask.any(1, keepdim=True), pooled, torch.zeros_like(pooled))


@dataclass(frozen=True)
class TCNState:
    """各 TCN 块缓冲的历史帧；流式推理时需与模块一同携带。"""

    caches: tuple[Tensor, ...]


class CausalTCNBlock(nn.Module):
    """因果 depthwise 卷积残差块：只看当前帧与左侧 ``context`` 帧。"""

    def __init__(self, dim: int, kernel: int, dilation: int, dropout: float) -> None:
        super().__init__()
        # 因果卷积在左边需要的历史长度（右侧不 padding）。
        self.context = dilation * (kernel - 1)
        self.depthwise = nn.Conv1d(dim, dim, kernel, dilation=dilation, groups=dim)
        self.pointwise = nn.Conv1d(dim, dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, sequence: Tensor, cache: Tensor | None) -> tuple[Tensor, Tensor]:
        """[B,T,dim] + 历史缓存 -> (输出, 新的历史缓存)。"""
        temporal = sequence.transpose(1, 2)
        if cache is None:
            # 序列起始处以零填充左侧上下文。
            cache = temporal.new_zeros((temporal.shape[0], temporal.shape[1], self.context))
        if cache.shape != (temporal.shape[0], temporal.shape[1], self.context):
            raise ValueError("invalid TCN cache")
        extended = torch.cat((cache, temporal), dim=-1)
        update = self.pointwise(self.depthwise(extended))
        output = temporal + self.dropout(torch.nn.functional.gelu(update))
        # 只保留最后 context 帧作为下一段的历史，与全序列前向结果严格等价
        # （见 test_tcn_chunked_equals_full_sequence）。
        return output.transpose(1, 2), extended[:, :, -self.context :]


class StatefulCausalTCN(nn.Module):
    """多层 dilation 堆叠的因果 TCN，可分段调用而不改变结果。"""

    def __init__(self, dim: int, kernel: int, dilations: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CausalTCNBlock(dim, kernel, d, dropout) for d in dilations])

    def forward(self, sequence: Tensor, state: TCNState | None = None) -> tuple[Tensor, TCNState]:
        caches = (None,) * len(self.blocks) if state is None else state.caches
        if len(caches) != len(self.blocks):
            raise ValueError("invalid TCN state")
        next_caches = []
        for block, cache in zip(self.blocks, caches, strict=True):
            sequence, cache = block(sequence, cache)
            next_caches.append(cache)
        return sequence, TCNState(tuple(next_caches))


class QueryPooling(nn.Module):
    """用可学习的少量 query 对序列做 cross-attention 池化。"""

    def __init__(self, count: int, dim: int, heads: int) -> None:
        super().__init__()
        self.queries = nn.Parameter(torch.empty(count, dim))
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        nn.init.normal_(self.queries, std=0.02)

    def forward(self, sequence: Tensor, mask: Tensor) -> Tensor:
        """[B,T,dim] + [B,T] -> [B,count,dim]。"""
        valid = mask.any(1)
        # 全是填充的样本（纯 padding 序列）无法做 softmax，临时开放一个 key 占位，
        # 结果随后被清零，保证输出有限。
        safe = mask.bool().clone()
        safe[~valid, 0] = True
        queries = self.queries.unsqueeze(0).expand(sequence.shape[0], -1, -1)
        pooled, _ = self.attention(queries, sequence, sequence, key_padding_mask=~safe, need_weights=False)
        return torch.where(valid[:, None, None], pooled, torch.zeros_like(pooled))


@dataclass(frozen=True)
class RadarStreamState:
    """跨 update 持续传递的流式状态：两条 TCN 的缓存与 update 序号。"""

    rd_tcn: TCNState
    point_tcn: TCNState
    update_index: int


@dataclass(frozen=True)
class RadarEncoderOutput:
    """radar_tokens 供 Event Former 使用；frame_hidden/mask 供 Stage 1 姿态监督使用。"""

    radar_tokens: Tensor  # [B,U*4,radar_dim]，U 为 update 数
    rd_tokens: Tensor  # [B,U*2,radar_dim]
    point_tokens: Tensor  # [B,U*2,radar_dim]
    frame_hidden: Tensor  # [B,T,radar_dim]，逐帧融合特征
    state: RadarStreamState
    radar_token_mask: Tensor  # [B,U*4]
    frame_mask: Tensor  # [B,T]


def _time_encoding(index: int, dim: int, reference: Tensor) -> Tensor:
    """正弦位置编码，参数为连续的 update 序号（不限于序列内位置）。

    区别于常规按序列内位置编码：这里直接用全局 update 序号，使流式推理时
    位置信息不会因分段而错位。
    """
    position = reference.new_tensor(float(index))
    frequencies = torch.exp(
        torch.arange(0, dim, 2, device=reference.device, dtype=reference.dtype) * (-math.log(10_000.0) / max(dim, 1))
    )
    output = reference.new_zeros(dim)
    output[0::2] = torch.sin(position * frequencies)
    output[1::2] = torch.cos(position * frequencies[: output[1::2].shape[0]])
    return output


class StreamingRadarEncoder(nn.Module):
    """20 fps RD/点云流式编码器，每 4 帧产出一个 update 的 4 个 radar token。

    管道：帧级编码器 -> 分支独立的因果 TCN -> QueryPooling 得到 [R1,R2,P1,P2]
    -> 加模态嵌入与时间编码 -> 两层 Transformer 融合。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.rd_frame_encoder = RDFrameEncoder(config.rd_channels, config.radar_dim)
        self.point_frame_encoder = PointNetFrameEncoder(config.point_dim, config.radar_dim)
        args = (config.radar_dim, config.tcn_kernel_size, config.tcn_dilations, config.dropout)
        self.rd_tcn = StatefulCausalTCN(*args)
        self.point_tcn = StatefulCausalTCN(*args)
        self.rd_pool = QueryPooling(config.rd_tokens, config.radar_dim, config.fusion_heads)
        self.point_pool = QueryPooling(config.point_tokens, config.radar_dim, config.fusion_heads)
        self.frame_fusion = nn.Linear(config.radar_dim * 2, config.radar_dim)
        self.modality_embedding = nn.Parameter(torch.empty(2, config.radar_dim))
        layer = nn.TransformerEncoderLayer(
            config.radar_dim,
            config.fusion_heads,
            config.radar_dim * 4,
            config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, config.fusion_layers)
        nn.init.normal_(self.modality_embedding, std=0.02)

    def forward_update(
        self,
        rd: Tensor,
        points: Tensor,
        point_mask: Tensor,
        state: RadarStreamState | None = None,
        frame_mask: Tensor | None = None,
    ) -> RadarEncoderOutput:
        """处理一个 update（4 帧）。

        rd:         [B,4,2,64,64]
        points:     [B,4,N,6]
        point_mask: [B,4,N]
        frame_mask: [B,4]，None 表示全部有效
        """
        frames = self.config.frames_per_update
        if (
            rd.shape[1:] != (frames, 2, 64, 64)
            or points.ndim != 4
            or points.shape[:2] != rd.shape[:2]
            or points.shape[-1] != 6
        ):
            raise ValueError("an update must be RD[B,4,2,64,64] and points[B,4,N,6]")
        if point_mask.shape != points.shape[:3]:
            raise ValueError("point_mask must be [B,4,N]")
        if frame_mask is None:
            frame_mask = torch.ones(rd.shape[:2], dtype=torch.bool, device=rd.device)
        if frame_mask.shape != rd.shape[:2]:
            raise ValueError("frame_mask must be [B,4]")
        frame_mask = frame_mask.bool()
        # 被 padding 的帧其内部点也必然无效，先取交以避免后续 mask 语义分歧。
        point_mask = point_mask.bool() & frame_mask.unsqueeze(-1)
        batch = rd.shape[0]
        # 序号从 state 中延续，保证流式推理时位置编码不回退。
        index = 0 if state is None else state.update_index
        # 把 [B,4,...] 展平成 [B*4,...] 一次性过帧级编码器，再恢复时间维。
        rd_frames = self.rd_frame_encoder(rd.flatten(0, 1)).reshape(batch, frames, -1)
        point_frames = self.point_frame_encoder(points.flatten(0, 1), point_mask.flatten(0, 1)).reshape(
            batch, frames, -1
        )
        # 前置清零填充帧，使 TCN 缓存不被 pad 内容污染。
        rd_frames = rd_frames * frame_mask.unsqueeze(-1)
        point_frames = point_frames * frame_mask.unsqueeze(-1)
        rd_hidden, rd_state = self.rd_tcn(rd_frames, None if state is None else state.rd_tcn)
        point_hidden, point_state = self.point_tcn(point_frames, None if state is None else state.point_tcn)
        frame_hidden = self.frame_fusion(torch.cat((rd_hidden, point_hidden), -1)) * frame_mask.unsqueeze(-1)
        # 模态内各取 2 个 query，拼接后即为 [R1,R2,P1,P2]。
        rd_tokens = self.rd_pool(rd_hidden, frame_mask)
        point_tokens = self.point_pool(point_hidden, frame_mask)
        tokens = torch.cat((rd_tokens, point_tokens), 1)
        modality = torch.cat(
            (
                self.modality_embedding[0].expand(self.config.rd_tokens, -1),
                self.modality_embedding[1].expand(self.config.point_tokens, -1),
            )
        )
        fused = self.fusion(
            tokens + modality.unsqueeze(0) + _time_encoding(index, self.config.radar_dim, tokens).view(1, 1, -1)
        )
        # 整个 update 的 4 个 token 共用一个有效性标记：任一帧有效即保留。
        token_mask = frame_mask.any(1, keepdim=True).expand(-1, self.config.radar_tokens_per_update)
        fused = torch.where(token_mask.unsqueeze(-1), fused, torch.zeros_like(fused))
        next_state = RadarStreamState(rd_state, point_state, index + 1)
        return RadarEncoderOutput(fused, rd_tokens, point_tokens, frame_hidden, next_state, token_mask, frame_mask)

    def forward_sequence(
        self,
        rd: Tensor,
        points: Tensor,
        point_mask: Tensor,
        frame_mask: Tensor | None = None,
    ) -> RadarEncoderOutput:
        """训练时把整段序列按 update 切分循环，相邻 chunk 共享 state，
        从而与真正流式推理得到完全一致的输出。

        rd:  [B,T,2,64,64]，T 为 4 的非零倍数
        """
        step = self.config.frames_per_update
        if rd.ndim != 5 or rd.shape[1] == 0 or rd.shape[1] % step:
            raise ValueError("sequence time must be a non-zero multiple of four")
        if frame_mask is None:
            frame_mask = torch.ones(rd.shape[:2], dtype=torch.bool, device=rd.device)
        state = None
        outputs = []
        for start in range(0, rd.shape[1], step):
            output = self.forward_update(
                rd[:, start : start + step],
                points[:, start : start + step],
                point_mask[:, start : start + step],
                state,
                frame_mask[:, start : start + step],
            )
            outputs.append(output)
            state = output.state
        # 沿时间维拼接，token 顺序为 update 内 [R1,R2,P1,P2] 的重复。
        return RadarEncoderOutput(
            torch.cat([o.radar_tokens for o in outputs], 1),
            torch.cat([o.rd_tokens for o in outputs], 1),
            torch.cat([o.point_tokens for o in outputs], 1),
            torch.cat([o.frame_hidden for o in outputs], 1),
            outputs[-1].state,
            torch.cat([o.radar_token_mask for o in outputs], 1),
            torch.cat([o.frame_mask for o in outputs], 1),
        )
