"""Frozen v2.2 defaults."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

# 固定 Event Prompt：Streaming 与 Commit 两条路径必须复用同一段文本，
# 因此在配置层冻结，任何路径都不得替换（见 StreamingSemanticEventFormer.__init__ 的校验）。
FIXED_EVENT_PROMPT = (
    "Maintain a compact semantic representation of the current human motion event from radar observations."
)


@dataclass(slots=True)
class ModelConfig:
    """v2.2 全部冻结的超参数，同时充当各模块间的维度契约。

    时间维度约定：雷达以 ``radar_fps`` 采帧、以 ``update_hz`` 输出，
    因此每次 update 消耗 ``frames_per_update`` 帧并产出 4 个 radar token。
    """

    # 时序/输入
    radar_fps: int = 20  # RD/点云原始帧率
    update_hz: int = 5  # radar token 更新频率，20 fps 下即每 4 帧一次
    rd_channels: int = 2  # RD 图通道数（距离-多普勒双分量）
    rd_height: int = 64
    rd_width: int = 64
    point_dim: int = 6  # 单点特征维度（xyz + 速度等）
    num_joints: int = 17  # Stage 1 骨架关节点数
    # Radar Encoder
    radar_dim: int = 256  # 每帧编码与融合后的隐层维度
    rd_tokens: int = 2  # 每次 update 的 RD query 数
    point_tokens: int = 2  # 每次 update 的点云 query 数
    tcn_kernel_size: int = 3
    tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)  # 因果 TCN 各层膨胀率，决定感受野
    fusion_layers: int = 2
    fusion_heads: int = 8
    # Event Former / LLM
    event_tokens: int = 8  # 事件记忆的 token 数（固定长度）
    llm_dim: int = 896  # Qwen2.5-0.5B 隐层维度
    llm_layers: int = 4  # 只取 LLM 前 N 层
    llm_heads: int = 14
    cross_attn_dim: int = 512  # 显式 bottleneck cross-attention 的内部维度
    cross_attn_heads: int = 8
    cross_attention_init_scale: float = 0.0  # tanh(alpha) 的有效初值；PARrad Stage 2 显式打开
    event_state_initial_update: float = 0.1  # 初始状态每次吸收 proposal 的比例
    semantic_dim: int = 128  # 事件/文本共享语义空间的维度
    lora_rank: int = 16
    max_commit_radar_tokens: int = 256  # Commit 路径允许的最大 radar token 数，超出则做平均池化
    dropout: float = 0.1
    fixed_event_prompt: str = FIXED_EVENT_PROMPT

    def __post_init__(self) -> None:
        # 以下校验保证后续 reshape/整除运算不会在训练中途才失败。
        if self.radar_fps <= 0 or self.update_hz <= 0 or self.radar_fps % self.update_hz:
            raise ValueError("radar_fps must be a positive multiple of update_hz")
        # [R1,R2,P1,P2] 的布局在方案书 v2.2 中被冻结为 4 个 token。
        if self.rd_tokens + self.point_tokens != 4:
            raise ValueError("v2.2 requires exactly four radar tokens per update")
        if self.radar_dim % self.fusion_heads:
            raise ValueError("radar_dim must be divisible by fusion_heads")
        if self.cross_attn_dim % self.cross_attn_heads:
            raise ValueError("cross_attn_dim must be divisible by cross_attn_heads")
        if any(d <= 0 for d in self.tcn_dilations):
            raise ValueError("TCN dilations must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 <= self.cross_attention_init_scale < 1:
            raise ValueError("cross_attention_init_scale must be in [0,1)")
        if not 0 < self.event_state_initial_update < 1:
            raise ValueError("event_state_initial_update must be in (0,1)")

    @property
    def frames_per_update(self) -> int:
        """一次 update 消耗的原始帧数（默认 20/5 = 4）。"""
        return self.radar_fps // self.update_hz

    @property
    def radar_tokens_per_update(self) -> int:
        """一次 update 产出的 radar token 数（rd_tokens + point_tokens）。"""
        return self.rd_tokens + self.point_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
