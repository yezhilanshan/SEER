"""Configuration contract for the PARrad validation front end."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from continuous_mmwave.config import FIXED_EVENT_PROMPT


@dataclass(slots=True)
class PARradConfig:
    """Frozen dimensions and timing for the PARrad validation variant."""

    frame_period_s: float = 0.09
    encoder_type: str = "rd_md_tcn"
    frames_per_update: int = 2
    rd_channels: int = 1
    rd_height: int = 93
    rd_width: int = 128
    md_bins: int = 128
    radar_dim: int = 256
    motion_dim: int = 128
    radar_tokens_per_update: int = 4
    sensor_vocab_size: int = 2
    use_sensor_embedding: bool = True
    fusion_heads: int = 8
    rd_tcn_dilations: tuple[int, ...] = (1, 2, 4, 8)
    md_tcn_dilations: tuple[int, ...] = (1, 2, 4)
    radmamba_dim: int = 32
    radmamba_d_state: int = 8
    radmamba_dt_rank: int = 4
    radmamba_depth: int = 1
    radmamba_channels: int = 1
    md_gru_dim: int = 128
    event_tokens: int = 8
    llm_dim: int = 896
    llm_layers: int = 4
    llm_heads: int = 14
    cross_attn_dim: int = 512
    cross_attn_heads: int = 8
    cross_attention_init_scale: float = 0.0
    event_state_initial_update: float = 0.1
    semantic_dim: int = 128
    lora_rank: int = 16
    max_commit_radar_tokens: int = 256
    dropout: float = 0.1
    fixed_event_prompt: str = FIXED_EVENT_PROMPT

    def __post_init__(self) -> None:
        positive = {
            "frame_period_s": self.frame_period_s,
            "frames_per_update": self.frames_per_update,
            "rd_channels": self.rd_channels,
            "rd_height": self.rd_height,
            "rd_width": self.rd_width,
            "md_bins": self.md_bins,
            "radar_dim": self.radar_dim,
            "motion_dim": self.motion_dim,
            "sensor_vocab_size": self.sensor_vocab_size,
            "fusion_heads": self.fusion_heads,
            "radmamba_dim": self.radmamba_dim,
            "radmamba_d_state": self.radmamba_d_state,
            "radmamba_dt_rank": self.radmamba_dt_rank,
            "radmamba_depth": self.radmamba_depth,
            "radmamba_channels": self.radmamba_channels,
            "md_gru_dim": self.md_gru_dim,
            "event_tokens": self.event_tokens,
            "llm_dim": self.llm_dim,
            "llm_layers": self.llm_layers,
            "llm_heads": self.llm_heads,
            "cross_attn_dim": self.cross_attn_dim,
            "cross_attn_heads": self.cross_attn_heads,
            "semantic_dim": self.semantic_dim,
            "lora_rank": self.lora_rank,
            "max_commit_radar_tokens": self.max_commit_radar_tokens,
        }
        if any(value <= 0 for value in positive.values()):
            raise ValueError("PARrad dimensions and timing values must be positive")
        if self.radar_tokens_per_update != 4:
            raise ValueError("PARrad validation requires exactly four radar tokens per update")
        if self.radar_dim % self.fusion_heads:
            raise ValueError("radar_dim must be divisible by fusion_heads")
        if self.radmamba_dim % self.fusion_heads:
            raise ValueError("radmamba_dim must be divisible by fusion_heads")
        if self.md_gru_dim % self.fusion_heads:
            raise ValueError("md_gru_dim must be divisible by fusion_heads")
        if self.llm_dim % self.llm_heads:
            raise ValueError("llm_dim must be divisible by llm_heads")
        if self.cross_attn_dim % self.cross_attn_heads:
            raise ValueError("cross_attn_dim must be divisible by cross_attn_heads")
        if self.encoder_type not in {"rd_md_tcn", "causal_radmamba", "causal_md_gru"}:
            raise ValueError("encoder_type must be rd_md_tcn, causal_radmamba, or causal_md_gru")
        if any(dilation <= 0 for dilation in (*self.rd_tcn_dilations, *self.md_tcn_dilations)):
            raise ValueError("TCN dilations must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if not 0 <= self.cross_attention_init_scale < 1:
            raise ValueError("cross_attention_init_scale must be in [0,1)")
        if not 0 < self.event_state_initial_update < 1:
            raise ValueError("event_state_initial_update must be in (0,1)")
        if not self.fixed_event_prompt.strip():
            raise ValueError("fixed_event_prompt must not be empty")

    @property
    def delta_t(self) -> float:
        """Physical duration represented by one online update."""
        return self.frame_period_s * self.frames_per_update

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
