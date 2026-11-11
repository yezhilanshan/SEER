"""Small offline encoder configuration; no streaming state contracts."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RadarTextConfig:
    rd_channels: int = 1
    feature_dim: int = 128
    gru_hidden: int = 128
    gru_layers: int = 2
    tokens: int = 8
    md_context: int = 9
    dropout: float = 0.1
    normalization: str = "signed_log_zscore"
    max_answer_tokens: int = 192

    def __post_init__(self):
        for name in ("rd_channels", "feature_dim", "gru_hidden", "gru_layers", "tokens",
                     "md_context", "max_answer_tokens"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.md_context % 2 != 1:
            raise ValueError("md_context must be odd")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if self.normalization not in {"none", "signed_log_zscore"}:
            raise ValueError("normalization must be none or signed_log_zscore")

    def to_dict(self):
        return asdict(self)
