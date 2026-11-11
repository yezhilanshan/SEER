"""Stage 1: pose-guided radar pretraining."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from continuous_mmwave.data import RadarBatch
from continuous_mmwave.models.pose import PoseHead, mpjpe_loss, velocity_loss
from continuous_mmwave.models.radar_encoder import RadarEncoderOutput, StreamingRadarEncoder


@dataclass(frozen=True)
class Stage1Loss:
    """Stage 1 的三个指标；total = mpjpe + velocity_weight * velocity。"""

    total: Tensor
    mpjpe: Tensor
    velocity: Tensor


class Stage1Module(nn.Module):
    """Stage 1：用骨架监督预训练 Radar Encoder，使其隐层包含运动信息。"""

    def __init__(
        self, radar_encoder: StreamingRadarEncoder, num_joints: int = 17, velocity_weight: float = 1.0
    ) -> None:
        super().__init__()
        self.radar_encoder = radar_encoder
        self.pose_head = PoseHead(radar_encoder.config.radar_dim, num_joints)
        self.velocity_weight = velocity_weight

    def forward(self, batch: RadarBatch) -> tuple[Tensor, RadarEncoderOutput]:
        """返回逐帧姿态预测 [B,T,J,3] 与 Radar Encoder 的完整输出。"""
        encoded = self.radar_encoder.forward_sequence(batch.rd, batch.points, batch.point_mask, batch.frame_mask)
        # 姿态头使用逐帧特征而非 5 Hz 的 radar token，以保留时间分辨率。
        return self.pose_head(encoded.frame_hidden), encoded

    def training_step(self, batch: RadarBatch) -> Stage1Loss:
        if batch.skeleton is None:
            raise ValueError("Stage 1 requires skeleton supervision")
        prediction, _ = self(batch)
        pose = mpjpe_loss(prediction, batch.skeleton, batch.frame_mask)
        velocity = velocity_loss(prediction, batch.skeleton, batch.frame_mask)
        return Stage1Loss(pose + self.velocity_weight * velocity, pose, velocity)

    def export_radar_encoder_state(self) -> dict[str, Tensor]:
        """导出仅含 Radar Encoder 的 state（脱离计算图并转到 CPU），供 Stage 2 加载。"""
        return {name: value.detach().cpu().clone() for name, value in self.radar_encoder.state_dict().items()}
