"""Temporary Stage-1 pose head and physical losses."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PoseHead(nn.Module):
    """Stage 1 的临时姿态头：逐帧回归 J×3 关节坐标。

    仅用于 Stage 1 预训练提供监督信号，导出 radar encoder 时会被严格剔除。
    """

    def __init__(self, input_dim: int = 256, joints: int = 17) -> None:
        super().__init__()
        self.joints = joints
        self.network = nn.Sequential(nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, joints * 3))

    def forward(self, hidden: Tensor) -> Tensor:
        """[B,T,radar_dim] -> [B,T,J,3]。"""
        return self.network(hidden).reshape(*hidden.shape[:-1], self.joints, 3)


def _masked_mean(values: Tensor, mask: Tensor | None) -> Tensor:
    """对 mask 为真的位置取均值；mask 为 None 时退化为全局均值。

    mask 会自动在尾部补维度以广播到 values，并 clamp 分母避免全 padding 时除零。
    """
    if mask is None:
        return values.mean()
    expanded = mask.to(values.dtype)
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(-1)
    return (values * expanded).sum() / expanded.expand_as(values).sum().clamp_min(1)


def mpjpe_loss(prediction: Tensor, target: Tensor, frame_mask: Tensor | None = None) -> Tensor:
    """MPJPE：逐关节欧氏距离在有效帧上的均值。"""
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    return _masked_mean(torch.linalg.vector_norm(prediction - target, dim=-1), frame_mask)


def velocity_loss(prediction: Tensor, target: Tensor, frame_mask: Tensor | None = None) -> Tensor:
    """速度损失：相邻帧差分（速度为随时间的变化量）的 L1 误差。

    diff 后时间长度减一，因此 mask 取相邻两帧均有效的位置。
    """
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    error = torch.abs(torch.diff(prediction, dim=1) - torch.diff(target, dim=1))
    mask = None if frame_mask is None else frame_mask[:, 1:] & frame_mask[:, :-1]
    return _masked_mean(error, mask)
