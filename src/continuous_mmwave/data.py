"""Manifest-based radar data loading with variable-length masks."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


@dataclass
class RadarSample:
    """单个事件的原始数据；不同样本的 T/N 可变，由 collate 负责对齐。"""

    rd: Tensor  # [T,2,64,64]
    points: Tensor  # [T,N,6]
    point_mask: Tensor | None = None  # [T,N]，缺省时视为全部有效
    skeleton: Tensor | None = None  # [T,J,3]，仅 Stage 1 监督需要
    caption: str | None = None  # 仅 Stage 2 需要
    event_id: str = ""


@dataclass(frozen=True)
class RadarBatch:
    """collate 之后的定长张量，填充位置一律由 mask 标记而非特殊值。"""

    rd: Tensor  # [B,T,2,64,64]，T 为 4 的倍数
    points: Tensor  # [B,T,N,6]
    point_mask: Tensor  # [B,T,N]，已与 frame_mask 取与
    skeleton: Tensor | None  # [B,T,J,3] 或 None
    captions: tuple[str | None, ...]
    event_ids: tuple[str, ...]
    frame_mask: Tensor | None = None  # [B,T]，标记真实帧


def _validate(rd: Tensor, points: Tensor, point_mask: Tensor | None, skeleton: Tensor | None) -> None:
    """在采样与 collate 两个入口共用，尽早暴露形状错误。"""
    if rd.ndim != 4 or rd.shape[0] == 0 or rd.shape[1:] != (2, 64, 64):
        raise ValueError("rd must have shape [T,2,64,64] with T>0")
    if points.ndim != 3 or points.shape[0] != rd.shape[0] or points.shape[-1] != 6:
        raise ValueError("points must have shape [T,N,6]")
    if points.shape[1] == 0:
        # 允许全填充帧，但必须保留至少一个 pad 槽位，否则 max-pool 无定义。
        raise ValueError("points must reserve at least one padded point slot")
    if point_mask is not None and point_mask.shape != points.shape[:2]:
        raise ValueError("point_mask must have shape [T,N]")
    if skeleton is not None and (skeleton.ndim != 3 or skeleton.shape[0] != rd.shape[0] or skeleton.shape[-1] != 3):
        raise ValueError("skeleton must have shape [T,J,3]")


class ManifestDataset(Dataset[RadarSample]):
    """读取 JSONL manifest，每行给出相对 manifest 目录的 npz 路径。"""

    def __init__(self, manifest_path: str | Path) -> None:
        self.manifest_path = Path(manifest_path)
        # npz 路径既可以是相对（相对 manifest 所在目录）也可以是绝对路径。
        self.root = self.manifest_path.parent
        with self.manifest_path.open(encoding="utf-8") as handle:
            self.entries = [json.loads(line) for line in handle if line.strip()]
        if not self.entries or any("path" not in row for row in self.entries):
            raise ValueError("manifest must contain rows with path")

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> RadarSample:
        row = self.entries[index]
        path = Path(row["path"])
        if not path.is_absolute():
            path = self.root / path
        with np.load(path, allow_pickle=False) as arrays:
            rd = torch.from_numpy(np.asarray(arrays["rd"], dtype=np.float32))
            points = torch.from_numpy(np.asarray(arrays["points"], dtype=np.float32))
            point_mask = None
            if "point_mask" in arrays:
                # 优先使用显式 bool mask。
                point_mask = torch.from_numpy(np.asarray(arrays["point_mask"], dtype=np.bool_))
            elif "point_counts" in arrays:
                # 否则由每帧有效点数展开为前 N 个点有效的 mask。
                counts = torch.from_numpy(np.asarray(arrays["point_counts"], dtype=np.int64))
                if counts.shape != (points.shape[0],) or (counts < 0).any() or (counts > points.shape[1]).any():
                    raise ValueError("point_counts must have shape [T] and remain in [0,N]")
                point_mask = torch.arange(points.shape[1]).unsqueeze(0) < counts.unsqueeze(1)
            skeleton = (
                torch.from_numpy(np.asarray(arrays["skeleton"], dtype=np.float32)) if "skeleton" in arrays else None
            )
        _validate(rd, points, point_mask, skeleton)
        return RadarSample(rd, points, point_mask, skeleton, row.get("caption"), str(row.get("event_id", index)))


def radar_collate(samples: Sequence[RadarSample]) -> RadarBatch:
    """把变长样本填充成定长 batch，并生成 frame_mask / point_mask。"""
    if not samples:
        raise ValueError("cannot collate an empty batch")
    for sample in samples:
        _validate(sample.rd, sample.points, sample.point_mask, sample.skeleton)
    skeleton_flags = [sample.skeleton is not None for sample in samples]
    # 若混合有无，无法构造统一的监督张量，直接拒绝。
    if any(skeleton_flags) and not all(skeleton_flags):
        raise ValueError("a batch must either contain skeletons for every sample or none")
    if any(sample.rd.shape[1:] != samples[0].rd.shape[1:] for sample in samples):
        raise ValueError("RD shapes must match")

    batch = len(samples)
    # 时间维向上取整到 4 的倍数，保证每个样本都能被切成整数个 update。
    frames = ((max(sample.rd.shape[0] for sample in samples) + 3) // 4) * 4
    points_count = max(sample.points.shape[1] for sample in samples)
    rd = samples[0].rd.new_zeros((batch, frames, 2, 64, 64))
    points = samples[0].points.new_zeros((batch, frames, points_count, 6))
    point_mask = torch.zeros((batch, frames, points_count), dtype=torch.bool)
    frame_mask = torch.zeros((batch, frames), dtype=torch.bool)
    # 填充值统一为 0，是否有效完全由 mask 决定（模型内部会用 mask 清零被填充位置）。
    for i, sample in enumerate(samples):
        time, count = sample.points.shape[:2]
        rd[i, :time] = sample.rd
        points[i, :time, :count] = sample.points
        frame_mask[i, :time] = True
        point_mask[i, :time, :count] = True if sample.point_mask is None else sample.point_mask.bool()

    skeleton = None
    if all(skeleton_flags):
        # 关节点数必须一致，否则无法堆到一个张量里。
        joints = samples[0].skeleton.shape[1]  # type: ignore[union-attr]
        if any(sample.skeleton.shape[1] != joints for sample in samples):  # type: ignore[union-attr]
            raise ValueError("joint layouts must match")
        skeleton = samples[0].skeleton.new_zeros((batch, frames, joints, 3))  # type: ignore[union-attr]
        for i, sample in enumerate(samples):
            skeleton[i, : sample.rd.shape[0]] = sample.skeleton  # type: ignore[index]
    return RadarBatch(
        rd,
        points,
        point_mask,
        skeleton,
        tuple(sample.caption for sample in samples),
        tuple(sample.event_id for sample in samples),
        frame_mask,
    )
