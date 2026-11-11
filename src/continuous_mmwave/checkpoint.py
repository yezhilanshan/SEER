"""Staged checkpoint contracts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

def _rng() -> dict[str, Any]:
    """快照所有随机源状态，保证断点恢复后的数据顺序与初始化完全一致。"""
    state = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def save_training_checkpoint(
    path: str | Path,
    *,
    stage: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    config: dict[str, Any],
    metric_history: list[dict[str, float]],
    trainable_only: bool = False,
    scaler: Any = None,
) -> None:
    state = model.state_dict()
    if trainable_only:
        # Stage 2 的冻结 LLM 权重巨大且不会变化，只保存 requires_grad 的部分。
        names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        state = {name: value for name, value in state.items() if name in names}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": stage,
            "model_state": state,
            "trainable_only": trainable_only,
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": None if scaler is None else scaler.state_dict(),
            "epoch": epoch,
            "step": step,
            "config": config,
            "metric_history": metric_history,
            "rng_state": _rng(),
        },
        path,
    )


def load_training_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    restore_rng: bool = False,
    scaler: Any = None,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    # trainable_only 的 checkpoint 只含部分参数，此时不能使用 strict 加载。
    model.load_state_dict(payload["model_state"], strict=not payload.get("trainable_only", False))
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scaler is not None and payload.get("scaler_state") is not None:
        scaler.load_state_dict(payload["scaler_state"])
    if restore_rng:
        random.setstate(payload["rng_state"]["python"])
        np.random.set_state(payload["rng_state"]["numpy"])
        torch.set_rng_state(payload["rng_state"]["torch"])
        if torch.cuda.is_available() and "cuda" in payload["rng_state"]:
            torch.cuda.set_rng_state_all(payload["rng_state"]["cuda"])
    return payload


def save_stage1_radar_export(path: str | Path, module: nn.Module, model_config: dict[str, Any]) -> None:
    """导出 Stage 1 的 Radar Encoder，同时记录 model_config 供 Stage 2 校验维度兼容性。"""
    if hasattr(module, "export_radar_encoder_state"):
        radar_state = module.export_radar_encoder_state()
    elif hasattr(module, "export_encoder_state"):
        radar_state = module.export_encoder_state()
    else:
        raise TypeError("Stage 1 module must provide an encoder export method")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "stage": "stage1-radar-export",
            "model_config": model_config,
            "radar_encoder_state": radar_state,
        },
        path,
    )


def load_stage1_radar_state(path: str | Path, radar_encoder: nn.Module) -> dict[str, Any]:
    """加载 Stage 1 导出的 Radar Encoder 权重。"""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("stage") != "stage1-radar-export":
        raise ValueError("not a Stage 1 radar export")
    state = payload["radar_encoder_state"]
    # 方案书要求：Pose Head 不得进入后续阶段，因此显式拒绝这类文件。
    if any("pose_head" in name for name in state):
        raise ValueError("radar export must not contain Pose Head")
    radar_encoder.load_state_dict(state, strict=True)
    return payload
