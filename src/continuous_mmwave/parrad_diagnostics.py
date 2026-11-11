"""Offline continuous diagnostics for a trained PARrad Stage 1 checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from continuous_mmwave.checkpoint import load_training_checkpoint
from continuous_mmwave.data_parrad import PARradManifestDataset, build_sensor_vocabulary
from continuous_mmwave.parrad_config import PARradConfig
from continuous_mmwave.train import (
    _build_parrad_stage1,
    evaluate_parrad_stage1_streaming,
    load_experiment_config,
)


def _class_names(manifest: str | Path) -> list[str] | None:
    label_map = Path(manifest).parent / "label_map.json"
    if not label_map.exists():
        return None
    payload = json.loads(label_map.read_text(encoding="utf-8"))
    classes = sorted(payload["classes"], key=lambda item: int(item["id"]))
    return [str(item["command"]) for item in classes]


def run_parrad_diagnostics(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Evaluate every configured split with the same continuous streaming protocol."""
    config = load_experiment_config(config_path)
    if str(config.stage.get("variant")) != "parrad" or not isinstance(config.model, PARradConfig):
        raise ValueError("PARrad diagnostics require a PARrad Stage 1 config")
    manifests = {
        split: config.data[key]
        for split, key in (
            ("train", "train_manifest"),
            ("validation", "validation_manifest"),
            ("test", "test_manifest"),
        )
        if config.data.get(key)
    }
    vocabulary = build_sensor_vocabulary(*manifests.values())
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    class_weights = checkpoint["model_state"].get("class_weights")
    module = _build_parrad_stage1(config, class_weights)
    load_training_checkpoint(checkpoint_path, module)
    configured_device = str(config.training.get("device", "auto"))
    device = torch.device(
        "cuda" if configured_device == "auto" and torch.cuda.is_available()
        else configured_device if configured_device != "auto"
        else "cpu"
    )
    module.to(device)
    boundary_seconds = float(config.data.get("label_boundary_ignore_seconds", 0.0))
    boundary_frames = int(round(boundary_seconds / config.model.frame_period_s))
    if boundary_seconds > 0 and boundary_frames == 0:
        boundary_frames = 1
    tolerance_frames = int(
        config.data.get(
            "diagnostic_boundary_tolerance_frames",
            max(1, round(0.5 / config.model.frame_period_s)),
        )
    )
    chunk_frames = int(config.data.get("streaming_chunk_frames", config.data.get("window_frames", 64)))
    recording_batch_size = int(config.training.get("evaluation_recording_batch_size", 1))
    amp_enabled = bool(config.training.get("amp", True)) and device.type == "cuda"
    split_metrics: dict[str, dict[str, Any]] = {}
    for split, manifest in manifests.items():
        dataset = PARradManifestDataset(manifest, config.model, vocabulary)
        recordings = tuple(dataset[index] for index in range(len(dataset)))
        split_metrics[split] = evaluate_parrad_stage1_streaming(
            module,
            recordings,
            device,
            chunk_frames=chunk_frames,
            recording_batch_size=recording_batch_size,
            boundary_ignore_frames=boundary_frames,
            boundary_tolerance_frames=tolerance_frames,
            amp_enabled=amp_enabled,
        )
        del recordings, dataset
        if device.type == "cuda":
            torch.cuda.empty_cache()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint": str(checkpoint_path),
        "boundary_tolerance_frames": tolerance_frames,
        "class_names": _class_names(manifests["train"]),
        "splits": split_metrics,
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(run_parrad_diagnostics(args.config, args.checkpoint, args.output))


if __name__ == "__main__":
    main()
