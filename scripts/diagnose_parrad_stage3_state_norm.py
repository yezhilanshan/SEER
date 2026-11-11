"""Capture validation-only Stage P3 controller state-norm trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from continuous_mmwave.parrad_stage3_calibrate import (
    build_runtime,
    load_validation_recordings,
)
from continuous_mmwave.parrad_stage3_diagnostics import (
    ControllerThresholds,
    evaluate_controller,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--thresholds", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    artifact = json.loads(args.thresholds.read_text(encoding="utf-8"))
    if artifact.get("source_split") != "validation":
        raise ValueError("state-norm diagnosis requires a validation-only artifact")
    if artifact.get("test_evaluation_allowed") is not False:
        raise ValueError("test evaluation must remain disabled during state-norm diagnosis")
    expected_sha = artifact["selected_checkpoint"]["sha256"]
    actual_sha = _sha256(args.checkpoint)
    if actual_sha != expected_sha:
        raise ValueError("checkpoint SHA does not match the selected validation artifact")

    threshold_payload = dict(artifact["thresholds"])
    thresholds = ControllerThresholds(**threshold_payload)
    device = torch.device(args.device)
    runtime = build_runtime(args.config, args.checkpoint, device)
    recordings = load_validation_recordings(args.config)
    metrics = evaluate_controller(
        runtime,
        recordings,
        thresholds,
        capture_state_trace=True,
    )
    payload = {
        "schema_version": 1,
        "source_split": "validation",
        "test_evaluation_allowed": False,
        "checkpoint_sha256": actual_sha,
        "metrics": metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "trace_rows": len(metrics["state_trace"]),
                "state_norm": metrics["state_norm"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
