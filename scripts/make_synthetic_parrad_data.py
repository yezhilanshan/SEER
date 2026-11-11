"""Create deterministic canonical PARrad fixtures for tests and smoke runs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def generate(root: str | Path = "data/synthetic_parrad") -> Path:
    """Generate four aligned RD/MD recordings and return their manifest."""
    output = Path(root)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(17)
    rows = []
    for subject_index in range(2):
        for sensor_index, sensor_id in enumerate(("60ghz", "77ghz")):
            frames = 3 + subject_index
            rd = rng.normal(size=(frames, 1, 93, 128)).astype(np.float32)
            md = rd.mean(axis=2).squeeze(1).astype(np.float32)
            labels = ((np.arange(frames) + subject_index) % 3).astype(np.int64)
            timestamps = np.arange(frames, dtype=np.float64) * 0.09
            recording = output / f"subject-{subject_index}-{sensor_id}.npz"
            np.savez_compressed(recording, rd=rd, md=md, labels=labels, timestamps=timestamps)
            rows.append(
                {
                    "path": recording.name,
                    "subject_id": f"synthetic-{subject_index}",
                    "session_id": f"session-{subject_index}",
                    "sensor_id": sensor_id,
                    "environment_id": "synthetic",
                    "sensor_index": sensor_index,
                }
            )
    manifest = output / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    print(generate())


if __name__ == "__main__":
    main()
