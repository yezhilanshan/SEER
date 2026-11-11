"""Read-only audit for canonical PARrad manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from continuous_mmwave.data_parrad import PARradManifestDataset, build_sensor_vocabulary


def audit_manifest(path: str | Path) -> dict[str, Any]:
    """Summarize shapes, timing, labels, metadata, and RD/MD agreement."""
    vocabulary = build_sensor_vocabulary(path)
    dataset = PARradManifestDataset(path, sensor_vocabulary=vocabulary)
    labels: Counter[int] = Counter()
    intervals: list[float] = []
    correlations: list[float] = []
    rd_shapes: set[tuple[int, ...]] = set()
    md_bins: set[int] = set()
    subjects: set[str] = set()
    sessions: set[str] = set()
    sensors: set[str] = set()
    environments: set[str] = set()
    frames = 0
    all_finite = True
    for sample in dataset:
        frames += sample.rd.shape[0]
        rd_shapes.add(tuple(sample.rd.shape[1:]))
        md_bins.add(sample.md.shape[1])
        labels.update(int(value) for value in sample.labels.tolist() if int(value) != -100)
        subjects.add(sample.subject_id)
        sessions.add(sample.session_id)
        sensors.add(sample.sensor_id)
        environments.add(sample.environment_id)
        intervals.extend(float(value) for value in sample.timestamps.diff().tolist())
        all_finite = all_finite and bool(sample.rd.isfinite().all() and sample.md.isfinite().all())
        range_mean = sample.rd.mean(dim=-2).mean(dim=1)
        if range_mean.shape == sample.md.shape:
            left = range_mean.flatten().numpy()
            right = sample.md.flatten().numpy()
            if left.std() > 0 and right.std() > 0:
                correlations.append(float(np.corrcoef(left, right)[0, 1]))
    interval_array = np.asarray(intervals, dtype=np.float64)
    interval_summary = {
        "minimum": float(interval_array.min()) if interval_array.size else None,
        "median": float(np.median(interval_array)) if interval_array.size else None,
        "maximum": float(interval_array.max()) if interval_array.size else None,
    }
    return {
        "recordings": len(dataset),
        "frames": frames,
        "rd_shapes": [list(shape) for shape in sorted(rd_shapes)],
        "md_bins": sorted(md_bins),
        "label_histogram": {str(label): labels[label] for label in sorted(labels)},
        "subjects": sorted(subjects),
        "sessions": sorted(sessions),
        "sensors": sorted(sensors),
        "environments": sorted(environments),
        "frame_interval_seconds": interval_summary,
        "all_finite": all_finite,
        "rd_md_range_mean_correlation": {
            "mean": float(np.mean(correlations)) if correlations else None,
            "recordings": len(correlations),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(audit_manifest(args.manifest), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
