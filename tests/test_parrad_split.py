from __future__ import annotations

import json
from pathlib import Path


def test_split_manifest_by_subject_preserves_paths_and_is_disjoint(tmp_path: Path) -> None:
    from continuous_mmwave.data_parrad import assert_disjoint_splits
    from continuous_mmwave.parrad_split import split_manifest_by_subject

    source_root = tmp_path / "canonical"
    recordings = source_root / "recordings"
    recordings.mkdir(parents=True)
    rows = []
    for subject in ("s1", "s2", "s3"):
        for sensor in ("60ghz", "77ghz"):
            recording = recordings / f"{subject}-{sensor}.npz"
            recording.touch()
            rows.append(
                {
                    "path": f"recordings/{recording.name}",
                    "subject_id": subject,
                    "session_id": "set000",
                    "sensor_id": sensor,
                    "environment_id": "test",
                }
            )
    source = source_root / "manifest.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    paths = split_manifest_by_subject(
        source,
        tmp_path / "splits",
        train_subjects=("s1",),
        validation_subjects=("s2",),
        test_subjects=("s3",),
    )

    split_rows = {
        name: [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for name, path in paths.items()
    }
    assert {name: len(items) for name, items in split_rows.items()} == {
        "train": 2,
        "validation": 2,
        "test": 2,
    }
    assert_disjoint_splits(**split_rows)
    for name, items in split_rows.items():
        for row in items:
            assert (paths[name].parent / row["path"]).resolve().is_file()

