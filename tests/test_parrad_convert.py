from __future__ import annotations

import csv
import importlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import h5py
import numpy as np


def _write_commands(path: Path, rows: list[tuple[datetime, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("timestamp", "command"))
        writer.writerows((timestamp.isoformat(), command) for timestamp, command in rows)


def test_assign_frame_labels_obeys_command_boundaries_and_ignore_prefix() -> None:
    module = importlib.import_module("continuous_mmwave.parrad_convert")
    frame_timestamps = np.array([0.0, 0.9, 1.0, 2.4, 2.5, 3.0], dtype=np.float64)
    command_offsets = np.array([1.0, 2.5], dtype=np.float64)

    labels = module.assign_frame_labels(
        frame_timestamps,
        command_offsets,
        ("walk to bed", "sit down on bed"),
        {"sit down on bed": 0, "walk to bed": 1},
    )

    assert labels.dtype == np.int64
    assert labels.tolist() == [-100, -100, 1, 1, 0, 0]


def test_read_command_annotations_normalizes_whitespace_and_rejects_nonmonotonic_time(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.parrad_convert")
    commands = tmp_path / "commands.csv"
    start = datetime(2019, 11, 25, 17, 34, 10)
    _write_commands(
        commands,
        [
            (start, "get in  bed"),
            (start + timedelta(seconds=1), "  sit down on bed  "),
        ],
    )

    annotations = module.read_command_annotations(commands)

    assert [annotation.command for annotation in annotations] == ["get in bed", "sit down on bed"]
    assert [annotation.timestamp for annotation in annotations] == [start, start + timedelta(seconds=1)]

    _write_commands(commands, [(start + timedelta(seconds=1), "walk"), (start, "sit")])
    try:
        module.read_command_annotations(commands)
    except ValueError as error:
        assert "strictly increasing" in str(error)
    else:
        raise AssertionError("non-monotonic command timestamps must fail closed")


def test_convert_recording_writes_two_canonical_sensor_npz_files(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.parrad_convert")
    source = tmp_path / "processed_data.h5"
    with h5py.File(source, "w") as handle:
        radar = handle.create_group("radar")
        for sensor_id, frames, offset in (("TI_xWR14xx", 3, 0.0), ("TI_xWR68xx", 4, 100.0)):
            sensor = radar.create_group(sensor_id)
            rdn = np.arange(frames * 93 * 128, dtype=np.float32).reshape(frames, 93, 128) + offset
            md = np.arange(frames * 128, dtype=np.float32).reshape(frames, 128) + offset
            sensor.create_dataset("rdn", data=rdn)
            sensor.create_dataset("mDoppler", data=md)

    commands = tmp_path / "timestamp_speech.csv"
    start = datetime(2019, 11, 25, 17, 34, 10)
    _write_commands(
        commands,
        [
            (start, "walk to bed"),
            (start + timedelta(seconds=0.18), "sit down on bed"),
        ],
    )

    manifest = module.convert_recording(
        source,
        commands,
        tmp_path / "canonical",
        subject_id="subject_04",
        session_id="set000",
        environment_id="parrad",
    )

    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    assert [row["sensor_id"] for row in rows] == ["TI_xWR14xx", "TI_xWR68xx"]
    assert all(row["label_policy"] == "command_active_until_next_command" for row in rows)

    first_path = manifest.parent / rows[0]["path"]
    with np.load(first_path, allow_pickle=False) as arrays:
        assert set(arrays.files) == {"rd", "md", "labels", "timestamps"}
        assert arrays["rd"].shape == (3, 1, 93, 128)
        assert arrays["md"].shape == (3, 128)
        assert arrays["rd"].dtype == arrays["md"].dtype == np.float32
        assert arrays["labels"].dtype == np.int64
        assert arrays["timestamps"].dtype == np.float64
        assert arrays["timestamps"].tolist() == [0.0, 0.09, 0.18]
        assert arrays["labels"].tolist() == [1, 1, 0]
        assert arrays["rd"][0, 0, 0, 0] == 0.0

    label_map = json.loads((manifest.parent / "label_map.json").read_text(encoding="utf-8"))
    assert label_map == {
        "version": 1,
        "ignore_index": -100,
        "classes": [
            {"id": 0, "command": "sit down on bed"},
            {"id": 1, "command": "walk to bed"},
        ],
    }


def test_convert_recording_md_only_writes_empty_rd_placeholder(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.parrad_convert")
    source = tmp_path / "processed_data.h5"
    expected_md = np.arange(2 * 128, dtype=np.float32).reshape(2, 128)
    with h5py.File(source, "w") as handle:
        sensor = handle.create_group("radar").create_group("TI_xWR14xx")
        sensor.create_dataset("rdn", data=np.ones((2, 93, 128), dtype=np.float32))
        sensor.create_dataset("mDoppler", data=expected_md)

    commands = tmp_path / "timestamp_speech.csv"
    _write_commands(commands, [(datetime(2019, 11, 25, 17, 34, 10), "walk to bed")])

    manifest = module.convert_recording(
        source,
        commands,
        tmp_path / "canonical",
        subject_id="subject_04",
        session_id="set000",
        environment_id="parrad",
        md_only=True,
    )

    row = json.loads(manifest.read_text(encoding="utf-8"))
    with np.load(manifest.parent / row["path"], allow_pickle=False) as arrays:
        assert arrays["rd"].shape == (2, 0, 0, 0)
        assert arrays["rd"].dtype == np.float32
        np.testing.assert_array_equal(arrays["md"], expected_md)
