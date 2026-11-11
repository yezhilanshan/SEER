from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def test_parrad_frozen_contract() -> None:
    """Changing timing or token count must break the public PARrad contract."""
    module = importlib.import_module("continuous_mmwave.parrad_config")
    config = module.PARradConfig()

    assert config.frames_per_update == 2
    assert config.delta_t == pytest.approx(0.18)
    assert config.radar_tokens_per_update == 4
    assert (config.rd_channels, config.rd_height, config.rd_width) == (1, 93, 128)
    assert (config.md_bins, config.radar_dim, config.motion_dim) == (128, 256, 128)


@pytest.mark.parametrize(
    "overrides",
    (
        {"frames_per_update": 0},
        {"frame_period_s": 0},
        {"radar_tokens_per_update": 3},
        {"radar_dim": 15, "fusion_heads": 8},
        {"dropout": 1.0},
    ),
)
def test_parrad_config_rejects_values_that_break_tensor_contracts(overrides: dict[str, object]) -> None:
    module = importlib.import_module("continuous_mmwave.parrad_config")
    with pytest.raises(ValueError):
        module.PARradConfig(**overrides)


def _write_recording(
    root: Path,
    name: str,
    frames: int,
    *,
    md_frames: int | None = None,
    labels: np.ndarray | None = None,
    timestamps: np.ndarray | None = None,
) -> Path:
    path = root / name
    rng = np.random.default_rng(frames)
    np.savez_compressed(
        path,
        rd=rng.normal(size=(frames, 1, 93, 128)).astype("float32"),
        md=rng.normal(size=(frames if md_frames is None else md_frames, 128)).astype("float32"),
        labels=(np.arange(frames, dtype=np.int64) % 3 if labels is None else np.asarray(labels, dtype=np.int64)),
        timestamps=(
            np.arange(frames, dtype=np.float64) * 0.09 if timestamps is None else np.asarray(timestamps, dtype=np.float64)
        ),
    )
    return path


def _write_manifest(root: Path, rows: list[dict[str, object]]) -> Path:
    path = root / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _row(path: Path, *, subject: str = "s1", session: str = "a", sensor: str = "60ghz") -> dict[str, object]:
    return {
        "path": path.name,
        "subject_id": subject,
        "session_id": session,
        "sensor_id": sensor,
        "environment_id": "hospital",
    }


def test_manifest_loads_aligned_canonical_npz(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    recording = _write_recording(tmp_path, "sample.npz", 3)
    manifest = _write_manifest(tmp_path, [_row(recording)])
    dataset = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    sample = dataset[0]

    assert sample.rd.shape == (3, 1, 93, 128)
    assert sample.md.shape == (3, 128)
    assert sample.labels.shape == sample.timestamps.shape == (3,)
    assert sample.sensor_index == 0
    assert sample.subject_id == "s1"


def test_parrad_collate_pads_time_and_builds_mask(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    first = _write_recording(tmp_path, "first.npz", 2)
    second = _write_recording(tmp_path, "second.npz", 4)
    manifest = _write_manifest(
        tmp_path,
        [_row(first, subject="s1", session="a"), _row(second, subject="s2", session="b")],
    )
    dataset = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    batch = module.parrad_collate([dataset[0], dataset[1]])

    assert batch.rd.shape == (2, 4, 1, 93, 128)
    assert batch.md.shape == (2, 4, 128)
    assert batch.labels.tolist() == [[0, 1, -100, -100], [0, 1, 2, 0]]
    assert batch.frame_mask.tolist() == [[True, True, False, False], [True, True, True, True]]
    assert batch.sensor_index.tolist() == [0, 0]
    assert torch.isfinite(batch.rd).all()


def test_alignment_mismatch_fails_closed(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    recording = _write_recording(tmp_path, "bad.npz", 3, md_frames=2)
    manifest = _write_manifest(tmp_path, [_row(recording)])
    dataset = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    with pytest.raises(ValueError, match="RD, MD, labels, and timestamps"):
        dataset[0]


def test_non_monotonic_timestamps_fail_closed(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    recording = _write_recording(tmp_path, "time.npz", 3, timestamps=np.array([0.0, 0.18, 0.09]))
    manifest = _write_manifest(tmp_path, [_row(recording)])
    dataset = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    with pytest.raises(ValueError, match="strictly increasing"):
        dataset[0]


def test_sensor_vocabulary_is_stable_and_rejects_unknown_ids(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    first = _write_recording(tmp_path, "first.npz", 2)
    second = _write_recording(tmp_path, "second.npz", 2)
    manifest = _write_manifest(
        tmp_path,
        [_row(first, sensor="77ghz"), _row(second, subject="s2", session="b", sensor="60ghz")],
    )

    vocabulary = module.build_sensor_vocabulary(manifest)

    assert vocabulary == {"60ghz": 0, "77ghz": 1}
    with pytest.raises(ValueError, match="unknown sensor_id"):
        module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})


def test_assert_disjoint_splits_rejects_subject_overlap() -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    train = [{"subject_id": "s1", "session_id": "a"}]
    test = [{"subject_id": "s1", "session_id": "b"}]

    with pytest.raises(ValueError, match="subject overlap"):
        module.assert_disjoint_splits(train=train, validation=[], test=test)


def test_assert_disjoint_splits_allows_local_session_ids_on_different_subjects() -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    train = [{"subject_id": "s1", "session_id": "a"}]
    test = [{"subject_id": "s2", "session_id": "a"}]

    module.assert_disjoint_splits(train=train, validation=[], test=test)


def test_audit_reports_shapes_timing_classes_and_metadata(tmp_path: Path) -> None:
    audit_module = importlib.import_module("continuous_mmwave.parrad_audit")
    first = _write_recording(tmp_path, "first.npz", 3)
    second = _write_recording(tmp_path, "second.npz", 3)
    manifest = _write_manifest(
        tmp_path,
        [
            _row(first, subject="s1", session="a", sensor="60ghz"),
            _row(second, subject="s2", session="b", sensor="77ghz"),
        ],
    )

    result = audit_module.audit_manifest(manifest)

    assert result["recordings"] == 2
    assert result["frames"] == 6
    assert result["rd_shapes"] == [[1, 93, 128]]
    assert result["md_bins"] == [128]
    assert result["subjects"] == ["s1", "s2"]
    assert result["sensors"] == ["60ghz", "77ghz"]
    assert result["label_histogram"] == {"0": 2, "1": 2, "2": 2}
    assert result["frame_interval_seconds"]["median"] == pytest.approx(0.09)
    assert result["all_finite"] is True


def test_synthetic_generator_produces_auditable_rd_md_pairs(tmp_path: Path) -> None:
    generator = importlib.import_module("scripts.make_synthetic_parrad_data")
    audit_module = importlib.import_module("continuous_mmwave.parrad_audit")

    manifest = generator.generate(tmp_path / "synthetic")
    result = audit_module.audit_manifest(manifest)

    assert result["recordings"] == 4
    assert result["subjects"] == ["synthetic-0", "synthetic-1"]
    assert result["sensors"] == ["60ghz", "77ghz"]
    assert result["rd_md_range_mean_correlation"]["mean"] == pytest.approx(1.0, abs=1e-6)


def test_window_dataset_keeps_recordings_separate_and_includes_tail(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    first = _write_recording(tmp_path, "first.npz", 5)
    second = _write_recording(tmp_path, "second.npz", 3)
    manifest = _write_manifest(
        tmp_path,
        [_row(first, subject="s1", session="a"), _row(second, subject="s2", session="b")],
    )
    recordings = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    windows = module.PARradWindowDataset(recordings, window_frames=4, window_stride=4)

    assert len(windows) == 3
    assert [windows[index].rd.shape[0] for index in range(3)] == [4, 1, 3]
    assert [windows[index].subject_id for index in range(3)] == ["s1", "s1", "s2"]
    assert windows[1].labels.tolist() == [1]


def test_mask_label_boundaries_ignores_equal_radius_on_both_sides() -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    labels = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

    masked = module.mask_label_boundaries(labels, radius_frames=1)

    assert masked.tolist() == [0, 0, -100, -100, 1, -100, -100, 2]
    assert labels.tolist() == [0, 0, 0, 1, 1, 1, 2, 2]


def test_window_dataset_drops_windows_fully_masked_by_label_boundaries(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    recording = _write_recording(
        tmp_path,
        "sample.npz",
        8,
        labels=np.array([0, 0, 1, 1, 1, 1, 1, 1]),
    )
    manifest = _write_manifest(tmp_path, [_row(recording)])
    recordings = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    windows = module.PARradWindowDataset(
        recordings,
        window_frames=4,
        window_stride=4,
        label_boundary_ignore_frames=2,
    )

    assert len(windows) == 1
    assert windows[0].labels.tolist() == [1, 1, 1, 1]


def test_window_dataset_requires_whole_update_window(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    recording = _write_recording(tmp_path, "sample.npz", 5)
    manifest = _write_manifest(tmp_path, [_row(recording)])
    recordings = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    with pytest.raises(ValueError, match="multiple of frames_per_update"):
        module.PARradWindowDataset(recordings, window_frames=3, window_stride=3)


def test_label_run_windows_trim_transitions_split_long_runs_and_keep_tails(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    labels = np.array([0] * 6 + [1] * 7, dtype=np.int64)
    recording = _write_recording(tmp_path, "runs.npz", len(labels), labels=labels)
    manifest = _write_manifest(tmp_path, [_row(recording)])
    recordings = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    windows = module.PARradWindowDataset(
        recordings,
        window_frames=4,
        label_boundary_ignore_frames=1,
        window_mode="label_run",
    )

    assert [window.labels.tolist() for window in windows] == [
        [0, 0, 0, 0],
        [0],
        [1, 1, 1, 1],
        [1, 1],
    ]
    assert [window.timestamps[0].item() for window in windows] == pytest.approx([0.0, 0.36, 0.63, 0.99])
    assert all(window.labels.ne(-100).all() for window in windows)


def test_label_run_windows_drop_ignored_gaps_and_runs_consumed_by_boundary_trim(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    labels = np.array([0, 0, -100, -100, 1, 1, 2, 2, 1, 1], dtype=np.int64)
    recording = _write_recording(tmp_path, "gaps.npz", len(labels), labels=labels)
    manifest = _write_manifest(tmp_path, [_row(recording)])
    recordings = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    untrimmed = module.PARradWindowDataset(
        recordings,
        window_frames=4,
        label_boundary_ignore_frames=0,
        window_mode="label_run",
    )
    fully_trimmed = module.PARradWindowDataset(
        recordings,
        window_frames=4,
        label_boundary_ignore_frames=2,
        window_mode="label_run",
    )

    assert [window.labels.tolist() for window in untrimmed] == [[0, 0], [1, 1], [2, 2], [1, 1]]
    assert len(fully_trimmed) == 1
    assert fully_trimmed[0].labels.tolist() == [0, 0]


def test_window_dataset_rejects_unknown_window_mode(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    recording = _write_recording(tmp_path, "sample.npz", 4, labels=np.zeros(4, dtype=np.int64))
    manifest = _write_manifest(tmp_path, [_row(recording)])
    recordings = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    with pytest.raises(ValueError, match="window_mode"):
        module.PARradWindowDataset(recordings, window_frames=4, window_mode="oracle")


def test_window_dataset_can_reuse_cached_recordings_between_training_phases(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    recording = _write_recording(tmp_path, "sample.npz", 8, labels=np.zeros(8, dtype=np.int64))
    manifest = _write_manifest(tmp_path, [_row(recording)])
    source = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})
    label_runs = module.PARradWindowDataset(source, window_frames=4, window_mode="label_run")

    streaming = module.PARradWindowDataset(
        source,
        window_frames=8,
        window_mode="fixed",
        cached_recordings=label_runs.recordings,
    )

    assert streaming.recordings[0] is label_runs.recordings[0]


def test_label_run_minimum_drops_short_runs_and_rebalances_tiny_tails(tmp_path: Path) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    labels = np.array([0] * 67 + [1] * 7, dtype=np.int64)
    recording = _write_recording(tmp_path, "minimum.npz", len(labels), labels=labels)
    manifest = _write_manifest(tmp_path, [_row(recording)])
    source = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    windows = module.PARradWindowDataset(
        source,
        window_frames=64,
        window_mode="label_run",
        min_label_run_frames=8,
    )

    assert [len(window.labels) for window in windows] == [34, 33]
    assert all(window.labels.eq(0).all() for window in windows)


@pytest.mark.parametrize("minimum", (0, 65))
def test_label_run_minimum_must_fit_the_window(tmp_path: Path, minimum: int) -> None:
    module = importlib.import_module("continuous_mmwave.data_parrad")
    recording = _write_recording(tmp_path, "sample.npz", 8, labels=np.zeros(8, dtype=np.int64))
    manifest = _write_manifest(tmp_path, [_row(recording)])
    source = module.PARradManifestDataset(manifest, sensor_vocabulary={"60ghz": 0})

    with pytest.raises(ValueError, match="min_label_run_frames"):
        module.PARradWindowDataset(
            source,
            window_frames=64,
            window_mode="label_run",
            min_label_run_frames=minimum,
        )
