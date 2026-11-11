"""Canonical PARrad data contracts and manifest loading."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from continuous_mmwave.parrad_config import PARradConfig


_METADATA_FIELDS = ("subject_id", "session_id", "sensor_id", "environment_id")


@dataclass
class PARradSample:
    """One aligned continuous PARrad recording."""

    rd: Tensor
    md: Tensor
    labels: Tensor
    timestamps: Tensor
    sensor_index: int
    subject_id: str
    session_id: str
    sensor_id: str
    environment_id: str


@dataclass(frozen=True)
class PARradBatch:
    """Padded PARrad batch with explicit frame validity."""

    rd: Tensor
    md: Tensor
    labels: Tensor
    timestamps: Tensor
    frame_mask: Tensor
    sensor_index: Tensor
    metadata: tuple[dict[str, str], ...]


def _read_manifest(path: str | Path) -> tuple[Path, list[dict[str, object]]]:
    manifest = Path(path)
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("PARrad manifest must contain at least one recording")
    for row in rows:
        if "path" not in row:
            raise ValueError("every PARrad manifest row requires path")
        for field in _METADATA_FIELDS:
            if not str(row.get(field, "")).strip():
                raise ValueError(f"every PARrad manifest row requires {field}")
    return manifest, rows


def build_sensor_vocabulary(*manifest_paths: str | Path) -> dict[str, int]:
    """Build one stable sensor vocabulary shared by every already-separated split."""
    sensors: set[str] = set()
    for path in manifest_paths:
        _, rows = _read_manifest(path)
        sensors.update(str(row["sensor_id"]) for row in rows)
    if not sensors:
        raise ValueError("sensor vocabulary cannot be empty")
    return {sensor: index for index, sensor in enumerate(sorted(sensors))}


def _validate(sample: PARradSample, config: PARradConfig) -> None:
    lengths = (sample.rd.shape[0], sample.md.shape[0], sample.labels.shape[0], sample.timestamps.shape[0])
    if len(set(lengths)) != 1:
        raise ValueError("RD, MD, labels, and timestamps must have identical frame counts")
    if lengths[0] == 0:
        raise ValueError("PARrad recordings must contain at least one frame")
    expected_rd = (config.rd_channels, config.rd_height, config.rd_width)
    md_only_placeholder = config.encoder_type in {"causal_radmamba", "causal_md_gru"} and sample.rd.shape[1:] == (0, 0, 0)
    if sample.rd.ndim != 4 or (sample.rd.shape[1:] != expected_rd and not md_only_placeholder):
        raise ValueError(
            f"rd must have shape [T,{config.rd_channels},{config.rd_height},{config.rd_width}]"
        )
    if sample.md.ndim != 2 or sample.md.shape[1] != config.md_bins:
        raise ValueError(f"md must have shape [T,{config.md_bins}]")
    if sample.labels.ndim != 1 or sample.timestamps.ndim != 1:
        raise ValueError("labels and timestamps must be one-dimensional")
    if not torch.isfinite(sample.rd).all() or not torch.isfinite(sample.md).all():
        raise ValueError("RD and MD values must be finite")
    if not torch.isfinite(sample.timestamps).all():
        raise ValueError("timestamps must be finite")
    if sample.timestamps.numel() > 1 and not torch.all(sample.timestamps[1:] > sample.timestamps[:-1]):
        raise ValueError("timestamps must be strictly increasing")


class PARradManifestDataset(Dataset[PARradSample]):
    """Load canonical NPZ recordings referenced by a JSONL manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        config: PARradConfig | None = None,
        sensor_vocabulary: Mapping[str, int] | None = None,
    ) -> None:
        self.manifest, self.rows = _read_manifest(manifest_path)
        self.config = PARradConfig() if config is None else config
        self.sensor_vocabulary = (
            build_sensor_vocabulary(self.manifest) if sensor_vocabulary is None else dict(sensor_vocabulary)
        )
        unknown = sorted({str(row["sensor_id"]) for row in self.rows} - set(self.sensor_vocabulary))
        if unknown:
            raise ValueError(f"unknown sensor_id values: {unknown}")
        indices = sorted(self.sensor_vocabulary.values())
        if indices != list(range(len(indices))):
            raise ValueError("sensor vocabulary indices must be contiguous from zero")
        if len(indices) > self.config.sensor_vocab_size:
            raise ValueError("sensor vocabulary exceeds configured sensor_vocab_size")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> PARradSample:
        row = self.rows[index]
        path = Path(str(row["path"]))
        if not path.is_absolute():
            path = self.manifest.parent / path
        with np.load(path, allow_pickle=False) as arrays:
            missing = {"rd", "md", "labels", "timestamps"} - set(arrays.files)
            if missing:
                raise ValueError(f"canonical PARrad NPZ is missing fields: {sorted(missing)}")
            md = torch.from_numpy(np.asarray(arrays["md"], dtype=np.float32))
            if self.config.encoder_type in {"causal_radmamba", "causal_md_gru"}:
                rd = torch.empty((md.shape[0], 0, 0, 0), dtype=torch.float32)
            else:
                rd = torch.from_numpy(np.asarray(arrays["rd"], dtype=np.float32))
            labels = torch.from_numpy(np.asarray(arrays["labels"], dtype=np.int64))
            timestamps = torch.from_numpy(np.asarray(arrays["timestamps"], dtype=np.float64))
        sensor_id = str(row["sensor_id"])
        sample = PARradSample(
            rd=rd,
            md=md,
            labels=labels,
            timestamps=timestamps,
            sensor_index=self.sensor_vocabulary[sensor_id],
            subject_id=str(row["subject_id"]),
            session_id=str(row["session_id"]),
            sensor_id=sensor_id,
            environment_id=str(row["environment_id"]),
        )
        _validate(sample, self.config)
        return sample


def mask_label_boundaries(labels: Tensor, radius_frames: int, ignore_index: int = -100) -> Tensor:
    """Mask an equal number of supervised frames on both sides of each label transition."""
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if radius_frames < 0:
        raise ValueError("label boundary radius must be non-negative")
    masked = labels.clone()
    if radius_frames == 0 or labels.numel() < 2:
        return masked
    supervised_transition = (
        labels[1:].ne(labels[:-1])
        & labels[1:].ne(ignore_index)
        & labels[:-1].ne(ignore_index)
    )
    for transition in supervised_transition.nonzero().flatten().add(1).tolist():
        start = max(0, transition - radius_frames)
        stop = min(labels.numel(), transition + radius_frames)
        masked[start:stop] = ignore_index
    return masked


class PARradWindowDataset(Dataset[PARradSample]):
    """Cache recordings and expose either fixed or single-label bounded windows."""

    def __init__(
        self,
        recordings: PARradManifestDataset,
        window_frames: int,
        window_stride: int | None = None,
        label_boundary_ignore_frames: int = 0,
        window_mode: str = "fixed",
        cached_recordings: Sequence[PARradSample] | None = None,
        min_label_run_frames: int = 1,
    ) -> None:
        stride = window_frames if window_stride is None else window_stride
        if window_frames <= 0 or stride <= 0:
            raise ValueError("window_frames and window_stride must be positive")
        if window_frames % recordings.config.frames_per_update:
            raise ValueError("window_frames must be a multiple of frames_per_update")
        if stride > window_frames:
            raise ValueError("window_stride cannot exceed window_frames")
        if window_mode not in {"fixed", "label_run"}:
            raise ValueError("window_mode must be 'fixed' or 'label_run'")
        if min_label_run_frames <= 0 or min_label_run_frames > window_frames:
            raise ValueError("min_label_run_frames must be between 1 and window_frames")
        if window_mode == "label_run" and stride != window_frames:
            raise ValueError("label_run window_mode requires non-overlapping window_stride")
        loaded = (
            tuple(recordings[index] for index in range(len(recordings)))
            if cached_recordings is None
            else tuple(cached_recordings)
        )
        if len(loaded) != len(recordings):
            raise ValueError("cached_recordings must match the manifest recording count")
        if window_mode == "fixed":
            self.recordings = (
                loaded
                if label_boundary_ignore_frames == 0
                else tuple(
                    replace(
                        sample,
                        labels=mask_label_boundaries(sample.labels, label_boundary_ignore_frames),
                    )
                    for sample in loaded
                )
            )
            self.windows = tuple(
                (recording_index, start, min(start + window_frames, sample.rd.shape[0]))
                for recording_index, sample in enumerate(self.recordings)
                for start in range(0, sample.rd.shape[0], stride)
                if sample.labels[start : min(start + window_frames, sample.rd.shape[0])].ne(-100).any()
            )
        else:
            self.recordings = loaded
            windows: list[tuple[int, int, int]] = []
            for recording_index, sample in enumerate(self.recordings):
                labels = sample.labels
                run_start = 0
                while run_start < labels.numel():
                    label = int(labels[run_start])
                    run_stop = run_start + 1
                    while run_stop < labels.numel() and int(labels[run_stop]) == label:
                        run_stop += 1
                    if label != -100:
                        start = run_start
                        stop = run_stop
                        if run_start > 0 and int(labels[run_start - 1]) not in {-100, label}:
                            start += label_boundary_ignore_frames
                        if run_stop < labels.numel() and int(labels[run_stop]) not in {-100, label}:
                            stop -= label_boundary_ignore_frames
                        run_frames = max(0, stop - start)
                        if run_frames >= min_label_run_frames:
                            if min_label_run_frames == 1:
                                chunk_sizes = [
                                    min(window_frames, stop - chunk_start)
                                    for chunk_start in range(start, stop, window_frames)
                                ]
                            else:
                                chunk_count = math.ceil(run_frames / window_frames)
                                if chunk_count > 1 and run_frames // chunk_count >= min_label_run_frames:
                                    base, extra = divmod(run_frames, chunk_count)
                                    chunk_sizes = [base + (index < extra) for index in range(chunk_count)]
                                else:
                                    full, remainder = divmod(run_frames, window_frames)
                                    chunk_sizes = [window_frames] * full
                                    if remainder >= min_label_run_frames:
                                        chunk_sizes.append(remainder)
                            chunk_start = start
                            for chunk_size in chunk_sizes:
                                windows.append((recording_index, chunk_start, chunk_start + chunk_size))
                                chunk_start += chunk_size
                    run_start = run_stop
            self.windows = tuple(windows)

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, index: int) -> PARradSample:
        recording_index, start, stop = self.windows[index]
        sample = self.recordings[recording_index]
        return PARradSample(
            rd=sample.rd[start:stop],
            md=sample.md[start:stop],
            labels=sample.labels[start:stop],
            timestamps=sample.timestamps[start:stop],
            sensor_index=sample.sensor_index,
            subject_id=sample.subject_id,
            session_id=sample.session_id,
            sensor_id=sample.sensor_id,
            environment_id=sample.environment_id,
        )


def parrad_collate(samples: Sequence[PARradSample], frames_per_update: int = 2) -> PARradBatch:
    """Pad recordings to a whole number of updates and build an explicit mask."""
    if not samples:
        raise ValueError("cannot collate an empty PARrad batch")
    if frames_per_update <= 0:
        raise ValueError("frames_per_update must be positive")
    spatial = samples[0].rd.shape[1:]
    md_bins = samples[0].md.shape[1]
    if any(sample.rd.shape[1:] != spatial or sample.md.shape[1] != md_bins for sample in samples):
        raise ValueError("all PARrad samples in a batch must share RD and MD shapes")
    frames = math.ceil(max(sample.rd.shape[0] for sample in samples) / frames_per_update) * frames_per_update
    batch = len(samples)
    rd = samples[0].rd.new_zeros((batch, frames, *spatial))
    md = samples[0].md.new_zeros((batch, frames, md_bins))
    labels = torch.full((batch, frames), -100, dtype=torch.long)
    timestamps = samples[0].timestamps.new_zeros((batch, frames))
    frame_mask = torch.zeros((batch, frames), dtype=torch.bool)
    metadata = []
    for index, sample in enumerate(samples):
        time = sample.rd.shape[0]
        rd[index, :time] = sample.rd
        md[index, :time] = sample.md
        labels[index, :time] = sample.labels
        timestamps[index, :time] = sample.timestamps
        frame_mask[index, :time] = True
        metadata.append(
            {
                "subject_id": sample.subject_id,
                "session_id": sample.session_id,
                "sensor_id": sample.sensor_id,
                "environment_id": sample.environment_id,
            }
        )
    return PARradBatch(
        rd=rd,
        md=md,
        labels=labels,
        timestamps=timestamps,
        frame_mask=frame_mask,
        sensor_index=torch.tensor([sample.sensor_index for sample in samples], dtype=torch.long),
        metadata=tuple(metadata),
    )


def assert_disjoint_splits(
    *,
    train: Sequence[Mapping[str, object]],
    validation: Sequence[Mapping[str, object]],
    test: Sequence[Mapping[str, object]],
) -> None:
    """Reject subject or session leakage across split metadata."""
    groups = {"train": train, "validation": validation, "test": test}
    names = tuple(groups)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1 :]:
            first, second = groups[first_name], groups[second_name]
            subject_overlap = {str(row["subject_id"]) for row in first} & {
                str(row["subject_id"]) for row in second
            }
            if subject_overlap:
                raise ValueError(f"subject overlap between {first_name} and {second_name}: {sorted(subject_overlap)}")
            session_overlap = {
                (str(row["subject_id"]), str(row["session_id"])) for row in first
            } & {
                (str(row["subject_id"]), str(row["session_id"])) for row in second
            }
            if session_overlap:
                raise ValueError(f"session overlap between {first_name} and {second_name}: {sorted(session_overlap)}")
