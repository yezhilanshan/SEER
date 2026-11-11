"""Convert audited PARrad HDF5 recordings into the canonical RD/MD format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np


IGNORE_INDEX = -100
LABEL_POLICY = "command_active_until_next_command"


@dataclass(frozen=True)
class CommandAnnotation:
    timestamp: datetime
    command: str


def _normalize_command(value: str) -> str:
    return " ".join(value.split())


def read_command_annotations(path: str | Path) -> tuple[CommandAnnotation, ...]:
    """Read command boundaries and fail closed on malformed or unordered rows."""
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not {"timestamp", "command"}.issubset(reader.fieldnames):
            raise ValueError("command CSV requires timestamp and command columns")
        annotations = []
        for row_number, row in enumerate(reader, start=2):
            timestamp_text = str(row.get("timestamp", "")).strip()
            command = _normalize_command(str(row.get("command", "")))
            if not timestamp_text or not command:
                raise ValueError(f"command CSV row {row_number} has an empty timestamp or command")
            try:
                timestamp = datetime.fromisoformat(timestamp_text)
            except ValueError as error:
                raise ValueError(f"command CSV row {row_number} has an invalid ISO timestamp") from error
            annotations.append(CommandAnnotation(timestamp=timestamp, command=command))

    if not annotations:
        raise ValueError("command CSV must contain at least one annotation")
    if any(current.timestamp <= previous.timestamp for previous, current in zip(annotations, annotations[1:])):
        raise ValueError("command timestamps must be strictly increasing")
    return tuple(annotations)


def assign_frame_labels(
    frame_timestamps: np.ndarray,
    command_offsets_s: np.ndarray,
    commands: Sequence[str],
    label_to_id: dict[str, int],
) -> np.ndarray:
    """Assign each frame the latest command, keeping any leading prefix ignored."""
    frame_timestamps = np.asarray(frame_timestamps, dtype=np.float64)
    command_offsets_s = np.asarray(command_offsets_s, dtype=np.float64)
    if frame_timestamps.ndim != 1 or command_offsets_s.ndim != 1:
        raise ValueError("frame timestamps and command offsets must be one-dimensional")
    if len(command_offsets_s) != len(commands):
        raise ValueError("command offsets and command names must have identical lengths")
    if len(command_offsets_s) == 0:
        raise ValueError("at least one command boundary is required")
    if np.any(np.diff(frame_timestamps) <= 0) or np.any(np.diff(command_offsets_s) <= 0):
        raise ValueError("frame timestamps and command offsets must be strictly increasing")
    if not np.isfinite(frame_timestamps).all() or not np.isfinite(command_offsets_s).all():
        raise ValueError("frame timestamps and command offsets must be finite")

    try:
        command_ids = np.asarray([label_to_id[command] for command in commands], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"command is missing from label vocabulary: {error.args[0]}") from error
    active = np.searchsorted(command_offsets_s, frame_timestamps, side="right") - 1
    labels = np.full(frame_timestamps.shape, IGNORE_INDEX, dtype=np.int64)
    supervised = active >= 0
    labels[supervised] = command_ids[active[supervised]]
    return labels


def _label_map_payload(label_to_id: dict[str, int]) -> dict[str, object]:
    ordered = sorted(label_to_id.items(), key=lambda item: item[1])
    return {
        "version": 1,
        "ignore_index": IGNORE_INDEX,
        "classes": [{"id": label_id, "command": command} for command, label_id in ordered],
    }


def _load_or_create_label_map(output_dir: Path, commands: Sequence[str]) -> tuple[dict[str, int], bool]:
    path = output_dir / "label_map.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        classes = payload.get("classes", [])
        label_to_id = {str(item["command"]): int(item["id"]) for item in classes}
        if sorted(label_to_id.values()) != list(range(len(label_to_id))):
            raise ValueError("existing label_map.json IDs must be contiguous from zero")
        unknown = sorted(set(commands) - set(label_to_id))
        if unknown:
            raise ValueError(f"commands are missing from existing label_map.json: {unknown}")
        return label_to_id, False
    unique_commands = sorted(set(commands))
    return {command: index for index, command in enumerate(unique_commands)}, True


def _atomic_write_text(path: Path, text: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def _sensor_paths(handle: h5py.File) -> list[tuple[str, h5py.Dataset, h5py.Dataset]]:
    if "radar" not in handle or not isinstance(handle["radar"], h5py.Group):
        raise ValueError("PARrad HDF5 is missing the /radar group")
    sensors = []
    radar = handle["radar"]
    for sensor_id in sorted(radar):
        group = radar[sensor_id]
        if not isinstance(group, h5py.Group) or "rdn" not in group or "mDoppler" not in group:
            continue
        rdn, md = group["rdn"], group["mDoppler"]
        if not isinstance(rdn, h5py.Dataset) or not isinstance(md, h5py.Dataset):
            raise ValueError(f"sensor {sensor_id} RD/MD entries must be datasets")
        if rdn.ndim != 3 or rdn.shape[1:] != (93, 128):
            raise ValueError(f"sensor {sensor_id} rdn must have shape [T,93,128]")
        if md.ndim != 2 or md.shape[1:] != (128,):
            raise ValueError(f"sensor {sensor_id} mDoppler must have shape [T,128]")
        if rdn.shape[0] != md.shape[0] or rdn.shape[0] == 0:
            raise ValueError(f"sensor {sensor_id} RD and MD frame counts must match and be non-zero")
        sensors.append((sensor_id, rdn, md))
    if not sensors:
        raise ValueError("PARrad HDF5 contains no sensors with rdn and mDoppler datasets")
    return sensors


def convert_recording(
    h5_path: str | Path,
    command_csv: str | Path,
    output_dir: str | Path,
    *,
    subject_id: str,
    session_id: str,
    environment_id: str,
    frame_period_s: float = 0.09,
    recording_start: datetime | str | None = None,
    overwrite: bool = False,
    md_only: bool = False,
) -> Path:
    """Convert every audited radar sensor in one PARrad session and return its manifest."""
    if frame_period_s <= 0:
        raise ValueError("frame_period_s must be positive")
    metadata = {"subject_id": subject_id, "session_id": session_id, "environment_id": environment_id}
    if any(not value.strip() for value in metadata.values()):
        raise ValueError("subject_id, session_id, and environment_id must be non-empty")

    h5_path = Path(h5_path).resolve()
    command_csv = Path(command_csv).resolve()
    output = Path(output_dir).resolve()
    recordings_dir = output / "recordings"
    recordings_dir.mkdir(parents=True, exist_ok=True)
    annotations = read_command_annotations(command_csv)
    if recording_start is None:
        start = annotations[0].timestamp
    elif isinstance(recording_start, str):
        start = datetime.fromisoformat(recording_start)
    else:
        start = recording_start
    command_offsets = np.asarray(
        [(annotation.timestamp - start).total_seconds() for annotation in annotations], dtype=np.float64
    )
    commands = tuple(annotation.command for annotation in annotations)
    label_to_id, write_label_map = _load_or_create_label_map(output, commands)

    manifest = output / "manifest.jsonl"
    if manifest.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing manifest: {manifest}")

    pending: list[tuple[Path, Path]] = []
    rows = []
    try:
        with h5py.File(h5_path, "r") as handle:
            for sensor_id, rdn_dataset, md_dataset in _sensor_paths(handle):
                filename = f"{subject_id}-{session_id}-{sensor_id}.npz"
                destination = recordings_dir / filename
                if destination.exists() and not overwrite:
                    raise FileExistsError(f"refusing to overwrite existing recording: {destination}")
                timestamps = np.arange(rdn_dataset.shape[0], dtype=np.float64) * frame_period_s
                labels = assign_frame_labels(timestamps, command_offsets, commands, label_to_id)
                if not np.any(labels != IGNORE_INDEX):
                    raise ValueError(f"sensor {sensor_id} has no supervised frames after command alignment")
                md = np.asarray(md_dataset, dtype=np.float32)
                rd = (
                    np.empty((md.shape[0], 0, 0, 0), dtype=np.float32)
                    if md_only
                    else np.asarray(rdn_dataset, dtype=np.float32)[:, np.newaxis, :, :]
                )
                if not np.isfinite(rd).all() or not np.isfinite(md).all():
                    raise ValueError(f"sensor {sensor_id} contains non-finite RD or MD values")

                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{filename}.", suffix=".tmp", dir=recordings_dir
                )
                os.close(descriptor)
                temporary = Path(temporary_name)
                with temporary.open("wb") as npz_handle:
                    np.savez(npz_handle, rd=rd, md=md, labels=labels, timestamps=timestamps)
                pending.append((temporary, destination))
                rows.append(
                    {
                        "path": (Path("recordings") / filename).as_posix(),
                        **metadata,
                        "sensor_id": sensor_id,
                        "source_h5": str(h5_path),
                        "source_commands": str(command_csv),
                        "frame_period_s": frame_period_s,
                        "label_policy": LABEL_POLICY,
                    }
                )

        for temporary, destination in pending:
            os.replace(temporary, destination)
        if write_label_map:
            _atomic_write_text(
                output / "label_map.json",
                json.dumps(_label_map_payload(label_to_id), ensure_ascii=False, indent=2) + "\n",
            )
        _atomic_write_text(
            manifest,
            "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        )
    except BaseException:
        for temporary, _ in pending:
            temporary.unlink(missing_ok=True)
        raise
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--h5", required=True, type=Path, help="audited processed_data.h5")
    parser.add_argument("--commands", required=True, type=Path, help="timestamp_speech.csv")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--subject-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--environment-id", default="parrad")
    parser.add_argument("--frame-period-s", type=float, default=0.09)
    parser.add_argument("--recording-start", help="optional ISO timestamp; defaults to the first command")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    manifest = convert_recording(
        args.h5,
        args.commands,
        args.output_dir,
        subject_id=args.subject_id,
        session_id=args.session_id,
        environment_id=args.environment_id,
        frame_period_s=args.frame_period_s,
        recording_start=args.recording_start,
        overwrite=args.overwrite,
    )
    print(manifest)


if __name__ == "__main__":
    main()
