"""Continuous-recording supervision and cache contracts for PARrad Stage P3."""

from __future__ import annotations

import json
import hashlib
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from continuous_mmwave.data_parrad import (
    PARradManifestDataset,
    PARradSample,
    assert_disjoint_splits,
    build_sensor_vocabulary,
)
from continuous_mmwave.parrad_config import PARradConfig


@dataclass(frozen=True)
class EventRun:
    """One half-open constant-label run from a continuous recording."""

    label: int
    start_frame: int
    stop_frame: int
    start_time: float
    stop_time: float
    boundary_from_previous: float | None


@dataclass(frozen=True)
class TransitionAudit:
    """Aggregate transition-supervision statistics for one split."""

    recordings: int
    valid_runs: int
    direct_transitions: int
    ignored_gaps: int
    class_run_counts: dict[int, int]
    sensor_transition_counts: dict[str, int]
    duration_seconds: tuple[float, ...]


@dataclass(frozen=True)
class CacheFingerprint:
    checkpoint_sha256: str
    config_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class Stage3CacheRecord:
    radar_tokens: np.ndarray
    radar_mask: np.ndarray
    frame_labels: np.ndarray
    timestamps: np.ndarray
    oracle_keys: np.ndarray
    oracle_maturity: np.ndarray
    oracle_run_id: np.ndarray


@dataclass(frozen=True)
class Stage3CacheItem:
    split: str
    key: str
    metadata: Mapping[str, str]
    record: Stage3CacheRecord


@dataclass(frozen=True)
class PhysicalBatch:
    previous: Tensor
    current: Tensor
    mask: Tensor
    local_target: Tensor
    radar_target: Tensor
    reset: Tensor


@dataclass(frozen=True)
class PhysicalSequenceBatch:
    """One or more padded event-local update sequences for physical supervision."""

    tokens: Tensor
    mask: Tensor
    local_target: Tensor
    radar_target: Tensor
    valid: Tensor
    weight: Tensor


@dataclass(frozen=True)
class SemanticPairBatch:
    active: Tensor
    candidate: Tensor
    maturity: Tensor
    target: Tensor


@dataclass(frozen=True)
class VerifyBatch:
    active: Tensor
    key_history: Tensor
    maturity: Tensor
    target: Tensor
    weight: Tensor


@dataclass(frozen=True)
class VerifyFeatureBatch:
    """Precomputed closed-loop inputs for the three-feature Verify head."""

    inputs: Tensor
    target: Tensor
    weight: Tensor


@dataclass(frozen=True)
class SemanticKeyRef:
    split: str
    record_index: int
    update_index: int
    run_id: int
    label: int
    maturity: float
    subject_id: str
    session_id: str


@dataclass(frozen=True)
class VerifyTrajectorySample:
    active_key: np.ndarray
    key_history: np.ndarray
    maturity: float
    target: float
    weight: float
    kind: str


@dataclass(frozen=True)
class VerifyFeatureSample:
    inputs: tuple[float, float, float]
    target: float
    weight: float
    kind: str
    recording_key: str = ""
    proposal_time: float = 0.0


def _validate_frame_sequence(labels: Tensor, timestamps: Tensor) -> None:
    if labels.ndim != 1 or timestamps.ndim != 1:
        raise ValueError("labels and timestamps must be one-dimensional")
    if labels.numel() != timestamps.numel():
        raise ValueError("labels and timestamps must have the same length")
    if labels.numel() == 0:
        raise ValueError("recordings must contain at least one frame")
    if not torch.isfinite(timestamps).all():
        raise ValueError("timestamps must be finite")
    if timestamps.numel() > 1 and not torch.all(timestamps[1:] > timestamps[:-1]):
        raise ValueError("timestamps must be strictly increasing")


def extract_event_runs(
    labels: Tensor,
    timestamps: Tensor,
    ignore_index: int = -100,
) -> tuple[EventRun, ...]:
    """Return valid constant-label runs; ignored gaps break adjacency."""

    _validate_frame_sequence(labels, timestamps)
    runs: list[EventRun] = []
    start = 0
    frame_count = labels.numel()
    while start < frame_count:
        label = int(labels[start].item())
        stop = start + 1
        while stop < frame_count and int(labels[stop].item()) == label:
            stop += 1
        if label != ignore_index:
            direct_previous = start > 0 and int(labels[start - 1].item()) != ignore_index
            runs.append(
                EventRun(
                    label=label,
                    start_frame=start,
                    stop_frame=stop,
                    start_time=float(timestamps[start].item()),
                    stop_time=float(timestamps[stop - 1].item()),
                    boundary_from_previous=(
                        float(timestamps[start].item()) if direct_previous else None
                    ),
                )
            )
        start = stop
    return tuple(runs)


def gaussian_boundary_targets(
    update_times: Tensor,
    boundary_times: Tensor,
    sigma_seconds: float,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Build nearest-boundary Gaussian targets in physical seconds."""

    if update_times.ndim != 1 or boundary_times.ndim != 1:
        raise ValueError("update_times and boundary_times must be one-dimensional")
    if not update_times.is_floating_point() or not boundary_times.is_floating_point():
        raise ValueError("update_times and boundary_times must be floating point")
    if not torch.isfinite(update_times).all() or not torch.isfinite(boundary_times).all():
        raise ValueError("boundary target timestamps must be finite")
    if not torch.isfinite(torch.tensor(sigma_seconds)) or sigma_seconds <= 0:
        raise ValueError("sigma_seconds must be positive and finite")
    if valid_mask is None:
        valid = torch.ones_like(update_times, dtype=torch.bool)
    else:
        if valid_mask.ndim != 1 or valid_mask.shape != update_times.shape:
            raise ValueError("valid_mask must match update_times")
        valid = valid_mask.to(device=update_times.device, dtype=torch.bool)
    if boundary_times.numel():
        boundaries = boundary_times.to(device=update_times.device, dtype=update_times.dtype)
        nearest = (update_times[:, None] - boundaries[None, :]).abs().min(dim=1).values
        target = torch.exp(-nearest.square() / (2 * sigma_seconds**2))
    else:
        target = torch.zeros_like(update_times)
    return torch.where(valid, target, torch.zeros_like(target)), valid


def _read_manifest_rows(path: str | Path) -> list[dict[str, object]]:
    manifest = Path(path)
    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{manifest} must contain at least one recording")
    return rows


def _count_ignored_gaps(labels: Tensor, ignore_index: int = -100) -> int:
    ignored = labels.eq(ignore_index)
    if not ignored.any():
        return 0
    starts = ignored.clone()
    starts[1:] &= ~ignored[:-1]
    return int(starts.sum().item())


def audit_stage3_manifests(
    manifests: Mapping[str, str | Path],
    config: PARradConfig,
) -> dict[str, TransitionAudit]:
    """Audit complete train/validation/test recordings without window resets."""

    required = {"train", "validation", "test"}
    if set(manifests) != required:
        raise ValueError("manifests must contain exactly train, validation, and test")
    rows = {name: _read_manifest_rows(manifests[name]) for name in required}
    assert_disjoint_splits(
        train=rows["train"],
        validation=rows["validation"],
        test=rows["test"],
    )
    vocabulary = build_sensor_vocabulary(*(manifests[name] for name in sorted(required)))
    audits: dict[str, TransitionAudit] = {}
    for split in ("train", "validation", "test"):
        dataset = PARradManifestDataset(manifests[split], config, vocabulary)
        class_counts: Counter[int] = Counter()
        sensor_counts: Counter[str] = Counter()
        valid_runs = 0
        direct_transitions = 0
        ignored_gaps = 0
        durations: list[float] = []
        for index in range(len(dataset)):
            sample = dataset[index]
            sample_runs = extract_event_runs(sample.labels, sample.timestamps)
            transitions = sum(run.boundary_from_previous is not None for run in sample_runs)
            class_counts.update(run.label for run in sample_runs)
            sensor_counts[sample.sensor_id] += transitions
            valid_runs += len(sample_runs)
            direct_transitions += transitions
            ignored_gaps += _count_ignored_gaps(sample.labels)
            durations.append(
                float(sample.timestamps[-1].item() - sample.timestamps[0].item())
                + config.frame_period_s
            )
        if direct_transitions == 0:
            raise ValueError(f"{split} split has zero direct transitions")
        audits[split] = TransitionAudit(
            recordings=len(dataset),
            valid_runs=valid_runs,
            direct_transitions=direct_transitions,
            ignored_gaps=ignored_gaps,
            class_run_counts=dict(sorted(class_counts.items())),
            sensor_transition_counts=dict(sorted(sensor_counts.items())),
            duration_seconds=tuple(durations),
        )
    all_classes = set().union(
        *(audit.class_run_counts.keys() for audit in audits.values())
    )
    missing_train = sorted(all_classes - audits["train"].class_run_counts.keys())
    if missing_train:
        raise ValueError(f"training split is missing valid semantic runs for classes: {missing_train}")
    return audits


_CACHE_SCHEMA_VERSION = 1
_CACHE_DTYPES = {
    "radar_tokens": np.dtype(np.float16),
    "radar_mask": np.dtype(np.bool_),
    "frame_labels": np.dtype(np.int64),
    "timestamps": np.dtype(np.float64),
    "oracle_keys": np.dtype(np.float16),
    "oracle_maturity": np.dtype(np.float32),
    "oracle_run_id": np.dtype(np.int64),
}


def _validate_cache_key(key: str) -> None:
    if not key or Path(key).name != key or key in {".", ".."}:
        raise ValueError("cache key must be one non-empty filename component")


def _validate_cache_record(record: Stage3CacheRecord) -> None:
    arrays = asdict(record)
    for name, expected_dtype in _CACHE_DTYPES.items():
        value = arrays[name]
        if not isinstance(value, np.ndarray) or value.dtype != expected_dtype:
            raise ValueError(f"{name} must be a NumPy array with dtype {expected_dtype}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values")
    updates = record.radar_tokens.shape[0]
    if record.radar_tokens.ndim != 3 or record.radar_mask.shape != record.radar_tokens.shape[:2]:
        raise ValueError("radar cache arrays must have shapes [U,R,D] and [U,R]")
    if record.frame_labels.ndim != 2 or record.timestamps.shape != record.frame_labels.shape:
        raise ValueError("frame labels and timestamps must share shape [U,F]")
    if record.oracle_keys.ndim != 2:
        raise ValueError("oracle_keys must have shape [U,D]")
    if record.oracle_maturity.ndim != 1 or record.oracle_run_id.ndim != 1:
        raise ValueError("oracle maturity and run id must have shape [U]")
    if any(
        value.shape[0] != updates
        for value in (
            record.frame_labels,
            record.timestamps,
            record.oracle_keys,
            record.oracle_maturity,
            record.oracle_run_id,
        )
    ):
        raise ValueError("all cache arrays must share the update dimension")
    if not 0 <= record.oracle_maturity.min(initial=0) or record.oracle_maturity.max(initial=0) > 1:
        raise ValueError("oracle maturity must be in [0,1]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_stage3_cache(
    root: Path,
    key: str,
    record: Stage3CacheRecord,
    fingerprint: CacheFingerprint,
    metadata: Mapping[str, str],
) -> None:
    """Atomically write one validated NPZ record and deterministic JSON sidecar."""

    _validate_cache_key(key)
    _validate_cache_record(record)
    if any(not value for value in asdict(fingerprint).values()):
        raise ValueError("cache fingerprint fields must be non-empty")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{key}.npz"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{key}.", suffix=".tmp", dir=root)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.savez_compressed(handle, **asdict(record))
        os.replace(temporary_name, npz_path)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    arrays = asdict(record)
    _atomic_json(
        root / f"{key}.json",
        {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "fingerprint": asdict(fingerprint),
            "metadata": dict(sorted(metadata.items())),
            "arrays": {
                name: {"shape": list(value.shape), "dtype": str(value.dtype)}
                for name, value in arrays.items()
            },
            "npz_sha256": _sha256(npz_path),
        },
    )


def load_stage3_cache(root: Path, key: str, expected: CacheFingerprint) -> Stage3CacheRecord:
    """Load one record, failing closed on schema, fingerprint, hash, shape, or dtype drift."""

    _validate_cache_key(key)
    root = Path(root)
    npz_path = root / f"{key}.npz"
    metadata_path = root / f"{key}.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != _CACHE_SCHEMA_VERSION:
        raise ValueError("unsupported Stage P3 cache schema")
    if payload.get("fingerprint") != asdict(expected):
        raise ValueError("Stage P3 cache fingerprint mismatch")
    if payload.get("npz_sha256") != _sha256(npz_path):
        raise ValueError("Stage P3 cache content hash mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        if set(arrays.files) != set(_CACHE_DTYPES):
            raise ValueError("Stage P3 cache array fields mismatch")
        record = Stage3CacheRecord(**{name: arrays[name].copy() for name in _CACHE_DTYPES})
    _validate_cache_record(record)
    described = payload.get("arrays", {})
    for name, value in asdict(record).items():
        if described.get(name) != {"shape": list(value.shape), "dtype": str(value.dtype)}:
            raise ValueError(f"Stage P3 cache metadata mismatch for {name}")
    return record


def _recording_cache_key(sample: PARradSample) -> str:
    raw = "__".join((sample.subject_id, sample.session_id, sample.sensor_id))
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in raw)


def _module_device(module: Any) -> torch.device:
    try:
        return next(module.parameters()).device
    except (AttributeError, StopIteration):
        return torch.device("cpu")


def _frame_run_ids(labels: Tensor, timestamps: Tensor) -> Tensor:
    ids = torch.full_like(labels, -1)
    for run_id, run in enumerate(extract_event_runs(labels, timestamps)):
        ids[run.start_frame : run.stop_frame] = run_id
    return ids


@torch.inference_mode()
def build_stage3_cache_for_recording(
    module: Any,
    sample: PARradSample,
    fingerprint: CacheFingerprint,
    output_root: Path,
) -> Stage3CacheRecord:
    """Extract frozen per-update features while preserving continuous encoder state."""

    step = int(module.radar_encoder.config.frames_per_update)
    if step <= 0:
        raise ValueError("frames_per_update must be positive")
    device = _module_device(module)
    frame_run_ids = _frame_run_ids(sample.labels, sample.timestamps)
    update_run_ids: list[int] = []
    for start in range(0, sample.labels.numel(), step):
        chunk_ids = frame_run_ids[start : start + step]
        valid_ids = chunk_ids[chunk_ids.ge(0)]
        update_run_ids.append(int(valid_ids[-1].item()) if valid_ids.numel() else -1)
    totals = Counter(run_id for run_id in update_run_ids if run_id >= 0)
    seen: Counter[int] = Counter()

    encoder_state = None
    event_state = None
    previous_run_id = -1
    prompt_cache = module.event_former.build_prompt_cache(1)
    radar_tokens: list[np.ndarray] = []
    radar_masks: list[np.ndarray] = []
    frame_labels: list[np.ndarray] = []
    timestamps: list[np.ndarray] = []
    oracle_keys: list[np.ndarray] = []
    oracle_maturity: list[float] = []
    period = (
        float(torch.median(sample.timestamps[1:] - sample.timestamps[:-1]).item())
        if sample.timestamps.numel() > 1
        else 0.0
    )
    for update_index, start in enumerate(range(0, sample.labels.numel(), step)):
        stop = min(start + step, sample.labels.numel())
        valid_frames = stop - start
        rd = sample.rd[start:stop]
        md = sample.md[start:stop]
        labels = sample.labels[start:stop]
        times = sample.timestamps[start:stop]
        if valid_frames < step:
            rd = torch.cat((rd, rd.new_zeros((step - valid_frames, *rd.shape[1:]))), dim=0)
            md = torch.cat((md, md.new_zeros((step - valid_frames, md.shape[1]))), dim=0)
            labels = torch.cat((labels, labels.new_full((step - valid_frames,), -100)), dim=0)
            extra_times = times[-1] + period * torch.arange(
                1, step - valid_frames + 1, dtype=times.dtype, device=times.device
            )
            times = torch.cat((times, extra_times), dim=0)
        frame_mask = torch.arange(step, device=device).unsqueeze(0) < valid_frames
        encoded = module.radar_encoder.forward_update(
            rd.unsqueeze(0).to(device),
            md.unsqueeze(0).to(device),
            sensor_index=torch.tensor([sample.sensor_index], dtype=torch.long, device=device),
            state=encoder_state,
            frame_mask=frame_mask,
        )
        encoder_state = encoded.state
        tokens = encoded.radar_tokens
        token_mask = encoded.radar_token_mask
        run_id = update_run_ids[update_index]
        if run_id >= 0:
            if event_state is None or run_id != previous_run_id:
                event_state = module.event_former.initial_state(1).to(device)
            seen[run_id] += 1
            maturity = seen[run_id] / totals[run_id]
            event_state = module.event_former.stream_update(
                event_state,
                tokens,
                maturity,
                token_mask,
                prompt_cache,
            ).event_tokens
            key = module.semantic_projection(event_state)
            previous_run_id = run_id
        else:
            maturity = 0.0
            semantic_dim = oracle_keys[0].shape[-1] if oracle_keys else 0
            if semantic_dim == 0:
                probe = module.semantic_projection(module.event_former.initial_state(1).to(device))
                semantic_dim = probe.shape[-1]
            key = torch.zeros((1, semantic_dim), device=device)
        radar_tokens.append(tokens[0].detach().float().cpu().numpy().astype(np.float16))
        radar_masks.append(token_mask[0].detach().cpu().numpy().astype(np.bool_))
        frame_labels.append(labels.cpu().numpy().astype(np.int64))
        timestamps.append(times.cpu().numpy().astype(np.float64))
        oracle_keys.append(key[0].detach().float().cpu().numpy().astype(np.float16))
        oracle_maturity.append(maturity)
    record = Stage3CacheRecord(
        radar_tokens=np.stack(radar_tokens),
        radar_mask=np.stack(radar_masks),
        frame_labels=np.stack(frame_labels),
        timestamps=np.stack(timestamps),
        oracle_keys=np.stack(oracle_keys),
        oracle_maturity=np.asarray(oracle_maturity, dtype=np.float32),
        oracle_run_id=np.asarray(update_run_ids, dtype=np.int64),
    )
    write_stage3_cache(
        Path(output_root),
        _recording_cache_key(sample),
        record,
        fingerprint,
        {
            "subject_id": sample.subject_id,
            "session_id": sample.session_id,
            "sensor_id": sample.sensor_id,
            "environment_id": sample.environment_id,
        },
    )
    return record


def _fingerprint_manifests(manifests: Mapping[str, str | Path]) -> str:
    digest = hashlib.sha256()
    for split in sorted(manifests):
        digest.update(split.encode("utf-8"))
        digest.update(Path(manifests[split]).read_bytes())
    return digest.hexdigest()


def build_stage3_cache(
    config: Any,
    stage2_checkpoint: Path,
    output_root: Path,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> Path:
    """Build a fingerprinted frozen Stage P2 feature cache for complete recordings."""

    from continuous_mmwave.checkpoint import load_training_checkpoint
    from continuous_mmwave.train import _build_parrad_stage2

    if not isinstance(config.model, PARradConfig):
        raise TypeError("Stage P3 cache requires PARradConfig")
    stage2_checkpoint = Path(stage2_checkpoint)
    if not stage2_checkpoint.is_file():
        raise FileNotFoundError(stage2_checkpoint)
    manifests = {
        split: Path(config.data[f"{split}_manifest"])
        for split in splits
    }
    if set(splits) == {"train", "validation", "test"}:
        audit_stage3_manifests(manifests, config.model)
    canonical_config = json.dumps(
        config.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    fingerprint = CacheFingerprint(
        checkpoint_sha256=_sha256(stage2_checkpoint),
        config_sha256=hashlib.sha256(canonical_config).hexdigest(),
        manifest_sha256=_fingerprint_manifests(manifests),
    )
    module, _ = _build_parrad_stage2(config)
    load_training_checkpoint(stage2_checkpoint, module)
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    requested_device = str(config.training.get("device", "auto"))
    device = torch.device(
        "cuda" if requested_device == "auto" and torch.cuda.is_available()
        else "cpu" if requested_device == "auto"
        else requested_device
    )
    module.to(device)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    vocabulary = build_sensor_vocabulary(*(manifests[split] for split in splits))
    index_rows: list[dict[str, str]] = []
    for split in splits:
        dataset = PARradManifestDataset(manifests[split], config.model, vocabulary)
        split_root = output_root / split
        for sample_index in range(len(dataset)):
            sample = dataset[sample_index]
            build_stage3_cache_for_recording(module, sample, fingerprint, split_root)
            index_rows.append(
                {"split": split, "key": _recording_cache_key(sample), "path": str(split_root)}
            )
    index_path = output_root / "index.json"
    _atomic_json(
        index_path,
        {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "fingerprint": asdict(fingerprint),
            "records": index_rows,
        },
    )
    return index_path


def _update_label(record: Stage3CacheRecord, update_index: int) -> int:
    labels = record.frame_labels[update_index]
    valid = labels[labels != -100]
    return int(valid[-1]) if valid.size else -100


class PhysicalTransitionDataset(Dataset[PhysicalBatch]):
    """Consecutive cached updates with time-based physical transition targets."""

    def __init__(self, records: tuple[Stage3CacheItem, ...], sigma_seconds: float = 0.36) -> None:
        if sigma_seconds <= 0:
            raise ValueError("sigma_seconds must be positive")
        self.records = tuple(records)
        self.sigma_seconds = sigma_seconds
        self.sample_index: list[tuple[int, int, float, bool]] = []
        for record_index, item in enumerate(self.records):
            record = item.record
            flat_labels = torch.from_numpy(record.frame_labels.reshape(-1))
            flat_times = torch.from_numpy(record.timestamps.reshape(-1))
            runs = extract_event_runs(flat_labels, flat_times)
            boundaries = torch.tensor(
                [run.boundary_from_previous for run in runs if run.boundary_from_previous is not None],
                dtype=torch.float64,
            )
            update_times = torch.from_numpy(record.timestamps[:, 0])
            targets, valid = gaussian_boundary_targets(
                update_times,
                boundaries,
                sigma_seconds,
                valid_mask=torch.from_numpy(record.frame_labels).ne(-100).any(dim=1),
            )
            for update_index in range(1, record.radar_tokens.shape[0]):
                mask = record.radar_mask[update_index - 1] & record.radar_mask[update_index]
                if valid[update_index] and mask.any():
                    run_id = int(record.oracle_run_id[update_index])
                    reset = run_id >= 0 and run_id != int(record.oracle_run_id[update_index - 1])
                    self.sample_index.append(
                        (record_index, update_index, float(targets[update_index].item()), reset)
                    )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> PhysicalBatch:
        record_index, update_index, target, reset = self.sample_index[index]
        record = self.records[record_index].record
        mask = record.radar_mask[update_index - 1] & record.radar_mask[update_index]
        return PhysicalBatch(
            previous=torch.from_numpy(record.radar_tokens[update_index - 1].astype(np.float32)),
            current=torch.from_numpy(record.radar_tokens[update_index].astype(np.float32)),
            mask=torch.from_numpy(mask.copy()),
            local_target=torch.tensor(target, dtype=torch.float32),
            radar_target=torch.tensor(target, dtype=torch.float32),
            reset=torch.tensor(reset, dtype=torch.bool),
        )


def collate_physical_batches(samples: list[PhysicalBatch]) -> PhysicalBatch:
    if not samples:
        raise ValueError("cannot collate an empty physical batch")
    return PhysicalBatch(
        previous=torch.stack([sample.previous for sample in samples]),
        current=torch.stack([sample.current for sample in samples]),
        mask=torch.stack([sample.mask for sample in samples]),
        local_target=torch.stack([sample.local_target for sample in samples]),
        radar_target=torch.stack([sample.radar_target for sample in samples]),
        reset=torch.stack([sample.reset for sample in samples]),
    )


class PhysicalSequenceDataset(Dataset[PhysicalSequenceBatch]):
    """Event-local chronological sequences ending at the next direct boundary."""

    def __init__(
        self,
        records: tuple[Stage3CacheItem, ...],
        sigma_seconds: float = 0.36,
        positive_weight: float | None = None,
    ) -> None:
        if sigma_seconds <= 0:
            raise ValueError("sigma_seconds must be positive")
        if positive_weight is not None and positive_weight <= 0:
            raise ValueError("positive_weight must be positive")
        self.records = tuple(records)
        self.sigma_seconds = sigma_seconds
        self.sequence_index: list[tuple[int, tuple[int, ...], tuple[float, ...]]] = []
        target_values: list[float] = []
        for record_index, item in enumerate(self.records):
            record = item.record
            flat_labels = torch.from_numpy(record.frame_labels.reshape(-1))
            flat_times = torch.from_numpy(record.timestamps.reshape(-1))
            runs = extract_event_runs(flat_labels, flat_times)
            boundaries = torch.tensor(
                [run.boundary_from_previous for run in runs if run.boundary_from_previous is not None],
                dtype=torch.float64,
            )
            targets, valid_updates = gaussian_boundary_targets(
                torch.from_numpy(record.timestamps[:, 0]),
                boundaries,
                sigma_seconds,
                valid_mask=torch.from_numpy(record.frame_labels).ne(-100).any(dim=1),
            )
            run_ids = record.oracle_run_id
            start = 0
            while start < len(run_ids):
                run_id = int(run_ids[start])
                stop = start + 1
                while stop < len(run_ids) and int(run_ids[stop]) == run_id:
                    stop += 1
                if run_id >= 0:
                    update_indices = list(range(start, stop))
                    if stop < len(run_ids) and int(run_ids[stop]) >= 0:
                        update_indices.append(stop)
                    transition_indices = update_indices[1:]
                    if transition_indices:
                        if not all(bool(valid_updates[index]) for index in transition_indices):
                            raise ValueError("physical sequence contains an invalid supervised update")
                        if not all(record.radar_mask[index].any() for index in update_indices):
                            raise ValueError("physical sequence contains an all-masked update")
                        sequence_targets = tuple(float(targets[index]) for index in transition_indices)
                        self.sequence_index.append(
                            (record_index, tuple(update_indices), sequence_targets)
                        )
                        target_values.extend(sequence_targets)
                start = stop
        if not self.sequence_index:
            raise ValueError("physical sequence construction requires valid transitions")
        binary_positive = sum(value >= 0.5 for value in target_values)
        binary_negative = len(target_values) - binary_positive
        if positive_weight is None:
            self.positive_weight = max(1.0, binary_negative / max(1, binary_positive))
        else:
            self.positive_weight = float(positive_weight)

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, index: int) -> PhysicalSequenceBatch:
        record_index, update_indices, target_values = self.sequence_index[index]
        record = self.records[record_index].record
        targets = torch.tensor(target_values, dtype=torch.float32)
        weights = 1 + (self.positive_weight - 1) * targets
        return PhysicalSequenceBatch(
            tokens=torch.from_numpy(record.radar_tokens[list(update_indices)].astype(np.float32)),
            mask=torch.from_numpy(record.radar_mask[list(update_indices)].copy()),
            local_target=targets,
            radar_target=targets.clone(),
            valid=torch.ones_like(targets, dtype=torch.bool),
            weight=weights,
        )


def collate_physical_sequence_batches(
    samples: list[PhysicalSequenceBatch],
) -> PhysicalSequenceBatch:
    if not samples:
        raise ValueError("cannot collate an empty physical sequence batch")
    max_updates = max(sample.tokens.shape[0] for sample in samples)
    batch_size = len(samples)
    token_shape = samples[0].tokens.shape[1:]
    tokens = samples[0].tokens.new_empty((batch_size, max_updates, *token_shape))
    mask = samples[0].mask.new_empty((batch_size, max_updates, *samples[0].mask.shape[1:]))
    local_target = samples[0].local_target.new_zeros((batch_size, max_updates - 1))
    radar_target = samples[0].radar_target.new_zeros((batch_size, max_updates - 1))
    valid = torch.zeros((batch_size, max_updates - 1), dtype=torch.bool)
    weight = samples[0].weight.new_zeros((batch_size, max_updates - 1))
    for batch_index, sample in enumerate(samples):
        updates = sample.tokens.shape[0]
        transitions = updates - 1
        tokens[batch_index, :updates] = sample.tokens
        mask[batch_index, :updates] = sample.mask
        if updates < max_updates:
            tokens[batch_index, updates:] = sample.tokens[-1]
            mask[batch_index, updates:] = sample.mask[-1]
        local_target[batch_index, :transitions] = sample.local_target
        radar_target[batch_index, :transitions] = sample.radar_target
        valid[batch_index, :transitions] = sample.valid
        weight[batch_index, :transitions] = sample.weight
    return PhysicalSequenceBatch(tokens, mask, local_target, radar_target, valid, weight)


@dataclass(frozen=True)
class _SemanticEvent:
    split: str
    label: int
    subject_id: str
    session_id: str
    nascent_ref: int
    mature_ref: int
    all_refs: tuple[int, ...]


def _semantic_events(
    records: tuple[Stage3CacheItem, ...],
) -> tuple[list[SemanticKeyRef], list[_SemanticEvent]]:
    refs: list[SemanticKeyRef] = []
    events: list[_SemanticEvent] = []
    for record_index, item in enumerate(records):
        record = item.record
        for run_id in sorted(set(record.oracle_run_id.tolist()) - {-1}):
            update_indices = np.flatnonzero(record.oracle_run_id == run_id).tolist()
            event_refs: list[int] = []
            for update_index in update_indices:
                event_refs.append(len(refs))
                refs.append(
                    SemanticKeyRef(
                        split=item.split,
                        record_index=record_index,
                        update_index=update_index,
                        run_id=run_id,
                        label=_update_label(record, update_index),
                        maturity=float(record.oracle_maturity[update_index]),
                        subject_id=item.metadata["subject_id"],
                        session_id=item.metadata["session_id"],
                    )
                )
            if event_refs:
                events.append(
                    _SemanticEvent(
                        split=item.split,
                        label=refs[event_refs[-1]].label,
                        subject_id=item.metadata["subject_id"],
                        session_id=item.metadata["session_id"],
                        nascent_ref=event_refs[0],
                        mature_ref=event_refs[-1],
                        all_refs=tuple(event_refs),
                    )
                )
    return refs, events


class SemanticPairDataset(Dataset[SemanticPairBatch]):
    """Balanced deterministic same/different semantic-key pairs within each split."""

    def __init__(self, records: tuple[Stage3CacheItem, ...], seed: int = 42) -> None:
        self.records = tuple(records)
        self.key_refs, events = _semantic_events(self.records)
        generator = np.random.default_rng(seed)
        self.pair_index: list[tuple[int, int, int, str]] = []
        for split in sorted({event.split for event in events}):
            split_events = [event for event in events if event.split == split]
            if len(split_events) < 2:
                continue
            order = generator.permutation(len(split_events)).tolist()
            for anchor_index in order:
                anchor = split_events[anchor_index]
                same_candidates = [
                    event
                    for event in split_events
                    if event is not anchor and event.label == anchor.label
                ]
                different = [event for event in split_events if event.label != anchor.label]
                preference = lambda event: (
                    event.subject_id == anchor.subject_id,
                    event.session_id == anchor.session_id,
                    event.subject_id,
                    event.session_id,
                    event.mature_ref,
                )
                same_candidates.sort(key=preference)
                different.sort(key=preference)
                if not same_candidates or not different:
                    continue
                same = same_candidates[0]
                other = different[0]
                self.pair_index.extend(
                    [
                        (anchor.mature_ref, anchor.nascent_ref, 0, "same_event_nascent"),
                        (anchor.mature_ref, same.mature_ref, 0, "same_class_mature"),
                        (anchor.mature_ref, other.nascent_ref, 1, "different_nascent"),
                        (anchor.mature_ref, other.mature_ref, 1, "different_mature"),
                    ]
                )
        if not self.pair_index:
            raise ValueError("semantic pair construction requires same- and different-class events")

    def __len__(self) -> int:
        return len(self.pair_index)

    def _key(self, ref_index: int) -> Tensor:
        ref = self.key_refs[ref_index]
        value = self.records[ref.record_index].record.oracle_keys[ref.update_index]
        return torch.from_numpy(value.astype(np.float32))

    def __getitem__(self, index: int) -> SemanticPairBatch:
        active_ref, candidate_ref, target, _ = self.pair_index[index]
        maturity = self.key_refs[candidate_ref].maturity
        return SemanticPairBatch(
            active=self._key(active_ref),
            candidate=self._key(candidate_ref),
            maturity=torch.tensor(maturity, dtype=torch.float32),
            target=torch.tensor(float(target), dtype=torch.float32),
        )


def collate_semantic_batches(samples: list[SemanticPairBatch]) -> SemanticPairBatch:
    if not samples:
        raise ValueError("cannot collate an empty semantic batch")
    return SemanticPairBatch(
        active=torch.stack([sample.active for sample in samples]),
        candidate=torch.stack([sample.candidate for sample in samples]),
        maturity=torch.stack([sample.maturity for sample in samples]),
        target=torch.stack([sample.target for sample in samples]),
    )


def _three_key_history(keys: np.ndarray) -> np.ndarray:
    if len(keys) >= 3:
        return keys[-3:].astype(np.float32)
    padding = np.repeat(keys[:1], 3 - len(keys), axis=0)
    return np.concatenate((padding, keys), axis=0).astype(np.float32)


class VerifyTrajectoryDataset(Dataset[VerifyBatch]):
    """Deterministic true/false Candidate trajectories for monotonic Verify training."""

    def __init__(self, records: tuple[Stage3CacheItem, ...], seed: int = 42) -> None:
        del seed
        self.records = tuple(records)
        self.samples: list[VerifyTrajectorySample] = []
        for item in self.records:
            record = item.record
            run_ids = [run_id for run_id in sorted(set(record.oracle_run_id.tolist())) if run_id >= 0]
            run_keys = {
                run_id: record.oracle_keys[record.oracle_run_id == run_id].astype(np.float32)
                for run_id in run_ids
            }
            for run_id, keys in run_keys.items():
                mature = keys[-1]
                self.samples.append(
                    VerifyTrajectorySample(
                        active_key=mature.copy(),
                        key_history=_three_key_history(keys),
                        maturity=1.0,
                        target=0.0,
                        weight=1.0,
                        kind="within_event",
                    )
                )
                if len(run_ids) > 1:
                    other_id = next(candidate for candidate in run_ids if candidate != run_id)
                    spike = np.stack((run_keys[other_id][0], mature), axis=0)
                    self.samples.append(
                        VerifyTrajectorySample(
                            active_key=mature.copy(),
                            key_history=_three_key_history(spike),
                            maturity=2 / 3,
                            target=0.0,
                            weight=1.5,
                            kind="short_spike",
                        )
                    )
            for old_id, new_id in zip(run_ids, run_ids[1:], strict=False):
                active = run_keys[old_id][-1]
                candidate = run_keys[new_id]
                history = _three_key_history(candidate)
                self.samples.append(
                    VerifyTrajectorySample(
                        active_key=active.copy(),
                        key_history=history,
                        maturity=1.0,
                        target=1.0,
                        weight=1.0,
                        kind="true_boundary",
                    )
                )
                self.samples.append(
                    VerifyTrajectorySample(
                        active_key=active.copy(),
                        key_history=_three_key_history(np.stack((active, candidate[0]))),
                        maturity=0.5,
                        target=0.0,
                        weight=1.0,
                        kind="shifted",
                    )
                )
        if not self.samples:
            raise ValueError("Verify trajectories require at least one valid event")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VerifyBatch:
        sample = self.samples[index]
        return VerifyBatch(
            active=torch.from_numpy(sample.active_key.copy()),
            key_history=torch.from_numpy(sample.key_history.copy()),
            maturity=torch.tensor(sample.maturity, dtype=torch.float32),
            target=torch.tensor(sample.target, dtype=torch.float32),
            weight=torch.tensor(sample.weight, dtype=torch.float32),
        )


def collate_verify_batches(samples: list[VerifyBatch]) -> VerifyBatch:
    if not samples:
        raise ValueError("cannot collate an empty Verify batch")
    return VerifyBatch(
        active=torch.stack([sample.active for sample in samples]),
        key_history=torch.stack([sample.key_history for sample in samples]),
        maturity=torch.stack([sample.maturity for sample in samples]),
        target=torch.stack([sample.target for sample in samples]),
        weight=torch.stack([sample.weight for sample in samples]),
    )


class VerifyFeatureDataset(Dataset[VerifyFeatureBatch]):
    """Tensor dataset for Verify inputs mined from the online controller."""

    def __init__(self, samples: tuple[VerifyFeatureSample, ...]) -> None:
        if not samples:
            raise ValueError("closed-loop Verify dataset requires at least one sample")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> VerifyFeatureBatch:
        sample = self.samples[index]
        return VerifyFeatureBatch(
            inputs=torch.tensor(sample.inputs, dtype=torch.float32),
            target=torch.tensor(sample.target, dtype=torch.float32),
            weight=torch.tensor(sample.weight, dtype=torch.float32),
        )


def collate_verify_feature_batches(
    samples: list[VerifyFeatureBatch],
) -> VerifyFeatureBatch:
    if not samples:
        raise ValueError("cannot collate an empty closed-loop Verify batch")
    return VerifyFeatureBatch(
        inputs=torch.stack([sample.inputs for sample in samples]),
        target=torch.stack([sample.target for sample in samples]),
        weight=torch.stack([sample.weight for sample in samples]),
    )
