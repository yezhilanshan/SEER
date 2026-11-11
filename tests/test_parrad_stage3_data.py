from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from continuous_mmwave.data_parrad import PARradSample
from continuous_mmwave.parrad_config import PARradConfig
from continuous_mmwave.parrad_stage3_data import (
    CacheFingerprint,
    PhysicalSequenceDataset,
    PhysicalTransitionDataset,
    SemanticPairDataset,
    Stage3CacheItem,
    Stage3CacheRecord,
    VerifyTrajectoryDataset,
    audit_stage3_manifests,
    build_stage3_cache_for_recording,
    extract_event_runs,
    gaussian_boundary_targets,
    load_stage3_cache,
    collate_physical_sequence_batches,
    write_stage3_cache,
)


def test_extract_event_runs_does_not_bridge_ignored_gap() -> None:
    labels = torch.tensor([0, 0, -100, -100, 1, 1])
    times = torch.arange(6, dtype=torch.float64) * 0.09

    runs = extract_event_runs(labels, times)

    assert [(run.label, run.start_frame, run.stop_frame) for run in runs] == [
        (0, 0, 2),
        (1, 4, 6),
    ]
    assert runs[1].boundary_from_previous is None


def test_extract_event_runs_records_direct_transition_time() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    times = torch.tensor([0.0, 0.09, 0.18, 0.27], dtype=torch.float64)

    runs = extract_event_runs(labels, times)

    assert runs[1].boundary_from_previous == pytest.approx(0.18)


def test_recording_edges_are_not_boundaries() -> None:
    runs = extract_event_runs(torch.tensor([2, 2]), torch.tensor([0.0, 0.09]))

    assert runs[0].boundary_from_previous is None


@pytest.mark.parametrize(
    ("labels", "timestamps", "message"),
    [
        (torch.empty(0, dtype=torch.long), torch.empty(0), "at least one frame"),
        (torch.zeros((1, 1), dtype=torch.long), torch.zeros(1), "one-dimensional"),
        (torch.zeros(2, dtype=torch.long), torch.zeros(1), "same length"),
        (torch.zeros(2, dtype=torch.long), torch.tensor([0.0, float("nan")]), "finite"),
        (torch.zeros(2, dtype=torch.long), torch.tensor([0.1, 0.0]), "strictly increasing"),
    ],
)
def test_extract_event_runs_rejects_invalid_inputs(
    labels: torch.Tensor, timestamps: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        extract_event_runs(labels, timestamps)


def test_gaussian_boundary_targets_use_seconds_and_mask_invalid_updates() -> None:
    update_times = torch.tensor([0.09, 0.27, 0.45], dtype=torch.float64)

    target, valid = gaussian_boundary_targets(
        update_times,
        boundary_times=torch.tensor([0.27], dtype=torch.float64),
        sigma_seconds=0.36,
        valid_mask=torch.tensor([True, True, False]),
    )

    assert target[1] == pytest.approx(1.0)
    assert target[0] == pytest.approx(math.exp(-(0.18**2) / (2 * 0.36**2)))
    assert target[2] == 0
    assert valid.tolist() == [True, True, False]


def test_gaussian_boundary_targets_without_boundaries_are_zero() -> None:
    target, valid = gaussian_boundary_targets(
        torch.tensor([0.09, 0.27], dtype=torch.float64),
        boundary_times=torch.empty(0, dtype=torch.float64),
        sigma_seconds=0.36,
    )

    assert target.tolist() == [0.0, 0.0]
    assert valid.tolist() == [True, True]


def _write_audit_split(root: Path, split: str, subject: str) -> Path:
    split_root = root / split
    split_root.mkdir()
    recording = split_root / "recording.npz"
    labels = np.array([0, 0, 1, 1, -100, 2, 2], dtype=np.int64)
    timestamps = np.arange(labels.size, dtype=np.float64) * 0.09
    np.savez_compressed(
        recording,
        rd=np.zeros((labels.size, 1, 93, 128), dtype=np.float32),
        md=np.zeros((labels.size, 128), dtype=np.float32),
        labels=labels,
        timestamps=timestamps,
    )
    row = {
        "path": recording.name,
        "subject_id": subject,
        "session_id": f"{subject}-session",
        "sensor_id": "60ghz",
        "environment_id": "lab",
    }
    manifest = split_root / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest


def test_audit_stage3_manifests_counts_runs_transitions_and_ignored_gaps(tmp_path: Path) -> None:
    manifests = {
        split: _write_audit_split(tmp_path, split, f"subject-{index}")
        for index, split in enumerate(("train", "validation", "test"))
    }

    audits = audit_stage3_manifests(manifests, PARradConfig())

    assert set(audits) == {"train", "validation", "test"}
    for audit in audits.values():
        assert audit.recordings == 1
        assert audit.valid_runs == 3
        assert audit.direct_transitions == 1
        assert audit.ignored_gaps == 1
        assert audit.class_run_counts == {0: 1, 1: 1, 2: 1}
        assert audit.sensor_transition_counts == {"60ghz": 1}
        assert audit.duration_seconds == pytest.approx((0.63,))


def test_audit_stage3_manifests_rejects_split_without_direct_transition(tmp_path: Path) -> None:
    manifests = {
        split: _write_audit_split(tmp_path, split, f"subject-{index}")
        for index, split in enumerate(("train", "validation", "test"))
    }
    validation_path = Path(manifests["validation"])
    row = json.loads(validation_path.read_text(encoding="utf-8"))
    recording = validation_path.parent / row["path"]
    with np.load(recording) as arrays:
        payload = {name: arrays[name] for name in arrays.files}
    payload["labels"] = np.zeros_like(payload["labels"])
    np.savez_compressed(recording, **payload)

    with pytest.raises(ValueError, match="validation split has zero direct transitions"):
        audit_stage3_manifests(manifests, PARradConfig())


def _cache_record() -> Stage3CacheRecord:
    return Stage3CacheRecord(
        radar_tokens=np.ones((3, 4, 8), np.float16),
        radar_mask=np.ones((3, 4), np.bool_),
        frame_labels=np.array([[0, 0], [0, 0], [1, 1]], np.int64),
        timestamps=np.array([[0.0, 0.09], [0.18, 0.27], [0.36, 0.45]], np.float64),
        oracle_keys=np.ones((3, 6), np.float16),
        oracle_maturity=np.array([0.5, 1.0, 1.0], np.float32),
        oracle_run_id=np.array([0, 0, 1], np.int64),
    )


def test_stage3_cache_round_trip_and_fingerprint(tmp_path: Path) -> None:
    fingerprint = CacheFingerprint("checkpoint", "config", "manifest")
    write_stage3_cache(tmp_path, "recording", _cache_record(), fingerprint, {"subject_id": "s1"})

    loaded = load_stage3_cache(tmp_path, "recording", fingerprint)

    assert loaded.radar_tokens.shape == (3, 4, 8)
    assert loaded.radar_tokens.dtype == np.float16
    with pytest.raises(ValueError, match="fingerprint"):
        load_stage3_cache(
            tmp_path,
            "recording",
            replace(fingerprint, checkpoint_sha256="other"),
        )


def test_stage3_cache_metadata_is_deterministic(tmp_path: Path) -> None:
    fingerprint = CacheFingerprint("checkpoint", "config", "manifest")
    write_stage3_cache(tmp_path, "recording", _cache_record(), fingerprint, {"subject_id": "s1"})
    first = (tmp_path / "recording.json").read_bytes()

    write_stage3_cache(tmp_path, "recording", _cache_record(), fingerprint, {"subject_id": "s1"})

    assert (tmp_path / "recording.json").read_bytes() == first


class _FakeRadarEncoder:
    def __init__(self) -> None:
        self.config = SimpleNamespace(frames_per_update=2, radar_tokens_per_update=4)
        self.update_indices: list[int] = []

    def forward_update(self, rd, md, sensor_index, state, frame_mask):
        update_index = 0 if state is None else state.update_index
        self.update_indices.append(update_index)
        tokens = torch.full((1, 4, 8), float(update_index + 1))
        return SimpleNamespace(
            radar_tokens=tokens,
            radar_token_mask=torch.ones((1, 4), dtype=torch.bool),
            state=SimpleNamespace(update_index=update_index + 1),
        )


class _FakeEventFormer:
    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((batch, 2, 6))

    def build_prompt_cache(self, batch: int):
        return None

    def stream_update(self, previous, radar, maturity, radar_mask, prompt_cache):
        return SimpleNamespace(event_tokens=previous + 1)


class _FakeStage2:
    def __init__(self) -> None:
        self.radar_encoder = _FakeRadarEncoder()
        self.event_former = _FakeEventFormer()
        self.semantic_projection = lambda hidden: hidden.mean(dim=1)


def test_cache_builder_carries_encoder_state_and_resets_oracle_event_state_only_at_runs(
    tmp_path: Path,
) -> None:
    module = _FakeStage2()
    labels = torch.tensor([0, 0, 0, 0, 1, 1])
    sample = PARradSample(
        rd=torch.zeros((6, 1, 1, 1)),
        md=torch.zeros((6, 8)),
        labels=labels,
        timestamps=torch.arange(6, dtype=torch.float64) * 0.09,
        sensor_index=0,
        subject_id="s1",
        session_id="session1",
        sensor_id="60ghz",
        environment_id="lab",
    )

    cache = build_stage3_cache_for_recording(
        module,
        sample,
        CacheFingerprint("checkpoint", "config", "manifest"),
        tmp_path,
    )

    assert module.radar_encoder.update_indices == [0, 1, 2]
    assert cache.oracle_run_id.tolist() == [0, 0, 1]
    assert cache.oracle_maturity.tolist() == pytest.approx([0.5, 1.0, 1.0])
    assert cache.oracle_keys[:, 0].tolist() == pytest.approx([1.0, 2.0, 1.0])


def _dataset_item(key: str, subject: str, offset: float = 0.0) -> Stage3CacheItem:
    keys = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.9, 0.1, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.1, 0.9, 0.0, 0.0],
        ],
        dtype=np.float16,
    )
    record = Stage3CacheRecord(
        radar_tokens=(np.arange(4 * 4 * 8).reshape(4, 4, 8) + offset).astype(np.float16),
        radar_mask=np.ones((4, 4), dtype=np.bool_),
        frame_labels=np.array([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int64),
        timestamps=(np.arange(8).reshape(4, 2) * 0.09 + offset).astype(np.float64),
        oracle_keys=keys,
        oracle_maturity=np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float32),
        oracle_run_id=np.array([0, 0, 1, 1], dtype=np.int64),
    )
    return Stage3CacheItem(
        split="train",
        key=key,
        metadata={
            "subject_id": subject,
            "session_id": f"session-{subject}",
            "sensor_id": "60ghz",
            "environment_id": "lab",
        },
        record=record,
    )


def test_semantic_pairs_are_balanced_deterministic_and_split_local() -> None:
    records = (_dataset_item("a", "s1"), _dataset_item("b", "s2", 1.0))

    first = SemanticPairDataset(records, seed=42)
    second = SemanticPairDataset(records, seed=42)

    assert first.pair_index == second.pair_index
    assert len(first) == 16
    assert Counter(target for _, _, target, _ in first.pair_index) == {0: 8, 1: 8}
    anchor_counts = Counter(active for active, _, _, _ in first.pair_index)
    assert len(anchor_counts) == 4
    assert set(anchor_counts.values()) == {4}
    assert all(
        first.key_refs[a].split == first.key_refs[b].split
        for a, b, _, _ in first.pair_index
    )
    assert all(
        first.key_refs[active].subject_id != first.key_refs[candidate].subject_id
        for active, candidate, _, kind in first.pair_index
        if kind != "same_event_nascent"
    )


def test_verify_trajectories_include_false_proposals_and_short_candidates() -> None:
    dataset = VerifyTrajectoryDataset(
        (_dataset_item("a", "s1"), _dataset_item("b", "s2", 1.0)), seed=42
    )

    kinds = {sample.kind for sample in dataset.samples}

    assert {"true_boundary", "shifted", "within_event", "short_spike"} <= kinds
    batch = dataset[0]
    assert batch.active.shape == (4,)
    assert batch.key_history.shape == (3, 4)


def test_physical_dataset_uses_consecutive_updates_and_marks_oracle_reset() -> None:
    dataset = PhysicalTransitionDataset((_dataset_item("a", "s1"),), sigma_seconds=0.36)

    sample = dataset[1]

    assert len(dataset) == 3
    assert sample.previous.shape == sample.current.shape == (4, 8)
    assert sample.reset.item() is True
    assert sample.local_target.item() == pytest.approx(1.0)


def test_physical_sequences_keep_event_history_through_the_next_boundary() -> None:
    item = _dataset_item("a", "s1")

    dataset = PhysicalSequenceDataset((item,), sigma_seconds=0.36)

    assert len(dataset) == 2
    first = dataset[0]
    torch.testing.assert_close(
        first.tokens,
        torch.from_numpy(item.record.radar_tokens[:3].astype(np.float32)),
    )
    assert first.valid.tolist() == [True, True]
    assert first.local_target[-1].item() == pytest.approx(1.0)


def test_physical_sequence_collate_masks_padding_from_loss() -> None:
    dataset = PhysicalSequenceDataset((_dataset_item("a", "s1"),), sigma_seconds=0.36)

    batch = collate_physical_sequence_batches([dataset[0], dataset[1]])

    assert batch.tokens.shape == (2, 3, 4, 8)
    assert batch.valid.tolist() == [[True, True], [True, False]]
    assert batch.weight[1, 1].item() == 0.0
