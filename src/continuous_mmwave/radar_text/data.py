"""Timestamp-based windows and explicitly weak caption supervision."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from continuous_mmwave.data_parrad import PARradManifestDataset, build_sensor_vocabulary
from continuous_mmwave.parrad_config import PARradConfig


def collapse_actions(labels) -> list[int]:
    actions = []
    for value in labels:
        label = int(value)
        if label < 0:
            raise ValueError("unknown labels cannot be bridged into a continuous caption")
        if not actions or actions[-1] != label:
            actions.append(label)
    if not actions:
        raise ValueError("empty action sequence")
    return actions


def caption_from_actions(actions: list[int], phrases: dict[int, str]) -> str:
    if not actions:
        raise ValueError("empty action sequence")
    words = [phrases[action].strip().rstrip(".") for action in actions]
    if len(words) == 1:
        body = words[0]
    elif len(words) == 2:
        body = ", then ".join(words)
    else:
        body = ", then ".join(words[:-1]) + ", and then " + words[-1]
    return f"The person {body}."


def load_phrases(path: str | Path) -> dict[int, str]:
    """JSON object class ID -> third-person verb phrase, e.g. 'walks'."""
    raw = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    phrases = {int(key): str(value).strip() for key, value in raw.items()}
    if not phrases or sorted(phrases) != list(range(len(phrases))):
        raise ValueError("phrase IDs must be contiguous from zero and non-empty")
    if any(not value for value in phrases.values()) or len(set(phrases.values())) != len(phrases):
        raise ValueError("phrases must be non-empty and unique")
    return phrases


class OfflineCaptionDataset(Dataset):
    """Canonical recordings, indexed without retaining all recordings in RAM.

    captions JSONL optionally overrides a window using its stable sample_id.
    Phrase-derived actions remain evaluation references even with a human caption.
    """

    def __init__(self, manifest, phrases, *, window_seconds=12.0, stride_seconds=6.0,
                 min_seconds=8.0, max_actions=3, captions=None):
        if not 0 < min_seconds <= window_seconds or not 0 < stride_seconds <= window_seconds:
            raise ValueError("invalid window/stride/min seconds")
        if max_actions < 1:
            raise ValueError("max_actions must be positive")
        vocabulary = build_sensor_vocabulary(manifest)
        self.recordings = PARradManifestDataset(
            manifest, PARradConfig(sensor_vocab_size=len(vocabulary)), vocabulary,
        )
        self.phrases = phrases
        self.overrides = {}
        if captions:
            for line in Path(captions).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    key, text = row["sample_id"], row["caption"].strip()
                    if key in self.overrides or not text:
                        raise ValueError("duplicate or empty manual caption")
                    self.overrides[key] = text
        self.windows = []
        sample_ids = set()
        self.skipped = {"duration": 0, "unknown": 0, "actions": 0}
        self._cached_index, self._cached_recording = None, None
        for index, row in enumerate(self.recordings.rows):
            identity = str(row.get("recording_id", row["path"])).replace("\\", "/")
            recording_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
            path = Path(row["path"])
            if not path.is_absolute():
                path = self.recordings.manifest.parent / path
            with np.load(path, allow_pickle=False) as arrays:
                labels, times = arrays["labels"], arrays["timestamps"]
            if times.ndim != 1 or labels.shape != times.shape or len(times) < 2:
                raise ValueError("window indexing requires aligned labels and at least two timestamps")
            if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
                raise ValueError("timestamps must be finite and strictly increasing")
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("labels must be integer class IDs")
            if not set(np.unique(labels)).issubset(set(phrases) | {-100}):
                raise ValueError("label IDs are missing from phrase map")
            period = float(np.median(np.diff(times)))
            # Irregular/gapped recordings must be repaired explicitly, not silently stitched.
            if not np.allclose(np.diff(times), period, rtol=0.05, atol=1e-6):
                raise ValueError("recording timestamps contain gaps or irregular frame spacing")
            for begin in np.arange(times[0], times[-1] + period - 1e-7, stride_seconds):
                start = int(np.searchsorted(times, begin - 1e-7))
                stop = int(np.searchsorted(times, begin + window_seconds - 1e-7))
                if stop <= start or times[stop - 1] - times[start] + period < min_seconds - 1e-6:
                    self.skipped["duration"] += 1
                    continue
                segment = labels[start:stop]
                if np.any(segment < 0):
                    self.skipped["unknown"] += 1
                    continue
                actions = collapse_actions(segment)
                if len(actions) > max_actions:
                    self.skipped["actions"] += 1
                    continue
                sample_id = f"{row['subject_id']}/{row['session_id']}/{row['sensor_id']}/{recording_id}:{start}:{stop}"
                if sample_id in sample_ids:
                    raise ValueError("duplicate recording/window identity in manifest")
                sample_ids.add(sample_id)
                self.windows.append((index, start, stop, sample_id, actions))
        if not self.windows:
            raise ValueError(f"no eligible caption windows; skipped={self.skipped}")
        unknown_overrides = set(self.overrides) - {window[3] for window in self.windows}
        if unknown_overrides:
            raise ValueError(f"manual captions refer to unavailable windows: {sorted(unknown_overrides)[:3]}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, index):
        recording_index, start, stop, sample_id, actions = self.windows[index]
        if self._cached_index != recording_index:
            self._cached_recording = self.recordings[recording_index]
            self._cached_index = recording_index
        recording = self._cached_recording
        return {
            "rd": recording.rd[start:stop], "md": recording.md[start:stop],
            "labels": recording.labels[start:stop], "timestamps": recording.timestamps[start:stop],
            "caption": self.overrides.get(sample_id, caption_from_actions(actions, self.phrases)),
            "actions": list(actions), "sample_id": sample_id,
            "subject_id": recording.subject_id,
        }


def collate_windows(samples):
    if not samples:
        raise ValueError("empty batch")
    lengths = torch.tensor([len(sample["rd"]) for sample in samples])
    rd = torch.zeros(len(samples), int(lengths.max()), *samples[0]["rd"].shape[1:])
    md = torch.zeros(len(samples), int(lengths.max()), samples[0]["md"].shape[1])
    labels = torch.full(rd.shape[:2], -100, dtype=torch.long)
    for index, sample in enumerate(samples):
        length = int(lengths[index])
        rd[index, :length], md[index, :length] = sample["rd"], sample["md"]
        labels[index, :length] = sample["labels"]
    return dict(rd=rd, md=md, lengths=lengths, labels=labels,
                captions=[s["caption"] for s in samples], actions=[s["actions"] for s in samples],
                sample_ids=[s["sample_id"] for s in samples], subjects=[s["subject_id"] for s in samples])


class RecordingWindowSampler(Sampler):
    """Shuffle recordings and their windows while retaining one-recording cache locality."""

    def __init__(self, dataset):
        self.groups = {}
        for index, window in enumerate(dataset.windows):
            self.groups.setdefault(window[0], []).append(index)
        self.size = len(dataset.windows)

    def __len__(self):
        return self.size

    def __iter__(self):
        groups = list(self.groups.values())
        for group_index in torch.randperm(len(groups)).tolist():
            group = groups[group_index]
            for position in torch.randperm(len(group)).tolist():
                yield group[position]
