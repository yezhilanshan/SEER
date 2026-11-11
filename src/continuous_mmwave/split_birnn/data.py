"""MD preprocessing and approximate command-clock audits, independent of HDF5."""
from __future__ import annotations

import numpy as np

CLASSES = ('walk', 'fall', 'stand_up', 'sit_down', 'get_in_bed', 'lie_in_bed',
           'roll_in_bed', 'sit_in_bed', 'get_out_bed')
COMMANDS = {
    'walk to bed': 0, 'walk to chair': 0, 'walk to room': 0,
    'fall on floor of room': 1, 'stand up from bed': 2, 'stand up from chair': 2,
    'stand up from floor of room': 2, 'sit down on bed': 3, 'sit down on chair': 3,
    'get in bed': 4, 'lie in bed': 5, 'roll in bed': 6, 'sit in bed': 7, 'get out bed': 8,
}


def merge_labels(labels: np.ndarray, id_to_command: dict[int, str]) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('labels must be a one-dimensional integer array')
    result = np.full(labels.shape, -100, dtype=np.int64)
    for key in np.unique(labels):
        if key == -100:
            continue
        command = ' '.join(id_to_command.get(int(key), '').split())
        if command not in COMMANDS:
            raise ValueError(f'unknown command or label ID: {key}, {command!r}')
        result[labels == key] = COMMANDS[command]
    return result


def standardize_frames(md: np.ndarray) -> np.ndarray:
    md = np.asarray(md, dtype=np.float64)
    if md.ndim != 2 or md.shape[1] != 128 or not np.isfinite(md).all():
        raise ValueError('MD must be finite [T,128]')
    centered = md - md.mean(axis=-1, keepdims=True)
    scale = np.maximum(md.std(axis=-1, keepdims=True), 1e-6)
    return (centered / scale).astype(np.float32)


def audit_labels(labels: np.ndarray) -> dict:
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer) or len(labels) == 0:
        raise ValueError('nonempty integer labels required')
    valid = labels != -100
    if np.any((labels[valid] < 0) | (labels[valid] >= 9)):
        raise ValueError('labels outside the nine-class vocabulary')
    starts = np.r_[True, labels[1:] != labels[:-1]]
    runs = labels[starts & valid]
    transitions = valid[1:] & valid[:-1] & (labels[1:] != labels[:-1])
    return {'frame_count': len(labels), 'valid_frame_count': int(valid.sum()),
            'ignored_frame_count': int((~valid).sum()), 'ignored_fraction': float((~valid).mean()),
            'transition_count': int(transitions.sum()),
            'class_frame_counts': np.bincount(labels[valid], minlength=9).tolist(),
            'class_run_counts': np.bincount(runs, minlength=9).tolist()}


def align_commands(frame_count: int, offsets_s: np.ndarray, command_ids: np.ndarray, *,
                   frame_period_s: float = .09, frame_origin_s: float = 0.) -> tuple[np.ndarray, dict]:
    offsets_s = np.asarray(offsets_s, dtype=np.float64)
    command_ids = np.asarray(command_ids)
    if (frame_count <= 0 or not np.isfinite(frame_period_s) or frame_period_s <= 0
            or not np.isfinite(frame_origin_s) or offsets_s.ndim != 1 or len(offsets_s) == 0
            or command_ids.shape != offsets_s.shape or not np.isfinite(offsets_s).all()
            or not np.issubdtype(command_ids.dtype, np.integer)
            or np.any(np.diff(offsets_s) <= 0) or np.any((command_ids < 0) | (command_ids >= 9))):
        raise ValueError('invalid frame clock or non-increasing command sequence')
    times = frame_origin_s + frame_period_s * np.arange(frame_count, dtype=np.float64)
    # Float tolerance only handles representation at exact grid boundaries (sub-nanosecond).
    active = np.searchsorted(offsets_s, times + 1e-10, side='right') - 1
    y = np.full(frame_count, -100, dtype=np.int64)
    valid = active >= 0
    y[valid] = command_ids[active[valid]]
    indices = np.searchsorted(times + 1e-10, offsets_s).astype(np.int64)
    end = frame_origin_s + frame_period_s * frame_count
    audit = audit_labels(y)
    audit.update({'command_frame_indices': indices.tolist(), 'command_indices_monotonic': bool(np.all(np.diff(indices) >= 0)),
                  'collapsed_command_boundaries': int((np.diff(indices) == 0).sum()),
                  'commands_before_start': int((offsets_s < frame_origin_s - 1e-10).sum()),
                  'commands_at_or_after_end': int((offsets_s >= end - 1e-10).sum()),
                  'unlabeled_prefix_frames': int(np.argmax(valid)) if valid.any() else frame_count,
                  'unlabeled_tail_frames': 0 if valid.any() else frame_count,
                  'last_command_tail_is_extrapolated': bool(valid.any() and end > offsets_s[-1]),
                  'unconfirmed_extrapolated_tail_s': float(max(0., end-offsets_s[-1])) if valid.any() else 0.,
                  'csv_end_to_recording_end_signed_s': float(end-offsets_s[-1]),
                  'csv_span_s': float(offsets_s[-1] - offsets_s[0]),
                  'frame_span_s': float(frame_period_s * (frame_count - 1)),
                  'frame_coverage_duration_s': float(frame_period_s * frame_count),
                  'last_command_to_end_s': float(end - offsets_s[-1]),
                  'frame_origin_relative_to_first_command_s': float(frame_origin_s - offsets_s[0]),
                  'frame_period_s': frame_period_s, 'clock_calibrated': False,
                  'physical_action_onset_available': False,
                  'label_policy': 'command_active_until_next_command_with_extrapolated_tail',
                  'origin_policy': 'explicit_approximation_not_measured_sync'})
    return y, audit


def build_folds(rows: list[dict], *, seed: int = 42) -> list[dict]:
    ids = [r['recording_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate recording identity')
    subjects = sorted({r['subject_id'] for r in rows})
    if len(subjects) < 5:
        raise ValueError('at least five subjects required')
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(subjects)
    groups = [sorted(g.tolist()) for g in np.array_split(shuffled, 5)]
    folds = []
    for i, test in enumerate(groups):
        remaining = [s for s in shuffled.tolist() if s not in test]
        val_size = max(1, int(np.ceil(len(remaining) * .2)))
        validation = sorted(remaining[:val_size])
        train = sorted(set(remaining) - set(validation))
        f = {'fold': i, 'seed': seed, 'grouping': 'subject', 'train_subjects': train,
             'validation_subjects': validation, 'test_subjects': test}
        for role, selected in (('train', train), ('validation', validation), ('test', test)):
            f[role + '_recordings'] = sorted(r['recording_id'] for r in rows if r['subject_id'] in selected)
        folds.append(f)
    return folds
