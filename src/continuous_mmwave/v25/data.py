"""Locked PARrad roles and causal MD-statistic proxy supervision.

Manifest construction reads filesystem identity only, never radar/label arrays.
Proxy statistics describe radar values, not body height, velocity or pose.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np


DATASET_V2 = {
    'fit': {**{f'subject_{s}': ('set000', 'set001') for s in ('02', '03', '04', '08')},
            'subject_13': ('set000',)},
    'dev_regression': {f'subject_{s}': ('set002',) for s in ('02', '03', '04')},
    'dev_new': {'subject_08': ('set002',)},
    'session_holdout': {f'subject_{s}': ('set003',) for s in ('02', '03', '04', '08')},
    'subject_holdout': {'subject_12': ('set000', 'set001', 'set002', 'set003')},
}
SENSORS = ('TI_xWR14xx', 'TI_xWR68xx')


def fixed_role(subject, session, specification=DATASET_V2):
    if subject in ('subject_05', 'subject_06'):
        raise ValueError('protected official subject')
    roles = [role for role, assignments in specification.items()
             if session in assignments.get(subject, ())]
    if len(roles) != 1:
        raise ValueError(f'no unique locked role for {subject}/{session}')
    return roles[0]


def build_training_manifest(rows, specification=DATASET_V2):
    splits = {role: [] for role in specification}
    identities, duplicates = {}, []
    for source in rows:
        item = dict(source)
        path = Path(item['path']).resolve()
        if '.part' in path.name or not path.is_file():
            raise ValueError(f'part or missing source: {path}')
        role = fixed_role(item['subject_id'], item['session_id'], specification)
        if item.get('role', role) != role:
            raise ValueError('claimed role differs from locked role')
        if item['sensor_id'] not in SENSORS:
            raise ValueError('unsupported sensor identity')
        identity = (item['subject_id'], item['session_id'], item['sensor_id'])
        # A hard-link alias is also one physical source, not an extra sample.
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino) if stat.st_ino else str(path)
        if key in identities:
            previous = identities[key]
            if previous['identity'] != identity:
                raise ValueError('source aliases disagree on identity')
            if previous['source'] != item:
                raise ValueError('duplicate source metadata disagree on identity')
            duplicates.append(dict(path=str(path), role=role))
            continue
        identities[key] = dict(identity=identity, source=dict(item))
        item.update(path=str(path), role=role)
        splits[role].append(item)
    splits['dev'] = splits['dev_regression'] + splits['dev_new']
    return dict(splits=splits, duplicates=duplicates)


def require_array_access(row, *, allowed_roles, terminal_locked=False):
    role = row['role']
    expected = fixed_role(row['subject_id'], row['session_id'])
    if role != expected and not (role == 'dev' and expected in ('dev_new', 'dev_regression')):
        raise ValueError('row role does not match locked partition')
    if role.endswith('holdout') and not terminal_locked:
        raise ValueError('holdout arrays remain locked')
    if role not in allowed_roles:
        raise ValueError(f'array access forbidden for role {role}')


def build_evidence_targets(md, *, normalization=None):
    values = np.asarray(md, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 128 or not len(values) or not np.isfinite(values).all():
        raise ValueError('finite nonempty [frames,128] MD required')
    targets = np.zeros((len(values), 4), dtype=np.float64)
    mask = np.ones_like(targets, dtype=bool)
    targets[:, 0] = np.abs(values).mean(axis=1)
    targets[:, 1] = values.std(axis=1)
    mask[0, 2:] = False
    if len(values) > 1:
        targets[1:, 2] = np.abs(values[1:] - values[:-1]).mean(axis=1)
        norms = np.linalg.norm(values, axis=1)
        valid = (norms[1:] > 1e-12) & (norms[:-1] > 1e-12)
        mask[1:, 3] = valid
        denominator = norms[1:][valid] * norms[:-1][valid]
        similarity = (values[1:][valid] * values[:-1][valid]).sum(axis=1) / denominator
        targets[1:, 3][valid] = 1 - np.clip(similarity, -1, 1)
    if normalization is not None:
        low, high = np.asarray(normalization['low']), np.asarray(normalization['high'])
        active = np.asarray(normalization['active'], dtype=bool)
        if low.shape != (4,) or high.shape != (4,) or active.shape != (4,):
            raise ValueError('four attribute normalization required')
        if not np.isfinite(low).all() or not np.isfinite(high).all() or (active & (high <= low)).any():
            raise ValueError('invalid fixed normalization')
        mask &= active[None, :]
        targets = np.clip((targets - low) / np.where(active, high - low, 1), 0, 1)
        targets[~mask] = 0
    return targets.astype(np.float32), mask


def fit_normalization(records):
    by_sensor = {}
    for record in records:
        if record['role'] != 'fit':
            raise ValueError('normalization accepts fit records only')
        targets, mask = build_evidence_targets(record['md'])
        bucket = by_sensor.setdefault(record['sensor_id'], [[] for _ in range(4)])
        for index in range(4):
            bucket[index].append(targets[mask[:, index], index])
    if not by_sensor:
        raise ValueError('fit normalization sources are empty')
    result = {}
    for sensor, columns in by_sensor.items():
        low, high, median, active = [], [], [], []
        for chunks in columns:
            values = np.concatenate(chunks)
            quantiles = np.quantile(values, (.05, .5, .95)) if len(values) else (0, 0, 0)
            lo, med, hi = map(float, quantiles)
            low.append(lo); high.append(hi); median.append(med)
            active.append(hi - lo > max(1e-8, abs(hi) * 1e-7))
        result[sensor] = dict(low=low, high=high, median=median, active=active,
                              source_role='fit', kind='MD-statistic-proxy')
    return result


def mean_evidence_target(targets, mask):
    targets, mask = np.asarray(targets), np.asarray(mask, dtype=bool)
    if targets.ndim != 2 or targets.shape[1] != 4 or targets.shape != mask.shape or not len(targets):
        raise ValueError('nonempty matching proxy targets/mask required')
    counts = mask.sum(axis=0)
    value = np.where(mask, targets, 0).sum(axis=0) / np.maximum(counts, 1)
    return value.astype(np.float32), counts > 0


def update_evidence_targets(targets, mask):
    length = len(targets)
    if length == 0 or length % 2:
        raise ValueError('two-frame updates require positive even frame count')
    rows = [mean_evidence_target(targets[i:i+2], mask[i:i+2]) for i in range(0, length, 2)]
    return np.stack([r[0] for r in rows]), np.stack([r[1] for r in rows])


def clean_runs(frame_labels):
    labels = np.asarray(frame_labels)
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError('two frame labels per update required')
    result, i = [], 0
    while i < len(labels):
        label = int(labels[i, 0])
        if label < 0 or not (labels[i] == label).all():
            i += 1
            continue
        end = i + 1
        while end < len(labels) and (labels[end] == label).all():
            end += 1
        result.append((i, end, label))
        i = end
    return result
