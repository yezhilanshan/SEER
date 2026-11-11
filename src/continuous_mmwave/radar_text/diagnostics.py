"""Timestamp-aware diagnostics; no model fitting or ground-truth based postprocessing."""
from collections import Counter

import numpy as np


BED_CLASSES = (1, 2, 3, 4, 5, 7, 8)
FOCUS_CLASSES = (1, 2, 3, 8, 9, 10)


def classification_metrics(matrix):
    matrix = np.asarray(matrix, dtype=np.int64)
    support, predicted = matrix.sum(1), matrix.sum(0)
    tp = matrix.diagonal()
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(len(tp)), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(len(tp)), where=precision + recall > 0)
    return dict(frames=int(matrix.sum()), frame_accuracy=float(tp.sum() / max(1, matrix.sum())),
                macro_precision=float(precision.mean()), macro_recall=float(recall.mean()),
                macro_f1=float(f1.mean()), confusion_matrix=matrix.tolist(),
                per_class=[dict(class_id=i, support=int(support[i]), predicted=int(predicted[i]),
                                precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i]))
                           for i in range(len(tp))])


def confusion(labels, predicted, mask=None, classes=14):
    labels, predicted = np.asarray(labels), np.asarray(predicted)
    valid = (labels >= 0) & (labels < classes)
    if mask is not None:
        valid &= mask
    if np.any((predicted[valid] < 0) | (predicted[valid] >= classes)):
        raise ValueError('predictions outside class vocabulary')
    return np.bincount(labels[valid] * classes + predicted[valid], minlength=classes**2).reshape(classes, classes)


def period(times):
    times = np.asarray(times, dtype=np.float64)
    if len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('need at least two finite, increasing timestamps')
    dt = float(np.median(np.diff(times)))
    if not np.allclose(np.diff(times), dt, rtol=.05, atol=1e-6):
        raise ValueError('irregular timestamps require explicit repair')
    return dt


def steady_mask(labels, times, radius=.3):
    labels, times = np.asarray(labels), np.asarray(times)
    period(times)
    keep = labels >= 0
    for index in np.flatnonzero(labels[1:] != labels[:-1]) + 1:
        boundary = (times[index] + times[index - 1]) / 2
        keep &= np.abs(times - boundary) > radius + 1e-8
    return keep


def runs(labels):
    labels = np.asarray(labels)
    if not len(labels):
        return []
    edges = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(edges[:-1], edges[1:])]


def smooth_predictions(predicted, times, median_seconds=0., minimum_seconds=0.):
    predicted, times = np.asarray(predicted), np.asarray(times)
    dt = period(times)
    if median_seconds < 0 or minimum_seconds < 0:
        raise ValueError('negative smoothing threshold')
    filtered = predicted.copy()
    if median_seconds:
        for i, time in enumerate(times):
            lo = np.searchsorted(times, time - median_seconds / 2 - 1e-8)
            hi = np.searchsorted(times, time + median_seconds / 2 + 1e-8, side='right')
            # Integer class-ID median, including its arbitrary ordering, is explicit.
            filtered[i] = np.sort(predicted[lo:hi])[(hi - lo) // 2]
    actions = []
    for start, stop, action in runs(filtered):
        duration = times[stop - 1] - times[start] + dt
        if duration + 1e-8 >= minimum_seconds and (not actions or actions[-1] != action):
            actions.append(action)
    return actions


def action_instances(labels, times, start, stop):
    times = np.asarray(times)
    dt = period(times)
    length = times[stop - 1] - times[start] + dt
    rows = []
    for left, right, action in runs(labels):
        if action < 0 or right <= start or left >= stop:
            continue
        a, b = max(left, start), min(right, stop)
        cut_left, cut_right = left < start, right > stop
        rows.append(dict(class_id=action, start_frame=a, stop_frame=b,
                         duration=float(times[b - 1] - times[a] + dt),
                         full_duration=float(times[right - 1] - times[left] + dt),
                         relative_start=float((times[a] - times[start]) / length),
                         relative_end=float((times[b - 1] + dt - times[start]) / length),
                         at_window_edge=bool(a == start or b == stop),
                         truncation='both' if cut_left and cut_right else 'left' if cut_left else 'right' if cut_right else 'complete'))
    return rows


def attribute_instances(instances, predicted_actions):
    expected = Counter(row['class_id'] for row in instances)
    available = Counter(predicted_actions)
    rows = []
    for instance in instances:
        action = instance['class_id']
        covered = available[action] > 0
        available[action] -= int(covered)
        coverage = min(expected[action], predicted_actions.count(action))
        rows.append(dict(instance, missed=not covered,
                         missed_weight=(expected[action] - coverage) / expected[action],
                         identity_ambiguous=expected[action] > 1,
                         temporal_substitution='unresolved'))
    return rows


def match_controls(rows, index, tolerance):
    current = rows[index]
    for offset in range(1, len(rows)):
        candidate = (index + offset) % len(rows)
        other = rows[candidate]
        if (other['sensor'] == current['sensor']
                and abs(other['duration'] - current['duration']) <= tolerance + 1e-8
                and other['actions'] != current['actions']):
            if ('recording' in current and current['recording'] == other.get('recording')
                    and current['start'] < other['stop'] and other['start'] < current['stop']):
                continue
            return candidate
    return None
