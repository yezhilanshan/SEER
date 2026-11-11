import numpy as np


def sample_points(points, mean, std, *, seed, max_points=128):
    points = np.asarray(points)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) > max_points:
        points = points[np.random.default_rng(seed).choice(len(points), max_points, replace=False)]
    output = np.zeros((max_points, 5), dtype=np.float32)
    mask = np.zeros(max_points, dtype=bool)
    output[:len(points)] = (points - mean) / np.maximum(std, 1e-6)
    mask[:len(points)] = True
    return output, mask


def make_windows(radar_valid, targets_valid, *, size=16, stride=8, min_valid=12):
    if len(radar_valid) != len(targets_valid) or size <= 0 or stride <= 0:
        raise ValueError('Invalid frame counts or window configuration')
    if not 0 < min_valid <= size:
        raise ValueError('min_valid must be in [1,size]')
    windows = []
    for start in range(0, len(radar_valid) - size + 1, stride):
        end = start + size
        if np.count_nonzero(radar_valid[start:end]) >= min_valid and np.count_nonzero(targets_valid[start:end]) >= min_valid:
            windows.append((start, end))
    return windows
