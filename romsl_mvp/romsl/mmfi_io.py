from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


@dataclass
class Recording:
    recording_id: str
    action_id: str
    frame_ids: tuple[int, ...]
    points_per_frame: list[np.ndarray]
    pose: np.ndarray


def read_frame(path: Path) -> np.ndarray:
    size = path.stat().st_size
    if size % 40:
        raise ValueError(f'{path}: expected complete 40-byte point records, got {size} bytes')
    return np.fromfile(path, dtype='<f8').reshape(-1, 5)


def load_recording(path: Path) -> Recording:
    pose = np.load(path / 'ground_truth.npy', allow_pickle=False)
    if pose.ndim != 3 or pose.shape[1:] != (17, 3) or not len(pose) or pose.dtype.kind != 'f':
        raise ValueError(f'{path}: invalid skeleton array {pose.shape}, {pose.dtype}')
    frames = []
    for file in (path / 'mmwave').glob('*.bin'):
        match = re.fullmatch(r'frame(\d+)\.bin', file.name)
        if match is None:
            raise ValueError(f'{path}: invalid radar filename {file.name}')
        frames.append((int(match[1]), file))
    frames.sort(key=lambda pair: pair[0])
    ids = tuple(number for number, _ in frames)
    if ids != tuple(range(1, len(pose) + 1)):
        raise ValueError(f'{path}: frame IDs must be contiguous, unique and match the skeleton length')
    return Recording('/'.join(path.parts[-3:]), path.name, ids, [read_frame(file) for _, file in frames], pose)
