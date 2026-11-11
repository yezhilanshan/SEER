import numpy as np
import pytest

from romsl_mvp.romsl.mmfi_io import load_recording, read_frame
from romsl_mvp.romsl.audit import gate_status, subject_splits


def test_bin_dtype(tmp_path):
    path = tmp_path / 'frame001.bin'
    expected = np.arange(10, dtype='<f8').reshape(2, 5)
    expected.tofile(path)
    np.testing.assert_array_equal(read_frame(path), expected)
    path.write_bytes(b'')
    assert read_frame(path).shape == (0, 5)
    path.write_bytes(b'\0' * 20)
    with pytest.raises(ValueError, match='40-byte'):
        read_frame(path)


def make_recording(tmp_path, ids):
    directory = tmp_path / 'E01' / 'S01' / 'A03'
    (directory / 'mmwave').mkdir(parents=True)
    np.save(directory / 'ground_truth.npy', np.ones((len(ids), 17, 3), dtype=np.float32))
    for frame_id in ids:
        np.full((1, 5), int(frame_id), dtype='<f8').tofile(directory / 'mmwave' / f'frame{frame_id}.bin')
    return directory


def test_file_order(tmp_path):
    directory = make_recording(tmp_path, ['10', '2', '1', '3', '4', '5', '6', '7', '8', '9'])
    recording = load_recording(directory)
    assert recording.frame_ids == tuple(range(1, 11))
    assert recording.recording_id == 'E01/S01/A03'
    assert [points[0, 0] for points in recording.points_per_frame] == list(range(1, 11))


@pytest.mark.parametrize('ids', [('1', '3'), ('1', '001'), ('0', '1')])
def test_reject_frame_misalignment(tmp_path, ids):
    with pytest.raises(ValueError, match='frame IDs'):
        load_recording(make_recording(tmp_path, ids))


def test_split_isolation():
    with pytest.raises(ValueError, match='S02'):
        subject_splits({'train': ['S01', 'S02'], 'val': ['S03'], 'test': ['S02']})


def test_geometry_and_doppler_gates():
    assert gate_status(True, False, False) == {'A': 'READY', 'B': 'READY', 'C': 'BLOCKED_GEOMETRY', 'D': 'BLOCKED_GEOMETRY'}
    assert gate_status(True, True, False)['D'] == 'BLOCKED_DOPPLER'
    assert set(gate_status(False, True, True).values()) == {'BLOCKED_DATA'}
