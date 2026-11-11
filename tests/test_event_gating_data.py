import json
from pathlib import Path
import numpy as np
import pytest
import torch


def test_constructed_sequence_replays_gru_and_keeps_same_label_instances():
    from continuous_mmwave.event_gating.data import construct_sequence
    from continuous_mmwave.models.md_gru_encoder import CausalMDGRUEncoder
    torch.manual_seed(9)
    temporal = CausalMDGRUEncoder(64, 128, 0).eval()
    frames = np.random.default_rng(4).normal(size=(15, 64)).astype(np.float32)
    instances = [dict(start=1, stop=4, label=2, instance_id=5),
                 dict(start=10, stop=14, label=2, instance_id=6)]
    result = construct_sequence(frames, instances, temporal)
    assert result['targets'] == [2, 2]
    assert result['source_frames'].tolist() == [1, 2, 3, 10, 11, 12, 13]
    joined = torch.from_numpy(np.concatenate((frames[1:4], frames[10:14])))[None]
    with torch.no_grad():
        expected = temporal(joined).frame_hidden[0]
        original = temporal(torch.from_numpy(frames)[None]).frame_hidden[0]
    torch.testing.assert_close(result['features'], expected)
    assert not torch.allclose(result['features'][3:], original[10:14])


def test_manifest_rejects_subject_leakage_before_opening_signals():
    from continuous_mmwave.event_gating.data import validate_records
    with pytest.raises(ValueError, match='split'):
        validate_records([dict(subject='subject8', role='train', file='x', frames=3,
                               continuous_path='do-not-open', instances=[])])
    with pytest.raises(ValueError, match='test'):
        validate_records([dict(subject='subject8', role='test', file='x', frames=3,
                               continuous_path='do-not-open', instances=[])])


def test_annotation_validation_rejects_overlap_and_bad_labels():
    from continuous_mmwave.event_gating.data import validate_records
    record = dict(subject='subject1', role='train', file='r', frames=10,
                  instances=[dict(start=1, stop=5, label=1, instance_id=0),
                             dict(start=4, stop=8, label=2, instance_id=1)])
    with pytest.raises(ValueError, match='interval'):
        validate_records([record])
    record['instances'][1].update(start=6, label=99)
    with pytest.raises(ValueError):
        validate_records([record])


def test_asset_lock_fails_on_changed_bytes_and_missing_path(tmp_path):
    from continuous_mmwave.event_gating.data import locked_path, sha256
    p = tmp_path/'x'
    p.write_bytes(b'old')
    lock = dict(path=str(p), sha256=sha256(p))
    assert locked_path(lock) == p
    p.write_bytes(b'new')
    with pytest.raises(ValueError, match='SHA'):
        locked_path(lock)
    with pytest.raises(FileNotFoundError):
        locked_path(dict(path=str(tmp_path/'missing'), sha256='0'*64))


def test_sampling_is_reproducible_and_never_uses_test():
    from continuous_mmwave.event_gating.data import sample_sequences
    inst = [dict(start=i*5, stop=i*5+3, label=0, instance_id=i) for i in range(7)]
    r = dict(file='a', subject='subject1', role='train', frames=40, instances=inst)
    a = sample_sequences([r], seed=12)
    assert a == sample_sequences([r], seed=12)
    assert any(len(row['instances'])>1 for row in a)
    assert len({row['sample_id'] for row in a}) == len(a)
    assert all(len({item['label'] for item in row['instances']}) == 1 for row in a)
