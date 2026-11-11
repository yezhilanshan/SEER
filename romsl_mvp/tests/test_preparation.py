import json

import numpy as np

from romsl_mvp.romsl.pipeline import Windows, prepare


def test_preparation_uses_train_stats_and_defers_test(tmp_path):
    root, job = tmp_path / 'data', tmp_path / 'job'
    audit = job / 'reports/audit'
    audit.mkdir(parents=True)
    (job / 'artifacts').mkdir()
    (audit / 'data_audit.json').write_text(json.dumps({'gates': {'A': 'READY', 'B': 'READY'}}))
    config = {'data': {'scene': 'E01', 'fps': 10,
                      'splits': {'train': ['S01'], 'val': ['S02'], 'test': ['S03']}}}
    joints = {'left_hip': 4, 'right_hip': 1, 'groups': {
        'left_arm': [11,12,13], 'right_arm': [14,15,16],
        'left_leg': [4,5,6], 'right_leg': [1,2,3]}}
    # No S03 files exist. Preparation must complete without touching the held-out subject.
    for subject, point_value in [('S01', 2), ('S02', 900)]:
        for action in range(1,28):
            path = root / 'E01' / subject / f'A{action:02}'
            (path / 'mmwave').mkdir(parents=True)
            pose = np.ones((20,17,3), dtype=np.float32)
            pose[...,0] += np.arange(20)[:,None] * .01
            np.save(path / 'ground_truth.npy', pose)
            for frame in range(1,21):
                np.full((2,5), point_value, dtype='<f8').tofile(path / 'mmwave' / f'frame{frame:03}.bin')
    counts = prepare(root, job, config, joints)
    assert counts == {'train': 27, 'val': 27}
    with np.load(job / 'artifacts/input_stats.npz') as stats:
        np.testing.assert_allclose(stats['mean'], 2)
    for exp in ('A','B'):
        data = Windows(job, 'train', exp)
        points, mask, target, target_mask = data[0]
        assert np.isfinite(target.numpy()).all()
        assert mask.sum() == 32 and target.shape[-1] == (51 if exp == 'A' else 5)
    assert not (job / 'artifacts/manifest_test.jsonl').exists()
