import numpy as np
import pytest

from romsl_mvp.romsl.experiment import classification_metrics, clip_classification, recording_mae, reserve_test


def test_full_experiment_on_synthetic_recordings(tmp_path):
    import json
    from romsl_mvp.romsl.experiment import run_all
    from romsl_mvp.romsl.pipeline import digest, prepare
    import torch

    root, job = tmp_path / 'data', tmp_path / 'job'
    (job / 'reports/audit').mkdir(parents=True)
    (job / 'artifacts').mkdir()
    (job / 'source/configs').mkdir(parents=True)
    config = {'data': {'scene': 'E01', 'fps': 10,
                      'splits': {'train': ['S01'], 'val': ['S02'], 'test': ['S03']},
                      'probe_actions': ['A03','A06','A12','A13','A14','A17','A18','A26']}}
    joints = {'left_hip': 4, 'right_hip': 1, 'groups': {
        'left_arm': [11,12,13], 'right_arm': [14,15,16],
        'left_leg': [4,5,6], 'right_leg': [1,2,3]}}
    (job / 'source/configs/mvp.yaml').write_text(json.dumps(config))
    (job / 'source/configs/joints.yaml').write_text(json.dumps(joints))
    (job / 'reports/audit/data_audit.json').write_text(json.dumps({'gates': {'A': 'READY', 'B': 'READY'}}))
    for subject in ('S01','S02','S03'):
        for action in range(1,28):
            path = root / 'E01' / subject / f'A{action:02}'
            (path / 'mmwave').mkdir(parents=True)
            pose = np.ones((20,17,3),dtype=np.float32)
            pose[...,0] += np.arange(20)[:,None] * .01
            np.save(path / 'ground_truth.npy',pose)
            for frame in range(1,21):
                np.full((2,5),action + frame * .01,dtype='<f8').tofile(path / 'mmwave' / f'frame{frame:03}.bin')
    prepare(root,job,config,joints)
    files = {str(p.relative_to(job)).replace('\\','/'):digest(p) for p in job.rglob('*') if p.is_file()}
    (job / 'experiment_lock.json').write_text(json.dumps({'files':files}))
    run_all(job,root,torch.device('cpu'),{'pretrain_epochs':1,'probe_epochs':1})
    results = json.loads((job / 'test_results.json').read_text())
    assert (job / 'completed.json').exists()
    assert results['A']['classification']['recordings'] == 8
    assert results['B']['classification']['recordings'] == 8
    assert len(results['A']['motion']['per_recording']) == 27
    assert results['C']['status'] == 'BLOCKED_GEOMETRY'


def test_clip_metrics_give_each_recording_one_vote():
    logits = np.array([[10, 0], [10, 0], [0, 30], [0, 10]])
    metrics = clip_classification(logits, ['S01/A', 'S01/A', 'S01/A', 'S02/B'], [0, 0, 0, 1], 2)
    assert metrics['accuracy'] == .5
    assert metrics['recordings'] == 2
    assert metrics['confusion_matrix'] == [[0, 1], [0, 1]]
    assert metrics['per_subject']['S01']['accuracy'] == 0
    assert metrics['per_subject']['S02']['accuracy'] == 1
    assert classification_metrics([0,1], [0,1], 3)['macro_f1'] == pytest.approx(2/3)


def test_mae_balances_recordings_before_subjects():
    errors = np.array([[2., 4.], [2., 4.], [8., 6.]])
    valid = np.array([[1,1],[1,1],[1,1]], dtype=bool)
    result = recording_mae(errors, valid, ['S01/A','S01/A','S02/B'])
    np.testing.assert_allclose(result['per_component'], [5,5])
    assert result['mean_mae'] == 5


def test_locked_test_refuses_second_access(tmp_path):
    reserve_test(tmp_path, {'A': 'locked'})
    with pytest.raises(FileExistsError):
        reserve_test(tmp_path, {'A': 'changed'})


def test_seed_controls_initialization():
    import torch
    from romsl_mvp.romsl.experiment import seed_all
    from romsl_mvp.romsl.model import RadarEncoder
    seed_all(43)
    first = RadarEncoder().point_mlp[0].weight.detach().clone()
    seed_all(43)
    torch.testing.assert_close(first, RadarEncoder().point_mlp[0].weight, atol=0, rtol=0)
    seed_all(44)
    assert not torch.equal(first, RadarEncoder().point_mlp[0].weight)
