import numpy as np


def test_complete_record_training_replays_and_scores_online_predictions(tmp_path):
    import torch
    from continuous_mmwave.harrad.boundary import boundary_targets
    from continuous_mmwave.harrad.boundary_training import fit_boundary
    torch.set_num_threads(1)
    records = []
    for index, role in enumerate(('train', 'train', 'validation')):
        labels = boundary_targets(81, [dict(start=9, stop=30), dict(start=40, stop=69)])
        x = np.random.default_rng(index).normal(size=(41, 256)).astype('float32')
        records.append({**labels, 'x': x, 'file': f'subject{index+1}_01.hdf5',
                        'subject': f'subject{index+1}', 'role': role, 'frames': 81})
    config = {'epochs': 2, 'seed': 42, 'learning_rate': .001, 'weight_decay': .01,
              'thresholds': [.3, .5, .7], 'main_tolerance_frames': 4, 'tolerances_frames': [2, 4, 8]}
    report = fit_boundary(records, config, tmp_path / 'run', 'cpu')
    assert report['epochs_completed'] == 2 and report['successful_steps'] == 4
    assert report['replay_max_abs'] == 0 and report['independent_verified']
    assert report['training_recordings_per_epoch'] == 2
    assert report['train_frame_exposures'] == 324
    assert report['trainable_parameters'] == 33154
    assert (tmp_path / 'run/validation-best/subject3_01.npz').exists()
