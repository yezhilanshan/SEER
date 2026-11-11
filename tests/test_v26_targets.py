import importlib
import importlib.util
import numpy as np
import pytest
import torch


def api():
    assert importlib.util.find_spec('continuous_mmwave.v26.targets') is not None, 'future target interface missing'
    return importlib.import_module('continuous_mmwave.v26.targets')


def test_future_indices_respect_gaps_and_odd_tail():
    module = api()
    intervals = np.stack((np.arange(6) * .18, (np.arange(6) + 1) * .18), 1)
    assert module.future_indices(intervals, np.ones(6, bool), (1,))[1].tolist() == [1, 2, 3, 4, 5, -1]
    intervals[3:] += .09
    assert module.future_indices(intervals, np.ones(6, bool), (1,))[1][2] == -1
    with pytest.raises(ValueError, match='recording'):
        module.future_indices(np.concatenate([intervals, intervals]), np.ones(12, bool), (1,))
    grid = module.update_grid(np.arange(7) * .09, np.array([0., 10.]))
    assert len(grid['intervals']) == 3 and grid['trailing_frames'] == 1


def test_frozen_masked_target_and_fit_only_normalization():
    module = api()
    tokens = torch.tensor([[[1., 2.], [3., 4.], [99., 99.]]], requires_grad=True)
    target = module.fixed_motion_target(tokens, torch.tensor([[True, True, False]]))
    assert torch.equal(target, torch.tensor([[2., 3.]])) and not target.requires_grad
    stats = module.fit_normalization([{'role': 'train', 'motion': target, 'valid': torch.tensor([True])}])
    assert stats['mean'].tolist() == [2., 3.]
    with pytest.raises(ValueError, match='fit'):
        module.fit_normalization([{'role': 'validation', 'motion': target, 'valid': torch.tensor([True])}])


def test_command_rows_remain_distinct_and_mixed_updates_are_masked(tmp_path):
    module = api()
    csv = tmp_path / 'commands.csv'
    csv.write_text('timestamp,command\n2026-01-01T00:00:00,walk\n2026-01-01T00:00:00.270,walk\n')
    episodes = module.build_command_episodes({'source_commands': str(csv), 'recording_start_rule': 'first_command_as_legacy_converter'}, np.arange(10)*.09)
    assert len(episodes) == 2 and episodes[0]['episode_id'] != episodes[1]['episode_id']
    grid = module.update_grid(np.arange(10)*.09, np.array([e['start'] for e in episodes]))
    assert grid['valid'].tolist() == [True, False, True, True, True]
