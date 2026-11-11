from __future__ import annotations

import importlib
import numpy as np
import pytest


def module():
    return importlib.import_module('continuous_mmwave.split_birnn.data')


def test_command_mapping_preserves_unknown_frames_and_rejects_unknown_class():
    m = module()
    commands = ['fall on floor of room', 'get in bed', 'get out bed', 'lie in bed', 'roll in bed',
                'sit down on bed', 'sit down on chair', 'sit in bed', 'stand up from bed',
                'stand up from chair', 'stand up from floor of room', 'walk to bed', 'walk to chair', 'walk to room']
    y = m.merge_labels(np.array([-100] + list(range(14))), dict(enumerate(commands)))
    assert y.tolist() == [-100, 1, 4, 8, 5, 6, 3, 3, 7, 2, 2, 2, 0, 0, 0]
    with pytest.raises(ValueError):
        m.merge_labels(np.array([0]), {0: 'unknown action'})
    with pytest.raises(ValueError):
        m.merge_labels(np.array([99]), dict(enumerate(commands)))


def test_normalization_is_per_frame_and_zero_variance_is_finite():
    m = module()
    x = np.stack([np.arange(128), 50 + 3 * np.arange(128), np.full(128, 4)])
    y = m.standardize_frames(x)
    np.testing.assert_allclose(y[0], y[1], atol=1e-6)
    np.testing.assert_allclose(y[:2].mean(1), 0, atol=1e-6)
    np.testing.assert_allclose(y[:2].std(1), 1, atol=1e-6)
    assert np.array_equal(y[2], np.zeros(128))
    with pytest.raises(ValueError):
        m.standardize_frames(np.full((1, 128), np.nan))


def test_nonzero_clock_origin_retains_real_unlabeled_prefix_and_boundaries():
    m = module()
    y, audit = m.align_commands(8, np.array([0.0, .18, .36]), np.array([0, 1, 2]),
                                frame_period_s=.09, frame_origin_s=-.18)
    assert y.tolist() == [-100, -100, 0, 0, 1, 1, 2, 2]
    assert audit['command_frame_indices'] == [2, 4, 6]
    assert audit['unlabeled_prefix_frames'] == 2
    assert audit['valid_frame_count'] == 6
    assert audit['clock_calibrated'] is False
    assert audit['physical_action_onset_available'] is False
    assert audit['csv_span_s'] == pytest.approx(.36)
    assert audit['last_command_to_end_s'] == pytest.approx(.18)


def test_coverage_flags_outside_commands_without_deleting_frames():
    y, audit = module().align_commands(4, np.array([0., .9]), np.array([0, 1]))
    assert y.tolist() == [0, 0, 0, 0]
    assert audit['commands_at_or_after_end'] == 1
    with pytest.raises(ValueError):
        module().align_commands(4, np.array([.2, .1]), np.array([0, 1]))


def test_runs_and_transitions_do_not_bridge_ignored_frames():
    a = module().audit_labels(np.array([-100, 0, 0, 1, -100, 1, 2, 2]))
    assert a['frame_count'] == 8
    assert a['valid_frame_count'] == 6
    assert a['transition_count'] == 2
    assert a['class_frame_counts'][:3] == [2, 2, 2]
    assert a['class_run_counts'][:3] == [1, 2, 1]


def test_subject_folds_keep_all_sources_together_and_cover_outer_test_once():
    rows = [{'subject_id': f's{i}', 'session_id': f'a{j}', 'sensor_id': str(k),
             'recording_id': f's{i}-a{j}-{k}'} for i in range(10) for j in range(2) for k in range(2)]
    folds = module().build_folds(rows, seed=42)
    outer = []
    for f in folds:
        roles = [set(f[k + '_subjects']) for k in ('train', 'validation', 'test')]
        assert all(not (roles[i] & roles[j]) for i in range(3) for j in range(i))
        assert set.union(*roles) == {f's{i}' for i in range(10)}
        outer.extend(f['test_subjects'])
    assert sorted(outer) == [f's{i}' for i in range(10)]


def test_duplicate_recording_identity_cannot_enter_folds_twice():
    rows = [{'subject_id': f's{i}', 'recording_id': f'r{i}'} for i in range(5)]
    with pytest.raises(ValueError):
        module().build_folds(rows + [rows[0]])
