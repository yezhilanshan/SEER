import numpy as np
import pytest


def example(role='train', subject='subject1'):
    from continuous_mmwave.harrad.joint_data import make_record
    frames = 81
    meta = dict(file=subject+'_01.hdf5', role=role, subject=subject, frames=frames,
                instances=[dict(start=3, stop=19, label=2, instance_id=0),
                           dict(start=19, stop=39, label=2, instance_id=1),
                           dict(start=42, stop=75, label=4, instance_id=2)])
    rng = np.random.default_rng(8)
    return make_record(meta, rng.normal(size=(frames,4,64)).astype('float32'),
                       rng.normal(size=(frames,64)).astype('float32'))


def test_common_targets_keep_crossings_and_same_class_mixed_is_not_semantic():
    r = example()
    assert len(r['legal']) == 35 and r['dropped_tail_frames'] == 1
    assert r['semantic'][9] == -100  # frames18/19 cross two same-class instances
    assert r['semantic'][8] == 2 and r['latest_start'][9] == 9
    assert 8 in r['legal'] and r['crossing'][8,0]
    assert not r['crossing'][5,0]
    with pytest.raises(ValueError, match='role'):
        example('test')


def test_fit_only_stats_and_fixed_shared_schedule_and_panels():
    from continuous_mmwave.harrad.joint_data import fit_stats, fixed_design
    records = [example(), example('validation','subject7')]
    stats = fit_stats(records)
    records[1]['motion'] *= 100
    again = fit_stats(records)
    np.testing.assert_array_equal(stats['mean'], again['mean'])
    assert stats['persistence_mse'].shape == (3,) and np.all(stats['persistence_mse'] > 0)
    a = fixed_design(records, [42], 7, 8)
    b = fixed_design(records, [42], 7, 8)
    np.testing.assert_array_equal(a['schedule_42'], b['schedule_42'])
    assert a['schedule_42'].shape == (7,2,2)
    assert a['validation'].shape[1] == 4
    assert a['validation'][:,2].sum() == a['validation'][:,3].sum() == 8


def test_batch_only_reads_causal_oracle_prefix_and_target_is_independent():
    from continuous_mmwave.harrad.joint_data import fit_stats, collate_origins
    r = example(); stats = fit_stats([r])
    batch = collate_origins([r], [(0,14)], stats)
    # Most recent official instance begins in Update9, so context is9..14.
    assert batch['mask'].sum().item() == 6
    original = batch['cells'].clone()
    r['cells'][15:] *= 100
    newer = collate_origins([r], [(0,14)], stats)
    np.testing.assert_array_equal(original.numpy(), newer['cells'].numpy())
    assert batch['future'].shape == (1,3,64) and not batch['future'].requires_grad


def test_mixed_future_is_not_misreported_as_unlabelled_future():
    from continuous_mmwave.harrad.joint_data import fit_stats, collate_origins
    r=example(); stats=fit_stats([r])
    batch=collate_origins([r],[(0,8)],stats)
    assert batch['crossing'][0,0]
    assert not batch['future_unknown'][0,0]  # frame18/19 both labelled, IDs differ
    assert batch['future_mixed'][0,0]
