import importlib
import numpy as np
import pytest
import torch


def api():
    return importlib.import_module('continuous_mmwave.v261.contract')


def fixture(role='train', n=45):
    torch.manual_seed(18)
    radar = torch.randn(n, 4, 256)
    cache = dict(uid='s01-x', role=role, metadata={'subject_id':'s01'}, radar=radar,
                 motion=radar.mean(1), intervals=torch.tensor(np.stack((np.arange(n)*.18, (np.arange(n)+1)*.18), 1)),
                 source_sha256='a'*64, command_sha256='b'*64)
    physical = dict(observation_valid=np.ones(n, bool), common_start=np.arange(1,n-5), mixed_update=np.zeros(n,bool))
    labels = torch.arange(n).remainder(14).repeat_interleave(2)
    return cache, labels, physical


def test_semantic_mask_does_not_delete_physical_prediction_or_reset_history():
    cache, labels, physical = fixture()
    original = api().build_joint_record(cache,labels,physical,'train')
    labels[16:18] = torch.tensor([2,3])
    labels[20:22] = -1
    physical['mixed_update'][12] = True
    record = api().build_joint_record(cache,labels,physical,'train')
    assert record.semantic_target[8] == -100 and record.semantic_target[10] == -100
    assert record.semantic_target[12] == -100
    assert torch.equal(original.common_start,record.common_start)
    assert api().prefix_bounds(record,35) == (4,36)
    physical['observation_valid'][20] = False
    physical['common_start'] = np.array([t for t in physical['common_start'] if not t <=20<=t+5])
    record = api().build_joint_record(cache,labels,physical,'train')
    assert api().prefix_bounds(record,35) == (21,36)


def test_short_prefix_bad_role_nonfinite_or_illegal_future_rejected():
    cache, labels, physical = fixture()
    record = api().build_joint_record(cache,labels,physical,'train')
    assert api().prefix_bounds(record,2) == (0,3)
    with pytest.raises(ValueError,match='role'):
        api().build_joint_record(cache,labels,physical,'validation')
    physical['common_start'] = np.array([44])
    with pytest.raises(ValueError,match='future|common'):
        api().build_joint_record(cache,labels,physical,'train')
    physical['common_start'] = np.array([4])
    cache['radar'][0,0,0] = float('nan')
    with pytest.raises(ValueError,match='finite'):
        api().build_joint_record(cache,labels,physical,'train')


def test_schedule_semantic_first_and_fit_statistics_never_use_validation():
    cache, labels, physical = fixture()
    labels[8:10] = -1
    record = api().build_joint_record(cache,labels,physical,'train')
    a = api().joint_schedule([record],42,steps=9)
    assert a == api().joint_schedule([record],42,steps=9)
    assert len(a)==9 and all(len(row)==2 for row in a)
    for row in a:
        assert record.semantic_target[row[0][1]] >= 0
        assert all(t in record.common_start for _,t in row)
    norm = dict(mean=torch.zeros(256),std=torch.ones(256))
    stats = api().fit_motion_contract([record],norm)
    assert stats['R'].shape==(3,) and (stats['R']>0).all()
    expected = torch.stack([((record.motion_ref[record.common_start+h]-record.motion_ref[record.common_start])**2).mean() for h in (1,3,5)])
    torch.testing.assert_close(stats['R'],expected)
    cache['role']='validation'
    val=api().build_joint_record(cache,labels,physical,'validation')
    with pytest.raises(ValueError,match='train|fit'):
        api().fit_motion_contract([val],norm)
    with pytest.raises(ValueError,match='train|fit'):
        api().joint_schedule([val],42,steps=9)


def test_class_mapping_and_locked_sources_reject_tampering(tmp_path):
    source=tmp_path/'file';source.write_bytes(b'original')
    with pytest.raises(ValueError,match='SHA'):
        api().verify_source_lock({str(source):'0'*64})
    old={'classes':[{'id':i,'command':str(i)} for i in range(14)]}
    changed={'classes':[{'id':i,'command':str((i+1)%14)} for i in range(14)]}
    with pytest.raises(ValueError,match='mapping'):
        api().validate_class_maps(old,changed)


def test_common_population_matches_p1c_previous_observation_requirement():
    cache,labels,physical=fixture(n=45)
    physical['common_start']=np.arange(0,40)
    aligned=api().with_previous_observation(physical,cache['intervals'])
    assert np.array_equal(aligned['common_start'],np.arange(1,40))
    assert np.array_equal(physical['common_start'],np.arange(0,40))
    assert aligned['observation_valid'].all()
