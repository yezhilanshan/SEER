"""Data leakage and causal attribute contracts for the approved dataset v2."""
from pathlib import Path
import importlib

import numpy as np
import pytest


def module():
    return importlib.import_module('continuous_mmwave.v25.data')


def row(tmp_path, subject, session='set000', sensor='TI_xWR14xx', **extra):
    path = tmp_path / f'{subject}-{session}-{sensor}.npz'
    path.write_bytes(b'source identity only; never load arrays in manifest builder')
    return dict(path=str(path), subject_id=subject, session_id=session,
                sensor_id=sensor, environment_id='source-unknown', **extra)


def test_fixed_roles_keep_new_subject_whole_and_both_sensors_bound(tmp_path):
    data = module()
    rows = [row(tmp_path, s, t, sensor) for s,t in [('subject_08','set000'),
            ('subject_08','set002'),('subject_08','set003'),('subject_12','set000'),
            ('subject_13','set000')] for sensor in ['TI_xWR14xx','TI_xWR68xx']]
    result = data.build_training_manifest(rows, data.DATASET_V2)
    assert {r['subject_id'] for r in result['splits']['fit']} == {'subject_08','subject_13'}
    assert len(result['splits']['dev']) == 2
    assert len(result['splits']['subject_holdout']) == 2
    for values in result['splits'].values():
        pairs = {(r['subject_id'],r['session_id']) for r in values}
        for subject,session in pairs:
            assert len([r for r in values if (r['subject_id'],r['session_id'])==(subject,session)]) == 2


def test_manifest_rejects_claimed_fit_for_holdout_and_official_subject(tmp_path):
    data=module()
    with pytest.raises(ValueError,match='role'):
        data.build_training_manifest([row(tmp_path,'subject_12',role='fit')],data.DATASET_V2)
    with pytest.raises(ValueError,match='protected'):
        data.build_training_manifest([row(tmp_path,'subject_06')],data.DATASET_V2)


def test_manifest_rejects_part_and_cross_metadata_alias(tmp_path):
    data=module()
    item=row(tmp_path,'subject_08'); item['path'] += '.part'
    with pytest.raises(ValueError,match='part|missing'):
        data.build_training_manifest([item],data.DATASET_V2)
    original=row(tmp_path,'subject_08')
    changed={**original,'subject_id':'subject_13'}
    with pytest.raises(ValueError,match='identity'):
        data.build_training_manifest([original,changed],data.DATASET_V2)


def test_exact_duplicate_is_not_an_extra_training_sample(tmp_path):
    data=module(); item=row(tmp_path,'subject_08')
    result=data.build_training_manifest([item,item],data.DATASET_V2)
    assert len(result['splits']['fit']) == 1
    assert len(result['duplicates']) == 1


def test_physical_targets_are_causal_and_missing_previous_not_zero_truth():
    data=module(); md=np.arange(6*128,dtype=np.float32).reshape(6,128)+1
    targets,mask=data.build_evidence_targets(md)
    changed=md.copy(); changed[-1]+=500
    later,_=data.build_evidence_targets(changed)
    np.testing.assert_allclose(targets[:-1],later[:-1])
    assert mask[0].tolist() == [True,True,False,False]
    assert mask[1:].all()
    assert targets.shape == (6,4)


def test_normalization_uses_fit_only_and_does_not_train_constant_attributes():
    data=module()
    md=np.arange(12*128,dtype=np.float32).reshape(12,128)+1
    records=[dict(role='fit',sensor_id='TI_xWR14xx',md=md)]
    norm=data.fit_normalization(records)
    target,mask=data.build_evidence_targets(md,normalization=norm['TI_xWR14xx'])
    assert np.isfinite(target[mask]).all()
    assert ((target[mask]>=0)&(target[mask]<=1)).all()
    assert not mask[:,1].any()  # per-frame spectral standard deviation is constant
    with pytest.raises(ValueError,match='fit'):
        data.fit_normalization([{**records[0],'role':'dev'}])


def test_holdout_cannot_enter_array_reader_before_lock(tmp_path):
    data=module(); item=row(tmp_path,'subject_12')
    item['role']='subject_holdout'
    with pytest.raises(ValueError,match='locked|holdout'):
        data.require_array_access(item,allowed_roles=('fit','dev'))
    data.require_array_access(item,allowed_roles=('subject_holdout',),terminal_locked=True)
