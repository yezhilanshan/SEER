import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec=importlib.util.spec_from_file_location('diagnostic',Path(__file__).parents[1]/'scripts/diagnose_harrad_stage1.py')
diagnostic=importlib.util.module_from_spec(spec);spec.loader.exec_module(diagnostic)
spec2=importlib.util.spec_from_file_location('verifier',Path(__file__).parents[1]/'scripts/verify_harrad_stage1_diagnostic.py')
verifier=importlib.util.module_from_spec(spec2);spec2.loader.exec_module(verifier)


def test_bootstrap_resamples_recordings_and_preserves_seed_pairing():
    result=diagnostic.bootstrap_records([[1,2,3],[-3,-2,-1]])
    assert result['mean']==0
    np.testing.assert_allclose(result['ci95'],[-2,2])
    np.testing.assert_allclose(result['leave_one_out'],[-2,2])
    assert diagnostic.bootstrap_records([[1,2,3]])['ci95'] is None


def test_randomization_keeps_reset_count_and_event_length_multiset():
    beginnings=np.array([0,0,2,2,2,5,5,5,5])
    starts,old,new=diagnostic.shuffled_starts(beginnings,20261105)
    assert len(starts)==3 and starts[0]==0
    assert sorted(old)==sorted(new)
    assert new.sum()==len(beginnings)


def test_scene_definitions_return_empty_candidate_groups_without_inventing_labels():
    x=np.zeros((100,64))
    instances=[dict(start=10,stop=90,label=0)]
    thresholds=dict(jump=1,return_=0,drift=1,contrast=1)
    thresholds['return']=thresholds.pop('return_')
    masks,meta=diagnostic.scene_masks(x,instances,np.array([30,40,50]),thresholds)
    assert masks.shape==(3,3,7)
    assert masks[:,:,0].all()
    assert not masks[:,:,(2,3,4,6)].any()
    assert meta['slow']==[] and meta['transients']==[]


def test_independent_readback_rejects_corrupt_bootstrap_or_leave_out():
    values=np.array([[1,2,3],[-3,-2,-1]])
    expected=diagnostic.bootstrap_records(values)
    verifier.check_pairing(expected,values)
    for key in ('ci95','leave_one_out'):
        wrong=dict(expected);wrong[key]=[0,0]
        with pytest.raises(AssertionError):verifier.check_pairing(wrong,values)


def test_control_cohort_requires_all_unique_seed_arms_and_all_files():
    expected={'subject7_01.hdf5','subject7_02.hdf5','subject7_03.hdf5','subject7_04.hdf5'}
    jobs=[dict(condition=c,seed=s,records=[dict(file=f) for f in expected])
          for c in ('swap','retrained') for s in (42,43,44)]
    assert verifier.check_cohort(jobs,expected)==4
    with pytest.raises(AssertionError):verifier.check_cohort(jobs[:-1]+[jobs[0]],expected)
    jobs[0]['records']=jobs[0]['records'][:-1]
    with pytest.raises(AssertionError):verifier.check_cohort(jobs,expected)
