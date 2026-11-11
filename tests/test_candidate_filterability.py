"""Catch future leakage, reference drift, split leakage and duplicate-count errors."""
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pytest

def core():
    p=Path(__file__).parents[1]/'.audit-staging/candidate_filterability_core.py'
    assert p.exists(), 'filterability core absent'
    sys.path.insert(0,str(p.parent))
    s=importlib.util.spec_from_file_location('filtercore',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def test_reference_frozen_and_future_cannot_change_prefix_features():
    c=core();h=np.array([[1.,0.],[0.,1.],[1.,1.],[1.,0.],[9.,9.]])
    key=np.array([[0.,1.],[1.,0.],[1.,1.]])
    d,s=c.evidence(h[1:4],h[0],key,np.array([1.,0.]))
    assert d.shape==s.shape==(2,4)
    assert d[0,0]==pytest.approx(1-1/np.sqrt(2))
    assert d[1,0]==pytest.approx(0.)
    assert d[1,2]==pytest.approx(1-1/np.sqrt(2))
    dd,ss=c.evidence(h[1:3],h[0],key[:2],np.array([1.,0.]))
    np.testing.assert_array_equal(d[:1],dd);np.testing.assert_array_equal(s[:1],ss)

def test_paired_recordings_split_without_dev_or_sensor_leak():
    c=core()
    rows=[dict(uid=f'subject_02-{s}-{sensor}',role=r) for s,r in [('set000','fit'),('set001','fit'),('set002','dev')] for sensor in ('TI_xWR14xx','TI_xWR68xx')]
    assert c.partitions(rows)==['train','train','calibration','calibration','dev','dev']
    with pytest.raises(ValueError):c.partitions([dict(uid='subject_02-set003-TI_xWR14xx',role='fit')])

def test_retained_duplicate_is_rematched_and_decision_latency_is_separate():
    c=core();t=np.arange(0,3,.18);v=np.ones(len(t),bool);b=np.array([1.])
    events=[dict(time_s=.9),dict(time_s=1.08)]
    before=c.filtered_metrics(t,v,b,events,[True,True],1.08)
    after=c.filtered_metrics(t,v,b,events,[False,True],1.08)
    assert before['matched']==after['matched']==1
    assert after['false_proposals']==0
    assert after['latency_median_s']==pytest.approx(.08)
    assert after['decision_latency_median_s']==pytest.approx(1.16)

def test_missing_observation_is_not_padded_or_removed_from_gt_denominator():
    c=core();t=np.arange(10)*.18;v=np.ones(10,bool);v[5]=False
    assert c.prefix_available(t,v,3,1)
    assert not c.prefix_available(t,v,3,2)
    assert not c.prefix_available(t,v,8,2)
    assert not c.prefix_available(t,v,0,1)
    m=c.filtered_metrics(t,v,np.array([1.62]),[dict(time_s=1.62)],[False],1.08)
    assert m['eligible_boundaries']==1 and m['recall']==0

def test_normalization_is_fit_train_only_and_threshold_uses_rematched_counts():
    c=core();mean,scale=c.fit_normalizer(np.array([[1.,2.],[3.,2.]]))
    np.testing.assert_array_equal(mean,[2.,2.]);np.testing.assert_array_equal(scale,[1.,1.])
    baseline=dict(recall=.9,false_per_minute=30.)
    grid=[dict(threshold=0.,recall=.9,false_per_minute=30.),dict(threshold=.5,recall=.86,false_per_minute=12.),dict(threshold=.8,recall=.8,false_per_minute=4.)]
    lock=c.choose_locks(grid,baseline)
    assert lock['retain_recall']['threshold']==.5
    assert lock['halve_fp']['threshold']==.5
    assert lock['fp8']['threshold']==.8

def test_fixed_budget_probe_updates_only_new_parameters_and_reloads_predictions(tmp_path):
    import torch
    c=core();x=np.zeros((4,8),np.float32);x[:,0]=[-2.,-1.,1.,2.];y=np.array([0,0,1,1],np.float32)
    model,check=c.train_probe(x,y,tmp_path/'probe')
    assert check['actual_steps']==500 and check['parameter_steps']==[500,500]
    saved=torch.load(tmp_path/'probe-step500.pt',map_location='cpu',weights_only=False)
    reload=torch.nn.Linear(8,1);reload.load_state_dict(saved['model_state'])
    with torch.no_grad():
        original=model(torch.from_numpy(x));again=reload(torch.from_numpy(x))
    assert torch.equal(original,again)
    assert original[0]<0 and original[-1]>0

def test_saved_normalization_preserves_fp32_pipeline():
    c=core();x=np.array([[100000.,1.0000001],[100001.,1.0000004]],np.float32)
    mean,scale=c.fit_normalizer(x)
    restored=c.normalize_features(x,dict(mean=mean.tolist(),scale=scale.tolist()))
    np.testing.assert_array_equal(restored,(x-mean)/scale)
    assert restored.dtype==np.float32

def test_bad_sha_is_rejected_before_capsule_launch():
    c=core()
    with pytest.raises(ValueError):c.validate_sha('a'*65)
    with pytest.raises(ValueError):c.validate_sha('x'*64)
    assert c.validate_sha('a'*64)=='a'*64
