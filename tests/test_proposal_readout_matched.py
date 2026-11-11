"""Contracts for the user-approved frozen readout position experiment."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
import torch

BASE=Path(__file__).parents[1]
DIRECTORY=BASE/'.audit-staging' if (BASE/'.audit-staging').is_dir() else BASE

def core():
    path=DIRECTORY/'proposal_readout_core.py'
    assert path.is_file(), 'matched readout core not implemented'
    sys.path.insert(0,str(DIRECTORY))
    try:
        spec=importlib.util.spec_from_file_location('readout_core',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    finally:sys.path.pop(0)

def test_lossless_alignment_preserves_all_native_coordinates():
    c=core();q=c.rotation();rng=np.random.default_rng(2)
    np.testing.assert_allclose(q.T@q,np.eye(256),atol=1e-6)
    for d in (128,256):
        x=rng.normal(size=(12,d)).astype(np.float32);y=c.align(x,q)
        assert y.shape==(12,256)
        np.testing.assert_allclose((y@q.T)[:,:d]/np.sqrt(256/d),x,atol=2e-6)
        np.testing.assert_allclose(np.sum(y*y,1),np.sum(x*x,1)*256/d,rtol=1e-6)

def test_all_modes_have_identical_initial_trainable_tensors():
    c=core();models=[c.Readout(mode) for mode in c.MODES]
    assert len({sum(p.numel() for p in m.parameters()) for m in models})==1
    first=models[0].state_dict()
    for m in models[1:]:
        assert all(torch.equal(first[k],v) for k,v in m.state_dict().items())
    assert models[0].project.in_features==256 and models[0].project.out_features==128

def test_ablation_masks_after_projection_and_excludes_information():
    c=core();x=torch.randn(7,4,256)
    for mode,excluded in [('state',(2,3)),('delta',(0,1))]:
        model=c.Readout(mode).eval();changed=x.clone();changed[:,excluded,:]+=100
        with torch.inference_mode():
            assert torch.equal(model(x),model(changed))
            z=model.feature_blocks(x)
        assert torch.count_nonzero(z[:,excluded,:])==0

def test_history_is_past_only_and_masks_match_common_context():
    c=core();rng=np.random.default_rng(5);x=rng.normal(size=(20,256)).astype(np.float32)
    t=np.arange(20)*.18;valid=np.ones(20,bool)
    features,mask=c.history(x,valid,t)
    np.testing.assert_allclose(features[9,3],x[9]-x[3:9].mean(0),atol=3e-7)
    assert not mask[:6].any() and mask[6:].all()
    changed=x.copy();changed[10:]+=99;f2,_=c.history(changed,valid,t)
    np.testing.assert_array_equal(features[:10],f2[:10])

def test_weighted_loss_uses_equal_chunks_not_equal_points():
    c=core();z=torch.tensor([0.,2.,-1.],requires_grad=True);y=torch.tensor([0.,1.,.5])
    v=torch.tensor([True,True,True]);sizes=[1,2]
    point=-(3.341091295046995*y*torch.nn.functional.logsigmoid(z)
            +.5879943564065809*(1-y)*torch.nn.functional.logsigmoid(-z))
    actual=c.chunk_loss(z,y,v,sizes)
    expected=(point[0]+point[1:].mean())/2
    torch.testing.assert_close(actual,expected)
    assert not torch.isclose(actual,point.mean())
    actual.backward();assert torch.isfinite(z.grad).all()

def test_empty_chunk_is_rejected_not_silently_reweighted():
    c=core()
    with pytest.raises(ValueError,match='valid'):
        c.chunk_loss(torch.zeros(3),torch.zeros(3),torch.tensor([False,True,True]),[1,2])

def test_primary_guardrail_rejects_dev_fp_confounded_gain():
    c=core();a=dict(recall=.4,ap=.3,latency_p95_s=.5,false_per_minute=5.)
    b=dict(recall=.46,ap=.3,latency_p95_s=.6,false_per_minute=7.)
    result=c.guardrails(a,b,[.01]*8)
    assert result['nominal_pass'] and not result['clear_support']
    assert result['dev_fp_confounded']

def test_six_fit_caps_and_primary_contrast_are_fixed():
    c=core()
    assert c.BUDGETS==(3,5,8,10,15,20)
    assert c.PRIMARY==('token_mean__combined','fused__combined')

def runner():
    path=DIRECTORY/'run_proposal_readout_matched.py'
    assert path.is_file(), 'matched runner not implemented'
    sys.path.insert(0,str(DIRECTORY))
    try:
        spec=importlib.util.spec_from_file_location('readout_runner',path)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    finally:sys.path.pop(0)

def test_schedule_rejects_dev_and_non500_budget():
    r=runner();rows=[dict(role='fit',context=np.ones(128,bool)),dict(role='dev',context=np.ones(128,bool))]
    valid=[[(0,6)]*16 for _ in range(500)]
    r.validate_schedule(valid,rows)
    with pytest.raises(ValueError):r.validate_schedule(valid[:499],rows)
    bad=[x.copy() for x in valid];bad[9][0]=(1,6)
    with pytest.raises(ValueError,match='fit'):r.validate_schedule(bad,rows)

def test_calibration_is_fit_only_and_fast_events_match_original():
    r=runner();rng=np.random.default_rng(6);records=[];scores=[]
    for role in ('fit','dev'):
        t=np.arange(200)*.18;b=t[np.arange(15,185,25)];dist=np.min(np.abs(t[:,None]-b),1)
        label=dict(time=t,boundaries=b,positive=dist<=.36,stable=dist>.72)
        records.append(dict(uid=role,role=role,subject_id='s',sensor_id='c',label=label,context=np.ones(200,bool)))
        scores.append(rng.uniform(size=200))
    first=r.evaluate(records,scores);changed=r.evaluate(records,[scores[0],1-scores[1]])
    assert first['fit_grid']==changed['fit_grid']
    assert [x['fit']['config'] for x in first['events'].values()]==[x['fit']['config'] for x in changed['events'].values()]
    sys.path.insert(0,str(DIRECTORY))
    try:
        from proposal_causal_triggers import emit,measure
        config=first['events']['8']['fit']['config'];a=records[0]
        expected=measure(a['label']['time'],a['context'],a['label']['boundaries'],emit(a['label']['time'],scores[0],a['context'],config))
        actual=r.metrics_for(a,r.candidates(a,scores[0]),config['high'])
        for key in ('recall','false_per_minute','latency_p95_s','proposals'):assert actual[key]==pytest.approx(expected[key])
    finally:sys.path.pop(0)

def test_runner_actual500steps_equal_initialization_and_no_source_mutation(tmp_path):
    r=runner();rng=np.random.default_rng(8);x=rng.normal(size=(128,4,256)).astype(np.float32)
    y=(x[:,0,0]>0).astype(np.float32);mask=np.ones(128,bool);original=x.copy()
    score,check=r.train_arm([x],[y],[mask],'combined',[[(0,6)]*16]*500,torch.device('cpu'),tmp_path/'combined')
    assert check['actual_steps']==500 and check['parameter_steps']==[500]*6
    assert check['initial_digest']!=check['final_digest']
    assert (tmp_path/'combined-step0.pt').is_file() and (tmp_path/'combined-step500.pt').is_file()
    np.testing.assert_array_equal(x,original)
    assert core().Readout('combined')(torch.zeros(1,4,256)).shape==(1,)
    assert len(score[0])==128

def test_cache_crosschecks_detect_token_fused_and_raw_mismatch():
    r=runner();rng=np.random.default_rng(9);projected=rng.normal(size=(5,128)).astype(np.float32)
    recurrent=rng.normal(size=(5,128)).astype(np.float32);norm=torch.nn.LayerNorm(128)
    with torch.inference_mode():fused=norm(torch.from_numpy(projected+recurrent)).numpy()
    radar=torch.randn(5,4,256);md=rng.normal(size=(10,128)).astype(np.float32)
    layers=dict(projected=projected,recurrent=recurrent,fused=fused,token_mean=radar.mean(1).numpy(),raw_md=md[1::2])
    result=r.crosscheck_layers(layers,radar,md,norm)
    assert result['token_mean_max_error']==0 and result['raw_md_max_error']==0
    for key in ('fused','token_mean','raw_md'):
        bad={k:v.copy() for k,v in layers.items()};bad[key][2,0]+=.1
        with pytest.raises(ValueError):r.crosscheck_layers(bad,radar,md,norm)
