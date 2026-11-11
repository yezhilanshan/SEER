"""Causality and extraction contracts for isolated MD-GRU diagnosis."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from continuous_mmwave.models.parrad_encoder import PARradCausalMDGRUEncoder
from continuous_mmwave.parrad_config import PARradConfig

PATH = Path(__file__).parents[1]/'.audit-staging/md_gru_signal_core.py'
if not PATH.exists():PATH=Path(__file__).parents[1]/'md_gru_signal_core.py'

def core():
    assert PATH.exists(), 'diagnostic core not implemented'
    spec=importlib.util.spec_from_file_location('md_signal_core', PATH)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module

def encoder():
    torch.manual_seed(42)
    return PARradCausalMDGRUEncoder(PARradConfig(encoder_type='causal_md_gru', md_bins=8,
        md_gru_dim=8, radar_dim=16, motion_dim=8, fusion_heads=4, dropout=0.)).eval().requires_grad_(False)

def test_observer_preserves_tokens_and_captures_raw_recurrence():
    c=core(); m=encoder(); md=torch.randn(1,2,8); rd=torch.empty(1,2,0,0,0)
    with torch.inference_mode():
        expected=m.forward_update(rd,md,sensor_index=torch.tensor([0]))
        with c.Observer(m) as obs:
            actual=m.forward_update(rd,md,sensor_index=torch.tensor([0]))
            values=obs.values(md, actual)
    assert torch.equal(actual.radar_tokens,expected.radar_tokens)
    assert torch.equal(values['recurrent'],actual.state.gru.hidden[0,0])
    assert not torch.equal(values['recurrent'],values['fused'])
    assert tuple(values['token_flat'].shape)==(64,)
    assert not m.gru.gru._forward_hooks

def test_observer_prefix_does_not_depend_on_future():
    c=core(); m=encoder(); stream=torch.randn(1,8,8)
    def run(x):
        state=None; result=[]
        with torch.inference_mode(), c.Observer(m) as obs:
            for i in range(0,8,2):
                md=x[:,i:i+2]; output=m.forward_update(torch.empty(1,2,0,0,0),md,
                    sensor_index=torch.tensor([0]),state=state);state=output.state
                result.append({k:v.clone() for k,v in obs.values(md,output).items()})
        return result
    first=run(stream); changed=stream.clone();changed[:,4:]+=100
    second=run(changed)
    for i in range(2):
        for key in first[i]: assert torch.equal(first[i][key],second[i][key])

def test_causal_design_has_no_future_or_gap_history():
    c=core(); x=np.arange(20,dtype=np.float32)[:,None];time=np.arange(20)*.18;valid=np.ones(20,bool)
    features,mask=c.causal_design(x,valid,time)
    assert not mask[:6].any() and mask[6:].all()
    assert features[6,3]==pytest.approx(3.5) # current minus mean of previous six
    future=x.copy();future[10:]+=1000
    f2,_=c.causal_design(future,valid,time)
    np.testing.assert_array_equal(features[:10],f2[:10])
    time[10:]+=1; _,gap_mask=c.causal_design(x,valid,time)
    assert not gap_mask[10:16].any() and gap_mask[16]
    valid[8]=False;_,unknown=c.causal_design(x,valid,np.arange(20)*.18)
    assert not unknown[8:15].any()

def test_fit_standardization_does_not_consume_dev():
    c=core();fit=np.array([[1.,3.],[3.,7.]],np.float32)
    mean,scale=c.fit_standardizer([fit],[np.ones(2,bool)])
    np.testing.assert_allclose(mean,[2,5]);np.testing.assert_allclose(scale,[1,2])
    result=c.standardize(np.array([[100,100]],np.float32),mean,scale)
    np.testing.assert_allclose(result,[[98,47.5]])

def test_boundary_labels_never_cross_unknown_labels():
    c=core();labels=np.array([[0,0],[-1,-1],[1,1],[1,2],[2,2],[2,2],[2,2],[2,2]])
    time=np.arange(16).reshape(-1,2)*.09
    target=c.labels_for(labels,time)
    np.testing.assert_allclose(target['boundaries'],[.63])
    assert not target['valid'][0] and not target['valid'][1]

def test_standardizer_rejects_missing_fit_observations():
    c=core()
    with pytest.raises(ValueError):c.fit_standardizer([np.ones((2,3))],[np.zeros(2,bool)])

def test_labels_do_not_invent_transition_across_time_gap():
    c=core();labels=np.array([[0,0],[1,1],[1,1],[1,1]])
    time=np.arange(8).reshape(-1,2)*.09;time[1:]+=1
    target=c.labels_for(labels,time)
    assert not len(target['boundaries'])

def test_roles_accept_existing_train_dev_subroles_and_reject_holdouts():
    c=core()
    assert c.require_role({'role':'fit','metadata':{'role':'fit'}})=='fit'
    for subrole in ('dev_regression','dev_new'):
        assert c.require_role({'role':'dev','metadata':{'role':subrole}})==subrole
    with pytest.raises(ValueError):c.require_role({'role':'dev','metadata':{'role':'subject_holdout'}})
    with pytest.raises(ValueError):c.require_role({'role':'fit','metadata':{'role':'dev_new'}})

def test_rank_metrics_respect_ties_and_missing_classes():
    c=core()
    assert c.rank_metrics([1,0,1,0],[.9,.9,.2,.1])==pytest.approx((.625,7/12))
    assert c.rank_metrics([1,0,1,0],[.5]*4)==pytest.approx((.5,.5))
    assert c.rank_metrics([0,0],[.1,.2])==(None,None)

def test_runner_learns_synthetic_signal_without_touching_source_encoder(tmp_path):
    import sys
    sys.path.insert(0,str(PATH.parent))
    try:
        import run_md_gru_signal as runner
        rng=np.random.default_rng(3); x=rng.normal(size=(128,4)).astype(np.float32)
        y=(x[:,0]>0).astype(np.float32); valid=np.ones(128,bool)
        runner.STEPS=500
        scores,check=runner.train_probe([x],[y],[valid],[],'linear',
            [[(0,0),(0,64)]]*500,torch.device('cpu'),tmp_path/'linear')
        assert core().rank_metrics(y,scores[0])[0]>.95
        assert check['optimizer_steps']==[500,500]
        assert check['initial_digest']!=check['final_digest']
    finally:sys.path.pop(0)
