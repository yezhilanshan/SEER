import importlib.util,sys
from pathlib import Path
import numpy as np
import pytest
import torch

BASE=Path(__file__).parents[1];DIR=BASE/'.audit-staging' if (BASE/'.audit-staging').is_dir() else BASE
def core():
    path=DIR/'stage3ag_residual_core.py';assert path.is_file(),'residual core not implemented'
    sys.path.insert(0,str(DIR))
    try:
        spec=importlib.util.spec_from_file_location('residual_core',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    finally:sys.path.pop(0)

def test_zero_init_exact_logits_and_identical_capacity_control():
    c=core();a=c.Residual('A');b=c.Residual('B');h=torch.randn(31,2,128);z=torch.randn(31);has=torch.ones(31,dtype=torch.bool)
    assert sum(p.numel() for p in a.parameters())==16641
    assert all(torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
    for model in (a,b):
        actual,delta=model(h,z,has);assert torch.equal(actual,z) and torch.count_nonzero(delta)==0
        assert torch.count_nonzero(model.output.weight)==0 and torch.count_nonzero(model.output.bias)==0

def test_capacity_control_ignores_real_gru_after_training_step():
    c=core();m=c.Residual('A');torch.nn.init.normal_(m.output.weight);h=torch.randn(20,2,128);z=torch.randn(20);has=torch.ones(20,dtype=torch.bool)
    assert torch.equal(m(h,z,has)[0],m(h+100,z,has)[0])

def test_missing_history_bypasses_computation_even_with_nan_placeholder():
    c=core();m=c.Residual('B');torch.nn.init.normal_(m.output.weight);h=torch.randn(9,2,128);z=torch.randn(9);has=torch.ones(9,dtype=torch.bool);has[0]=False;h[0]=float('nan')
    final,delta=m(h,z,has);assert torch.isfinite(final).all() and final[0]==z[0] and delta[0]==0

def test_pairing_reset_prefix_and_chunk_history_continuity():
    c=core();rng=np.random.default_rng(2);h=rng.normal(size=(30,128)).astype(np.float32);t=np.arange(30)*.18
    paired,has=c.pair_states(h,t)
    assert not has[0] and has[1:].all();np.testing.assert_array_equal(paired[16,0],h[15])
    short,_=c.pair_states(h[:16],t[:16]);np.testing.assert_array_equal(paired[:16],short)
    nextrecord,nexthas=c.pair_states(h[16:],t[16:]);assert not nexthas[0]
    # Use FP64 for the mathematical chunk-equivalence test: FP32 GEMM may
    # round differently when batch sizes change. Real training remains FP32.
    m=c.Residual('B').double();torch.nn.init.normal_(m.output.weight);z=torch.randn(30,dtype=torch.float64)
    first=m(torch.from_numpy(paired).double(),z,torch.from_numpy(has))[0]
    changed=paired.copy();changed[20:]+=100
    second=m(torch.from_numpy(changed).double(),z,torch.from_numpy(has))[0];torch.testing.assert_close(first[:20],second[:20],rtol=0,atol=1e-12)
    chunk=m(torch.from_numpy(paired[16:]).double(),z[16:],torch.from_numpy(has[16:]))[0];torch.testing.assert_close(first[16:],chunk,rtol=0,atol=1e-12)

def test_zero_output_gradient_progression_and_control_zero_projection_weight():
    c=core();h=torch.randn(32,2,128);z=torch.randn(32);has=torch.ones(32,dtype=torch.bool)
    for mode in ('A','B'):
        m=c.Residual(mode);o=torch.optim.AdamW(m.parameters(),lr=.001)
        for step in (1,2):
            o.zero_grad();final,_=m(h,z,has);torch.nn.functional.binary_cross_entropy_with_logits(final,torch.ones(32)).backward()
            g=c.gradient_norms(m)
            assert all(np.isfinite(v) for v in g.values())
            if step==1:assert g['project']==0 and g['input']==0
            if mode=='B' and step==2:assert g['project']>0 and g['input']>0
            if mode=='A':assert g['project.weight']==0
            o.step()

def test_original_weighted_bce_mass_normalization_and_equal_chunks():
    c=core();from continuous_mmwave.v25.stage3 import V25Stage3
    z=torch.tensor([1.,-2.,.3,1.4]);y=torch.tensor([0.,.3,.7,1.]);v=torch.ones(4,dtype=torch.bool)
    expected=torch.stack([V25Stage3._bce(z[:1],y[:1],class_weights=c.WEIGHTS),V25Stage3._bce(z[1:],y[1:],class_weights=c.WEIGHTS)]).mean()
    torch.testing.assert_close(c.original_loss(z,y,v,[1,3]),expected)

def test_oracle_union_and_disjoint_rescue_categories():
    c=core();a=np.array([1,1,0,0],bool);b=np.array([1,0,1,0],bool)
    assert c.hit_matrix(a,b)==dict(both=1,radar_only=1,gru_only=1,neither=1,total=4,oracle_union_recall=.75)
    nopeak=dict(category='peak_shape',stages_in_matching_window={'accepted':0})
    assert c.rescue_category(nopeak,dict(category='peak_shape'))=='old_zero_peak_shape'
    haspeak=dict(category='matched',stages_in_matching_window={'accepted':1})
    assert c.rescue_category(haspeak,dict(category='below_threshold'))=='old_has_peak_threshold_filtered'

def analysis():
    path=DIR/'stage3ag_analysis.py';assert path.is_file(),'residual analysis not implemented'
    sys.path.insert(0,str(DIR))
    try:
        spec=importlib.util.spec_from_file_location('residual_analysis',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    finally:sys.path.pop(0)

def test_boundary_overlap_keeps_record_identity_and_excludes_unscored():
    a=analysis();rows=[dict(role='dev',subject_id='s',sensor_id='c',transition_type='0->1',eligible=True,radar_hit=True,gru_hit=False,radar_failure='matched',gru_failure='peak_shape'),
                      dict(role='dev',subject_id='s',sensor_id='c',transition_type='1->2',eligible=True,radar_hit=False,gru_hit=True,radar_failure='below_threshold',gru_failure='matched'),
                      dict(role='dev',subject_id='s',sensor_id='c',transition_type='2->3',eligible=False,radar_hit=False,gru_hit=False,radar_failure='excluded',gru_failure='excluded')]
    result=a.overlap_summary(rows)
    assert result['dev']['all']['total']==2 and result['dev']['all']['oracle_union_recall']==1
    assert result['dev']['all']['gru_only']==1 and result['dev']['excluded']==1

def test_rescue_harm_does_not_double_count_peak_shape_subset():
    a=analysis();rows=[dict(role='dev',eligible=True,C_hit=False,B_hit=True,rescue_group='old_zero_peak_shape'),
                      dict(role='dev',eligible=True,C_hit=False,B_hit=True,rescue_group='old_zero_no_legal_peak_other'),
                      dict(role='dev',eligible=True,C_hit=True,B_hit=False,rescue_group=None)]
    out=a.rescue_summary(rows);assert out['dev']['rescue']==2 and out['dev']['harm']==1 and out['dev']['net']==1
    assert out['dev']['old_zero_no_legal_peak_rescue']==2

def runner():
    path=DIR/'run_stage3ag_residual.py';assert path.is_file(),'residual runner not implemented'
    sys.path.insert(0,str(DIR))
    try:
        spec=importlib.util.spec_from_file_location('residual_runner',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
    finally:sys.path.pop(0)

def test_training_is_exact500steps_preserves_inputs_and_saves100(tmp_path):
    r=runner();rng=np.random.default_rng(4);h=rng.normal(size=(128,2,128)).astype(np.float32)
    z=rng.normal(size=128).astype(np.float32);y=(h[:,1,0]>0).astype(np.float32);valid=np.ones(128,bool)
    saved=h.copy();schedule=[[(0,6)]*16]*500
    score,final,delta,check=r.train_arm([h],[z],[np.ones(128,bool)],[y],[valid],'B',schedule,torch.device('cpu'),tmp_path/'B')
    assert check['actual_steps']==500 and check['parameter_steps']==[500]*6
    assert check['gradient_checks']['1']['project']==0 and check['gradient_checks']['2']['project']>0
    assert all((tmp_path/f'B-step{x}.pt').exists() for x in (0,100,500))
    np.testing.assert_array_equal(h,saved);np.testing.assert_allclose(final[0],z+delta[0],atol=0,rtol=0)

def test_missing_auxiliary_or_cell_metrics_are_explicit_no_go():
    c=core()
    def result():
        m=dict(recall=.5,latency_p95_s=.5,false_per_minute=5.)
        return dict(score={'dev':{'ap':.5}},events={str(b):dict(dev=dict(m),groups={f's{i}/radar':dict(m) for i in range(8)}) for b in (3,5,8)})
    results={k:result() for k in ('C','A','B')};results['B']['events']['3']=None
    assert c.success(results,[])['assessed'] is False and c.success(results,[])['go'] is False
    results={k:result() for k in ('C','A','B')};results['B']['events']['8']['groups']['s0/radar']['recall']=None
    assert c.success(results,[])['assessed'] is False
