import importlib
import numpy as np
import pytest
import torch


def api(): return importlib.import_module('continuous_mmwave.v26.long_prediction')


@pytest.mark.parametrize('kind',['gru','tcn','event','event_free'])
def test_zero_residual_and_history_sequence_matches_every_prefix(kind):
    cfg=dict(motion_dim=4,radar_dim=3,event_dim=8,hidden=8,head_width=12,dilations=[1,2,4,8])
    torch.manual_seed(13);model=api().MatchedResidualPredictor(kind,cfg).eval()
    motion=torch.randn(1,13,4);radar=torch.randn(1,13,4,3);events=torch.randn(1,13,2,8)
    mask=torch.ones(1,13,dtype=torch.bool)
    mu,lv=model.forward_sequence(motion,radar,events,mask)
    assert torch.equal(mu,motion)
    torch.testing.assert_close(lv,torch.zeros_like(lv),rtol=0,atol=1e-6)
    with torch.no_grad():model.head[-1].weight[:4].normal_(std=.1)
    mu,lv=model.forward_sequence(motion,radar,events,mask)
    for t in (0,3,7,12):
        a,b=model(motion[:,t],radar[:,:t+1],events[:,t],mask[:,:t+1])
        torch.testing.assert_close(a,mu[:,t],rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(b,lv[:,t],rtol=1e-6,atol=1e-6)
    changed=radar.clone();changed[:,8:]=999
    after,_=model.forward_sequence(motion,changed,events,mask)
    torch.testing.assert_close(after[:,:8],mu[:,:8],rtol=0,atol=0)


def test_long_context_and_padding_no_silent_32_step_truncation():
    cfg=dict(motion_dim=4,radar_dim=3,event_dim=8,hidden=8,head_width=12,dilations=[1,2,4,8,16,32])
    for kind in ('gru','tcn'):
        torch.manual_seed(1);model=api().MatchedResidualPredictor(kind,cfg)
        radar=torch.randn(2,65,4,3);mask=torch.ones(2,65,dtype=torch.bool);mask[1,40:]=False
        context=model.history_context(radar,mask)
        short=model.history_context(radar[1:2,:40],mask[1:2,:40])
        torch.testing.assert_close(context[1,39],short[0,-1],atol=1e-6,rtol=1e-6)
        assert context.shape==(2,65,4)
        invalid=mask.clone();invalid[1,8]=False
        with pytest.raises(ValueError,match='right-padded'):model.history_context(radar,invalid)


def test_architecture_covers_entire_history_and_matches_budget():
    for maximum in (334,3665):
        cfg=api().matched_architecture('tcn',maximum)
        assert 1+sum(cfg['dilations'])>=maximum
        model=api().MatchedResidualPredictor('tcn',cfg)
        assert abs(sum(p.numel() for p in model.parameters())-400000)/400000<.02


def test_event_ablation_changes_only_event_input_and_uses_full_history():
    record=dict(motion=torch.arange(48).float().reshape(12,4),radar=torch.arange(144).float().reshape(12,4,3),
        events=torch.arange(192).float().reshape(12,2,8),segment_start=np.array([0]*8+[8]*4),
        selected=np.array([5,7,10]),shuffle_donor=np.array([-1]*10+[2,-1]))
    stats=dict(mean=torch.zeros(4),std=torch.ones(4))
    # Radar dimensions in real data equal motion width; use compatible fake tensors.
    record['radar']=torch.randn(12,4,4)
    a=api().make_long_batch([record],stats,[(0,2)],1,'event','cpu')
    b=api().make_long_batch([record],stats,[(0,2)],1,'event_shuffle','cpu')
    c=api().make_long_batch([record],stats,[(0,2)],1,'event_zero','cpu')
    assert a['radar_prefix'].shape[1]==3
    assert torch.equal(a['event_tokens'][0],record['events'][10])
    assert torch.equal(b['event_tokens'][0],record['events'][2])
    assert c['event_tokens'].count_nonzero()==0
    for key in ('motion','target','radar_prefix','prefix_mask'):
        assert torch.equal(a[key],b[key]) and torch.equal(a[key],c[key])
    record['selected']=np.array([7]);d=api().make_long_batch([record],stats,[(0,0)],3,'gru','cpu')
    assert d['radar_prefix'].shape[1]==8 and torch.equal(d['target'][0],record['motion'][10])


def test_state_diagnostics_detect_dominating_coordinate_without_claiming_constant():
    x=torch.randn(64,2,8)*.1;x[16:,:,0]=100
    result=api().state_dynamics(x)
    assert result['first_sustained_high']==16
    assert np.all(result['top_coordinate_energy'][16:]>.99)
    assert np.all(result['step_delta_rms'][17:]>0)


def test_registered_jobs_schedule_and_state_population_are_explicit():
    module=importlib.import_module('continuous_mmwave.v26.p1c_train')
    jobs=module.registered_jobs()
    assert len(jobs)==45 and len({(r['population'],r['horizon'],r['arm']) for r in jobs})==45
    assert not any(r['population']=='no_reset_full' and r['arm']=='event_shuffle' for r in jobs)
    assert sum(r['arm']=='event_zero' for r in jobs)==6


def test_native_prediction_uses_complete_segments_and_matches_independent_prefix():
    module=importlib.import_module('continuous_mmwave.v26.p1c_train')
    n=45;record=dict(motion=torch.randn(n,4),radar=torch.randn(n,4,4),events=torch.randn(n,2,8),
        selected=np.array([1,7,30,32,39]),segment_start=np.array([0]*31+[31]*14),shuffle_donor=np.full(n,-1))
    stats=dict(mean=torch.zeros(4),std=torch.ones(4))
    for kind in ('gru','tcn','event'):
        cfg=dict(motion_dim=4,radar_dim=4,event_dim=8,hidden=8,head_width=12,dilations=[1,2,4,8,16,32])
        model=api().MatchedResidualPredictor(kind,cfg).eval()
        with torch.no_grad():model.head[-1].weight[:4].normal_(std=.1)
        mean,lv=module.predict_long_record(model,record,stats,kind,'cpu')
        batch=api().make_long_batch([record],stats,[(0,i) for i in range(5)],1,kind,'cpu')
        a,b=model(batch['motion'],batch['radar_prefix'],batch['event_tokens'],batch['prefix_mask'])
        torch.testing.assert_close(mean,a,rtol=1e-6,atol=1e-6)
        torch.testing.assert_close(lv,b,rtol=1e-6,atol=1e-6)


def test_training_rejects_short_or_expanded_budget_before_writing(tmp_path):
    module=importlib.import_module('continuous_mmwave.v26.p1c_train')
    options=dict(steps=300,lr=.001,weight_decay=.01,clip_norm=1.)
    with pytest.raises(ValueError,match='fixed budget'):
        module.train_job(None,[],{},options,[[]],1,'gru',tmp_path/'invalid','cpu')
    assert not (tmp_path/'invalid').exists()


def test_saved_prediction_verification_rejects_tampered_baseline_and_alignment():
    module=importlib.import_module('continuous_mmwave.v26.p1c_train')
    assert hasattr(module,'verify_prediction_arrays'), 'complete baseline/index verification missing'
    selected=np.array([1,3]);mean=np.zeros((2,4),np.float32);lv=mean.copy();values=dict(mse=np.zeros(2))
    saved=dict(mean=mean.copy(),logvar=lv.copy(),selected=selected.copy(),future=selected+3,mse=np.zeros(2))
    module.verify_prediction_arrays(saved,mean,lv,selected,3,values)
    for key in ('mean','logvar','selected','future','mse'):
        broken={k:v.copy() for k,v in saved.items()};broken[key].flat[0]+=1
        with pytest.raises((AssertionError,ValueError)):module.verify_prediction_arrays(broken,mean,lv,selected,3,values)
    trend={k:v for k,v in saved.items() if k!='logvar'}
    module.verify_prediction_arrays(trend,mean,None,selected,3,values)
    with pytest.raises((AssertionError,ValueError)):module.verify_prediction_arrays(saved,mean,None,selected,3,values)


def test_saved_targets_groups_verification_rejects_changed_truth():
    module=importlib.import_module('continuous_mmwave.v26.p1c_train')
    assert hasattr(module,'verify_target_arrays'), 'saved targets verification missing'
    expected=dict(current=np.ones((2,4)),target=np.zeros((2,4)),selected=np.array([1,3]),
        future=np.array([4,6]),age=np.array([2,4]),state_rms=np.array([.5,1.]),all=np.ones(2,bool))
    module.verify_target_arrays(expected,expected)
    for key in expected:
        broken={k:v.copy() for k,v in expected.items()};broken[key].flat[0]=not broken[key].flat[0] if broken[key].dtype==bool else broken[key].flat[0]+1
        with pytest.raises((AssertionError,ValueError)):module.verify_target_arrays(broken,expected)


def test_preparation_requires_successful_matching_zero_step_diagnosis(tmp_path):
    import json
    module=importlib.import_module('continuous_mmwave.v26.p1c_train')
    assert hasattr(module,'begin_preparation'), 'diagnosis prerequisite and one-shot guard missing'
    (tmp_path/'protocol.json').write_text(json.dumps(dict(code_sha256='code',config_sha256='config')))
    with pytest.raises((ValueError,FileNotFoundError)):module.begin_preparation(tmp_path,'frozen')
    assert not (tmp_path/'preparation-started.json').exists()
    folder=tmp_path/'state-diagnosis';folder.mkdir()
    summary=dict(records=[dict(uid=str(i//2),view=str(i%2)) for i in range(48)],frontend_unchanged=True,
        frontend_sha256='frozen',optimizer_steps=0,code_sha256='code',config_sha256='config',gate_probes=[dict(state_exact=True)])
    for key,bad in [('frontend_unchanged',False),('frontend_sha256','wrong'),('optimizer_steps',1),('code_sha256','wrong')]:
        value=dict(summary);value[key]=bad;(folder/'summary.json').write_text(json.dumps(value))
        with pytest.raises(ValueError):module.begin_preparation(tmp_path,'frozen')
        assert not (tmp_path/'preparation-started.json').exists()
    (folder/'summary.json').write_text(json.dumps(summary));module.begin_preparation(tmp_path,'frozen')
    previous=(tmp_path/'preparation-started.json').read_bytes()
    with pytest.raises(FileExistsError):module.begin_preparation(tmp_path,'frozen')
    assert (tmp_path/'preparation-started.json').read_bytes()==previous
