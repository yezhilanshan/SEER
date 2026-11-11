import importlib
import numpy as np
import pytest
import torch


def contract_api():
    module = importlib.import_module('continuous_mmwave.v26.targets')
    assert hasattr(module, 'prediction_contract'), 'separate observation/event contract missing'
    return module.prediction_contract


def test_mixed_update_keeps_observation_and_deadline_crossing():
    fn = contract_api()
    intervals = np.column_stack((np.arange(8)*.18,(np.arange(8)+1)*.18))
    value = fn(intervals,np.ones(8,bool),np.array([0.,.27]),(1,3,5))
    assert value['observation_valid'].all()
    assert value['event_assignment_valid'].tolist() == [True,False,True,True,True,True,True,True]
    assert value['proxy_episode_id'][1] == -1
    assert value['future'][1].tolist() == [1,2,3,4,5,6,7,-1]
    assert value['proxy_boundary_cross'][1].tolist() == [True,False,False,False,False,False,False,False]
    assert value['mixed_current'][1][1] and value['mixed_future'][1][0]
    assert value['common_start'].tolist() == [0,1,2]


def test_deadline_equality_gaps_invalid_observations_and_recording_reset():
    fn = contract_api()
    intervals = np.column_stack((np.arange(6)*.18,(np.arange(6)+1)*.18))
    value = fn(intervals,np.ones(6,bool),np.array([0.,.36]),(1,))
    assert value['event_assignment_valid'].all()
    assert value['proxy_boundary_cross'][1].tolist() == [True,False,False,False,False,False]
    valid=np.ones(6,bool);valid[2]=False
    assert fn(intervals,valid,[0.,.27],(1,3))['future'][1].tolist() == [1,-1,-1,4,5,-1]
    intervals[3:] += .09
    assert fn(intervals,np.ones(6,bool),[0.,.27],(1,3))['future'][1][2] == -1
    with pytest.raises(ValueError,match='recording'):
        fn(np.concatenate((intervals,intervals)),np.ones(12,bool),[0.],(1,))


def residual_api():
    module=importlib.import_module('continuous_mmwave.v26.predictors')
    assert hasattr(module,'make_residual_predictor'), 'zero-initialized residual predictor missing'
    return module


def test_all_residual_arms_identity_variance_capacity_and_first_gradient():
    module=residual_api()
    config=dict(motion_dim=8,radar_dim=8,event_dim=16,hidden=8,head_width=12,logvar_min=-6.,logvar_max=2.)
    motion=torch.randn(2,8);prefix=torch.randn(2,4,4,8);event=torch.randn(2,8,16)
    mask=torch.ones(2,4,dtype=torch.bool)
    for kind in ('event_free','gru','tcn','event'):
        torch.manual_seed(42); direct=module.make_predictor(kind,config)
        torch.manual_seed(42); model=module.make_residual_predictor(kind,config)
        assert sum(p.numel() for p in direct.parameters()) == sum(p.numel() for p in model.parameters())
        mean,var=model(motion,prefix,event,mask)
        assert torch.equal(mean,motion)
        assert torch.equal(var,direct(motion,prefix,event,mask)[1])
        (mean-motion-1).square().mean().backward()
        assert model.head[-1].weight.grad[:8].abs().sum() > 0
        assert model.head[0].weight.grad.abs().sum() == 0


def test_residual_forward_remains_causal_after_learning_output_weights():
    module=residual_api()
    config=dict(motion_dim=8,radar_dim=8,event_dim=16,hidden=8,head_width=12)
    motion=torch.randn(2,8);prefix=torch.randn(2,5,4,8);event=torch.randn(2,8,16)
    mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
    for kind in ('event_free','gru','tcn','event'):
        model=module.make_residual_predictor(kind,config)
        with torch.no_grad(): model.head[-1].weight[:8].normal_()
        baseline=model(motion,prefix,event,mask)[0]
        changed=prefix.clone();changed[0,2:]+=100
        torch.testing.assert_close(model(motion,changed,event,mask)[0][0],baseline[0],rtol=0,atol=0)


def test_new_pair_audit_preserves_old_pairs_without_fabricating_event_states():
    import importlib.util
    assert importlib.util.find_spec('continuous_mmwave.v26.p1') is not None, 'P1 audit runner missing'
    module=importlib.import_module('continuous_mmwave.v26.p1')
    intervals=torch.tensor(np.column_stack((np.arange(8)*.18,(np.arange(8)+1)*.18)))
    record=dict(intervals=intervals,radar=torch.ones(8,4,8),radar_mask=torch.ones(8,4,dtype=torch.bool),
                motion=torch.ones(8,8),valid=torch.tensor([1,0,1,1,1,1,1,1],dtype=torch.bool),
                selected=torch.tensor([2,3,4]),future=torch.tensor([3,4,5]),event=torch.zeros(3,8,16))
    arrays,report=module.audit_prediction_record(record,np.array([0.,.27]))
    assert report['horizons']['1']['pairs'] == 7
    assert report['horizons']['1']['deadline_cross'] == 1
    assert report['horizons']['1']['restored_vs_old_full'] == 2
    assert report['old_selected_preserved'] == 3
    assert report['horizons']['1']['missing_cached_event'] == 4
    assert arrays['proxy_episode_id'][1] == -1


def test_future_target_perturbation_does_not_change_current_head_inputs():
    from continuous_mmwave.v26.prediction_runner import make_batch
    record=dict(radar=torch.randn(4,4,8),motion=torch.randn(4,8),event=torch.randn(1,8,16),
                selected=torch.tensor([1]),future=torch.tensor([2]),prefix_start=torch.tensor([0]))
    stats=dict(mean=torch.zeros(8),std=torch.ones(8))
    first=make_batch([record],stats,[(0,0)],'cpu')
    record['radar'][2:]+=100;record['motion'][2:]+=100
    second=make_batch([record],stats,[(0,0)],'cpu')
    for key in ('motion','radar_prefix','event_tokens','prefix_mask'):
        assert torch.equal(first[key],second[key])
    assert not torch.equal(first['target'],second['target'])


def test_schedule_roundtrip_keeps_exact_order_and_rejects_changed_position():
    import json
    module=importlib.import_module('continuous_mmwave.v26.p1')
    assert hasattr(module,'require_same_schedule'), 'JSON schedule canonicalization missing'
    schedule=[[(0,1),(1,2)],[(2,0),(0,3)]]
    module.require_same_schedule(schedule,json.loads(json.dumps(schedule)))
    with pytest.raises(ValueError,match='schedule'):
        module.require_same_schedule(schedule,[[[0,1],[1,2]],[[2,0],[0,2]]])


def test_residual_training_saves_fixed_checkpoints_and_exact_optimizer_counts(tmp_path):
    import importlib.util
    assert importlib.util.find_spec('continuous_mmwave.v26.p1_training') is not None,'P1 fixed checkpoint trainer missing'
    module=importlib.import_module('continuous_mmwave.v26.p1_training')
    api=residual_api();cfg=dict(motion_dim=8,radar_dim=8,event_dim=16,hidden=8,head_width=12)
    record=dict(uid='toy',metadata=dict(subject_id='person'),radar=torch.randn(4,4,8),motion=torch.randn(4,8),
                selected=torch.tensor([0,1,2]),future=torch.tensor([1,2,3]),prefix_start=torch.tensor([0,0,0]),
                age=torch.tensor([1,2,3]),event=torch.randn(3,8,16))
    model=api.make_residual_predictor('event_free',cfg)
    options=dict(steps=2,batch_size=1,seed=42,lr=.001,weight_decay=.01,clip_norm=1.)
    result=module.train_residual_arm(model,[record],dict(mean=torch.zeros(8),std=torch.ones(8)),options,
                                     [[(0,0)],[(0,1)]],tmp_path/'arm','cpu',checkpoints=(0,1,2))
    assert set(result['checkpoints']) == {'0','1','2'}
    assert result['actual_optimizer_steps'] == 2
    payload=torch.load(tmp_path/'arm/step002.pt',weights_only=False)
    assert all(int(row['step']) == 2 for row in payload['optimizer_state']['state'].values())
    assert result['fit_probes']['0']['subject_macro']['mse'] == result['fit_probes']['0']['subject_macro']['persistence_mse']
