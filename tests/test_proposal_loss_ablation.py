from pathlib import Path
import importlib.util
import numpy as np
import pytest
import torch
from continuous_mmwave.v25.stage3 import V25Stage3

spec=importlib.util.spec_from_file_location('loss_ab',Path(__file__).parents[1]/'.audit-staging/proposal_loss_ablation.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

def batch():
    torch.manual_seed(8)
    return dict(tokens=torch.randn(1,5,2,4),mask=torch.ones(1,5,2,dtype=torch.bool),
        target=torch.tensor([[0.,.1,.5,1.,.2]]),valid=torch.tensor([[False,True,True,True,True]]),class_weights=(3.,.6))

def test_only_unweighted_route_removes_class_weights_without_mutation():
    b=batch();a=m.loss_batch(b,'weighted');u=m.loss_batch(b,'unweighted')
    assert a is b and b['class_weights']==(3.,.6) and 'class_weights' not in u
    assert set(u)==set(b)-{'class_weights'}
    for key in u:assert u[key] is b[key]
    with pytest.raises(ValueError):m.loss_batch(b,'other')

def test_loss_changes_but_forward_prototype_stays_equal():
    torch.manual_seed(42);a=m.LossStage('weighted',radar_dim=4,event_dim=6,semantic_dim=3)
    b=m.LossStage('unweighted',radar_dim=4,event_dim=6,semantic_dim=3);b.load_state_dict(a.state_dict())
    a.eval();b.eval()
    source=batch();ra=a.physical_stream_step(source);rb=b.physical_stream_step(source)
    reference=V25Stage3(radar_dim=4,event_dim=6,semantic_dim=3);reference.load_state_dict(a.state_dict())
    reference.eval()
    rw=reference.physical_stream_step(source);ru=reference.physical_stream_step({k:v for k,v in source.items() if k!='class_weights'})
    assert torch.equal(ra['loss'],rw['loss']) and torch.equal(rb['loss'],ru['loss'])
    assert not torch.allclose(ra['loss'],rb['loss'])
    for key in ('fast','mean','m2'):assert torch.equal(getattr(ra['next_state'],key),getattr(rb['next_state'],key))
    assert ra['valid_transitions']==rb['valid_transitions']==4

def test_unweighted_bce_matches_formula_and_weighted_formula_differs():
    z=torch.tensor([-2.,-.5,.2,1.5],requires_grad=True);y=torch.tensor([0.,.1,.5,1.])
    plain=V25Stage3._bce(z,y)
    expected=-(y*torch.nn.functional.logsigmoid(z)+(1-y)*torch.nn.functional.logsigmoid(-z)).mean()
    assert torch.allclose(plain,expected)
    weighted=V25Stage3._bce(z,y,class_weights=(3.,.6))
    assert not torch.allclose(plain,weighted)

def test_training_forward_keeps_original_dropout_and_rng_schedule():
    torch.manual_seed(42);a=m.LossStage('weighted',radar_dim=4,event_dim=6,semantic_dim=3)
    reference=V25Stage3(radar_dim=4,event_dim=6,semantic_dim=3);reference.load_state_dict(a.state_dict())
    source=batch();rng=torch.get_rng_state().clone()
    actual=a.physical_stream_step(source);after=torch.get_rng_state().clone()
    torch.set_rng_state(rng);expected=reference.physical_stream_step(source)
    assert a.training and reference.training and torch.equal(actual['loss'],expected['loss'])
    assert torch.equal(after,torch.get_rng_state())

def test_update_counts_and_frozen_nonphysical_parameters():
    torch.manual_seed(42);a=m.LossStage('unweighted',radar_dim=4,event_dim=6,semantic_dim=3)
    frozen={k:v.clone() for k,v in a.state_dict().items() if not k.startswith(('heads.local.','heads.radar.'))}
    params=[p for p in a.parameters() if p.requires_grad];optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.01)
    for step in (1,2):
        row=m.adamw_step(a,lambda stage:stage.physical_stream_step(batch())['loss'],optimizer,step)
        assert row['actual_optimizer_steps']==step and a.optimizer_steps['physical']==step
    assert len(optimizer.state)==20 and all(int(s['step'])==2 for s in optimizer.state.values())
    assert all(torch.equal(v,a.state_dict()[k]) for k,v in frozen.items())

def test_loss_arm_validation_rejects_budget_or_loss_changes():
    m.validate_budget('weighted',500);m.validate_budget('unweighted',500)
    for arm,steps in [('weighted',501),('weighted',1000),('other',500)]:
        with pytest.raises(ValueError):m.validate_budget(arm,steps)

def test_duplicate_output_does_not_overwrite_existing_status(tmp_path,monkeypatch):
    existing=tmp_path/'run';existing.mkdir();p=existing/'status.json';p.write_text('{"phase":"training"}')
    monkeypatch.setattr(m,'OWNED_ROOT',None)
    with pytest.raises(FileExistsError):m.create_output(existing)
    m.record_failure(RuntimeError('duplicate launch'))
    assert p.read_text()=='{"phase":"training"}'
