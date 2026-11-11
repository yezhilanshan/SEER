import importlib
import math
import torch
import pytest


def api(): return importlib.import_module('continuous_mmwave.v261.training')


def test_joint_losses_masks_reductions_and_frozen_teacher():
    outputs=dict(mean=torch.ones(3,3,256,requires_grad=True),
                 logits=torch.zeros(3,14,requires_grad=True),
                 old_embedding=torch.tensor([[1.,0.]]*3,requires_grad=True),
                 old_logits=torch.zeros(3,14,requires_grad=True))
    teacher=dict(embedding=torch.tensor([[1.,0.]]*3,requires_grad=True),logits=torch.zeros(3,14,requires_grad=True))
    batch=dict(target=torch.zeros(3,3,256),semantic=torch.tensor([0,-100,2]),recording_ids=torch.tensor([0,0,1]))
    stats=dict(R=torch.tensor([1.,2.,4.]))
    loss=api().joint_losses(outputs,teacher,batch,stats,'AB')
    assert loss['semantic_count']==2 and loss['motion_count']==3
    assert float(loss['motion'].detach())==pytest.approx(.1+.3/2+.6/4)
    assert float(loss['semantic'].detach())==pytest.approx(1.)
    assert float(loss['preserve'].detach())==pytest.approx(0.,abs=1e-7)
    loss['total'].backward()
    assert teacher['embedding'].grad is None and teacher['logits'].grad is None
    assert outputs['logits'].grad[1].count_nonzero()==0
    batch['semantic'].fill_(-100)
    b=api().joint_losses(outputs,teacher,batch,stats,'B')
    assert not b['has_signal'] and b['semantic_count']==0


def test_equal_recording_loss_is_not_naive_batch_mean():
    output={'mean':torch.tensor([1.,1.,3.])[:,None,None].expand(3,3,256)}
    batch=dict(target=torch.zeros(3,3,256),semantic=torch.full((3,),-100),recording_ids=torch.tensor([0,0,1]))
    loss=api().joint_losses(output,{},batch,{'R':torch.ones(3)},'A')
    assert float(loss['total'])==pytest.approx(5.)
    with pytest.raises(ValueError,match='finite|positive'):
        api().joint_losses(output,{},batch,{'R':torch.zeros(3)},'A')


def test_attempt_does_not_count_amp_overflow_as_optimizer_update():
    class OverflowScaler:
        def __init__(self): self.value=8.
        def is_enabled(self):return True
        def get_scale(self):return self.value
        def scale(self,x):return x
        def unscale_(self,opt):pass
        def step(self,opt):pass
        def update(self):self.value/=2
    p=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.SGD([p],lr=.1)
    success,norm=api().optimizer_attempt(p.square(),optimizer,[p],OverflowScaler(),1.)
    assert not success and float(p.detach())==1.
    optimizer.zero_grad()
    success,norm=api().optimizer_attempt(p.square(),optimizer,[p],None,1.)
    assert success and float(p.detach())<1.


def test_tiny_smoke_counts_saved_updates_and_preserves_frozen_weights(tmp_path):
    from test_v261_contract import fixture
    from test_v261_model import frontend
    from continuous_mmwave.v261.contract import build_joint_record,fit_motion_contract,joint_schedule
    from continuous_mmwave.v261.model import JointPilotModel
    cache,labels,physical=fixture(n=14)
    record=build_joint_record(cache,labels,physical,'train')
    stats=fit_motion_contract([record],dict(mean=torch.zeros(256),std=torch.ones(256)))
    model=JointPilotModel(frontend(),'AB')
    options=dict(steps=2,max_attempts=2,lr=1e-4,weight_decay=.01,clip_norm=1.,smoke=True)
    result=api().train_joint_job(model,[record],stats,joint_schedule([record],42,steps=2),options,tmp_path/'job')
    assert result['actual_optimizer_steps']==2 and result['frozen_unchanged']
    payload=torch.load(tmp_path/'job'/'step0002.pt',weights_only=False)
    assert payload['actual_optimizer_steps']==2 and set(payload['parameter_steps'].values())=={2}


def test_checkpoint_checks_identity_complete_parameter_names_and_optimizer_state():
    import copy
    model=torch.nn.Linear(2,2);model.arm='A';model.seed=42
    optimizer=torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).sum().backward();optimizer.step()
    payload=dict(arm='A',seed=42,actual_optimizer_steps=1,
                 learned_state={k:v.detach().clone() for k,v in model.named_parameters()},
                 parameter_steps={k:1 for k,v in model.named_parameters()},optimizer_state=optimizer.state_dict())
    api().validate_optimizer_checkpoint(payload,model,1)
    for key,value in [('seed',43),('arm','B'),('parameter_steps',{})]:
        bad=copy.deepcopy(payload);bad[key]=value
        with pytest.raises(ValueError,match='checkpoint|optimizer'):
            api().validate_optimizer_checkpoint(bad,model,1)
    bad=copy.deepcopy(payload);bad['optimizer_state']['state'][0]['step']=torch.tensor(0.)
    with pytest.raises(ValueError,match='optimizer'):
        api().validate_optimizer_checkpoint(bad,model,1)


def test_fp16_shared_gradient_diagnostic_uses_amp_scale_to_avoid_underflow():
    from types import SimpleNamespace
    adapter=torch.nn.Linear(1,1,bias=False)
    loss=(adapter(torch.ones(1,1)).half()*1e-4*1e-4).float().sum()
    result=api().gradient_diagnostic(SimpleNamespace(shared_adapter=adapter),{'motion':loss,'semantic':loss*0},scale=256.)
    assert result['motion_norm']>0


def test_budget_validation_and_checkpoint_parameter_whitelist():
    api().validate_training_options(dict(steps=1000,max_attempts=1100,lr=1e-4,weight_decay=.01,clip_norm=1.),smoke=False)
    with pytest.raises(ValueError,match='budget'):
        api().validate_training_options(dict(steps=999,max_attempts=1100,lr=1e-4,weight_decay=.01,clip_norm=1.),smoke=False)
