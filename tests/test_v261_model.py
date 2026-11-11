import copy
import importlib
import torch
import pytest
from torch import nn

torch.set_num_threads(1)


def frontend():
    from continuous_mmwave.models.event_former import TinyLLMAdapter, StreamingSemanticEventFormer
    from continuous_mmwave.models.semantic import SemanticProjection
    from continuous_mmwave.parrad_config import PARradConfig
    torch.manual_seed(8)
    cfg=PARradConfig(llm_dim=16,llm_heads=2,llm_layers=2,cross_attn_dim=8,cross_attn_heads=2,
                     semantic_dim=8,lora_rank=2,dropout=0.,cross_attention_init_scale=.1)
    result=nn.Module()
    result.event_former=StreamingSemanticEventFormer(TinyLLMAdapter(32,16,2,2),cfg,torch.tensor([1,2,3]))
    result.semantic_projection=SemanticProjection(16,8)
    result.auxiliary_classifier=nn.Linear(8,14)
    result.requires_grad_(False).eval()
    return result


def api(): return importlib.import_module('continuous_mmwave.v261.model')


def inputs(n=7):
    torch.manual_seed(3)
    F=torch.randn(2,n,4,256);mask=torch.ones(2,n,dtype=torch.bool)
    return F,mask,F[:,-1].mean(1)


def test_identity_prediction_parameter_whitelist_and_shared_initialization():
    F,mask,m=inputs()
    models={arm:api().JointPilotModel(frontend(),arm,seed=42) for arm in ('M','A','B','AB')}
    for arm,model in models.items():
        assert torch.equal(model.shared_adapter(F),F)
        model.train()
        assert not model.former.training and not model.old_projection.training
        result=model.forward_prefix(F,mask,m)
        if arm in ('M','A','AB'):
            assert torch.equal(result['mean'],m[:,None,:].expand(-1,3,-1))
        else: assert 'mean' not in result
        if arm in ('B','AB'):
            teacher=model.teacher_prefix(F,mask)
            torch.testing.assert_close(result['logits'],teacher['logits'],rtol=0,atol=0)
        else: assert 'logits' not in result
        assert not any(p.requires_grad for p in model.former.parameters())
        assert not any(p.requires_grad for p in model.old_projection.parameters())
    for name,p in models['A'].dynamics.named_parameters():
        assert torch.equal(p,dict(models['AB'].dynamics.named_parameters())[name])
    for name,p in models['B'].reader.named_parameters():
        assert torch.equal(p,dict(models['AB'].reader.named_parameters())[name])


def test_padding_checkpoint_and_no_future_input_leak():
    F,mask,m=inputs(20);mask[1,11:]=False
    model=api().JointPilotModel(frontend(),'AB',seed=42)
    model.checkpoint_updates=False
    with torch.no_grad():model.forecast[-1].weight.normal_(std=.01)
    before=model.forward_prefix(F,mask,m)
    changed=F.clone();changed[1,11:]=999
    after=model.forward_prefix(changed,mask,m)
    torch.testing.assert_close(before['logits'],after['logits'],rtol=0,atol=0)
    torch.testing.assert_close(before['mean'],after['mean'],rtol=0,atol=0)
    short=model.forward_prefix(F[1:2,:11],mask[1:2,:11],m[1:2])
    torch.testing.assert_close(before['logits'][1],short['logits'][0],atol=1e-6,rtol=1e-6)
    plain=copy.deepcopy(model);model.checkpoint_updates=True
    model.zero_grad();plain.zero_grad()
    model.forward_prefix(F,mask,m)['logits'].square().sum().backward()
    plain.forward_prefix(F,mask,m)['logits'].square().sum().backward()
    for p,q in zip(model.shared_adapter.parameters(),plain.shared_adapter.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-5)


def test_both_losses_reach_shared_adapter_but_not_teacher():
    F,mask,m=inputs()
    for arm,key in (('A','mean'),('B','logits')):
        model=api().JointPilotModel(frontend(),arm,seed=42)
        optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            result=model.forward_prefix(F,mask,m)[key]
            loss=(result-1).square().mean()
            loss.backward()
            if _==1:
                assert sum(float(p.grad.abs().sum()) for p in model.shared_adapter.parameters() if p.grad is not None)>0
            optimizer.step()
        assert all(p.grad is None for p in model.former.parameters())
        assert all(p.grad is None for p in model.old_projection.parameters())
        assert not model.teacher_prefix(F,mask)['logits'].requires_grad


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_autocast_burnin_preserves_trainable_adapter_weight_derivatives(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable on local test host')
    F,mask,m=inputs(32)
    F,mask,m=F.to(device),mask.to(device),m.to(device)
    model=api().JointPilotModel(frontend(),'A').to(device)
    with torch.no_grad():model.forecast[-1].weight.normal_(std=.01)
    with torch.autocast(device,dtype=torch.bfloat16 if device=='cpu' else torch.float16):
        result=model.forward_prefix(F,mask,m)
        loss=result['mean'].square().mean()
    loss.backward()
    assert model.shared_adapter.up.weight.grad is not None
    assert model.shared_adapter.up.weight.grad.abs().sum()>0
