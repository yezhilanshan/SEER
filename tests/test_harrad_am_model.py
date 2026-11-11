import copy
import math
import pytest
import torch
from test_harrad_event import tiny_model


def make_am(arm='B_both'):
    from continuous_mmwave.harrad.am_model import AlignmentModel
    return AlignmentModel(copy.deepcopy(tiny_model()),arm,42)


def test_modulation_starts_identity_and_can_use_past_event_information():
    from continuous_mmwave.harrad.am_modules import EventModulation
    module=EventModulation(16,16,8)
    z=torch.randn(2,4,16); previous=torch.randn(2,2,16)
    torch.testing.assert_close(module(z,previous),z,atol=0,rtol=0)
    module(z,previous).square().mean().backward()
    assert module.affine[-1].weight.grad.abs().max()>0
    with torch.no_grad(): module.affine[-1].weight.normal_(std=.1)
    assert not torch.allclose(module(z,previous),module(z,-previous))


def test_contrastive_excludes_same_class_and_masks_unknown_queries():
    from continuous_mmwave.harrad.am_modules import AnchoredContrastive
    loss=AnchoredContrastive(temperature=1.)
    query=torch.tensor([[1.,0.],[0.,1.]],requires_grad=True)
    positive=torch.tensor([[1.,0.],[0.,1.]])
    bank=torch.tensor([[1.,0.],[0.,1.]])
    values=loss(query,positive,torch.tensor([0,-100]),bank,torch.tensor([0,1]))
    torch.testing.assert_close(values[0],torch.tensor(math.log1p(math.exp(-1))/math.log(2)))
    assert values[1]==0
    values.sum().backward(); assert not query.grad[1].any()
    with pytest.raises(ValueError,match='negative'):
        loss(query[:1],positive[:1],torch.tensor([0]),bank[:1],torch.tensor([0]))


def test_all_arms_start_as_original_b_and_reference_is_fixed():
    from continuous_mmwave.harrad.joint_model import JointModel
    baseline=JointModel(copy.deepcopy(tiny_model()),'B').eval()
    x=torch.randn(2,7,2,4,64); mask=torch.ones(2,7,dtype=torch.bool); current=torch.randn(2,64)
    expected=baseline(x,mask,current)['logits']
    for arm in ('B0','B_align','B_mod','B_both'):
        model=make_am(arm).eval()
        out=model(x,mask,current)
        torch.testing.assert_close(out['logits'],expected,atol=0,rtol=0)
        assert out['query'].shape==(2,16)
        assert all(not p.requires_grad for p in model.reference.parameters())
        assert not any(k.startswith(('reference.','former.adapter.')) for k in model.learned_state())


def test_alignment_gradient_reaches_event_and_adapter_but_not_reference_or_qwen():
    from continuous_mmwave.harrad.am_modules import AnchoredContrastive
    model=make_am('B_align').train()
    x=torch.randn(2,8,2,4,64); mask=torch.ones(2,8,dtype=torch.bool)
    result=model(x,mask,torch.zeros(2,64)); bank=torch.randn(12,16)
    AnchoredContrastive()(result['query'],result['reference'],torch.tensor([0,1]),bank,torch.arange(12)).mean().backward()
    assert model.tokens.project.weight.grad.abs().max()>0
    assert model.former.cross_attention[0].alpha.grad is not None
    assert all(p.grad is None for p in model.reference.parameters())
    assert all(p.grad is None for p in model.former.adapter.parameters())


def test_causal_reference_padding_and_burnin_do_not_read_future_or_old_gradients():
    model=make_am().eval(); x=torch.randn(2,20,2,4,64)
    mask=torch.ones(2,20,dtype=torch.bool); mask[0,4:]=False
    current=torch.zeros(2,64)
    with torch.no_grad():
        a=model(x,mask,current); changed=x.clone(); changed[0,4:]*=-100
        b=model(changed,mask,current); short=model(x[:1,:4],mask[:1,:4],current[:1])
    for key in ('logits','query','reference'):
        torch.testing.assert_close(a[key],b[key],atol=0,rtol=0)
        torch.testing.assert_close(a[key][:1],short[key],atol=1e-6,rtol=1e-6)
    model.train(); x.requires_grad_()
    model(x[1:],mask[1:],current[1:])['logits'].sum().backward()
    assert not x.grad[1,:4].any() and x.grad[1,4:].abs().max()>0


def test_short_prefix_trains_base_tokens_instead_of_detaching_initial_state():
    model=make_am('B0').train()
    x=torch.randn(2,4,2,4,64); mask=torch.ones(2,4,dtype=torch.bool)
    model(x,mask,torch.zeros(2,64))['logits'].sum().backward()
    assert model.former.base_event_tokens.grad is not None
    assert model.former.base_event_tokens.grad.abs().max()>0


@pytest.mark.skipif(not torch.cuda.is_available(),reason='real half-state CUDA regression')
def test_modulation_accepts_real_half_event_state_on_cuda():
    from continuous_mmwave.harrad.am_modules import EventModulation
    module=EventModulation(256,896,64).cuda()
    z=torch.randn(2,4,256,device='cuda'); previous=torch.randn(2,8,896,device='cuda',dtype=torch.float16)
    value=module(z,previous)
    torch.testing.assert_close(value,z,atol=0,rtol=0)
    value.square().mean().backward()
    assert torch.isfinite(module.affine[-1].weight.grad).all()
