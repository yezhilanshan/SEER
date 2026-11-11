import copy
import torch
from test_harrad_event import tiny_model


def make_joint(arm='AB',seed=42):
    from continuous_mmwave.harrad.joint_model import JointModel
    return JointModel(copy.deepcopy(tiny_model()),arm,seed,dynamics_dim=32)


def test_common_initialization_zero_forecast_and_semantic_warmup_equivalence():
    a=make_joint('A'); ab=make_joint('AB'); b=make_joint('B')
    for key,p in a.tokens.state_dict().items():
        assert torch.equal(p,ab.tokens.state_dict()[key]) and torch.equal(p,b.tokens.state_dict()[key])
    for key,p in a.dynamics.state_dict().items(): assert torch.equal(p,ab.dynamics.state_dict()[key])
    x=torch.randn(2,7,2,4,64); mask=torch.ones(2,7,dtype=torch.bool); current=torch.randn(2,64)
    ab.eval(); b.eval(); source=tiny_model().eval()
    with torch.no_grad():
        output=ab(x,mask,current)
        original=source(x.flatten(1,2),mask.repeat_interleave(2,1))
    torch.testing.assert_close(output['mean'],current[:,None].expand(-1,3,-1),atol=0,rtol=0)
    torch.testing.assert_close(output['logits'],original['final_logits'])


def test_prediction_and_semantic_losses_reach_shared_adapter_and_qwen_stays_frozen():
    model=make_joint().train()
    with torch.no_grad(): model.forecast[-1].weight.normal_(std=.01)
    x=torch.randn(2,8,2,4,64); mask=torch.ones(2,8,dtype=torch.bool); m=torch.randn(2,64)
    prediction=model(x,mask,m)
    prediction['mean'].square().mean().backward()
    assert model.tokens.project.weight.grad.abs().max()>0
    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x,mask,m)['logits'],torch.tensor([2,5])).backward()
    assert model.tokens.project.weight.grad.abs().max()>0
    assert model.former.cross_attention[0].alpha.grad is not None
    assert all(not p.requires_grad and p.grad is None for p in model.former.adapter.parameters())


def test_padding_burnin_and_resettable_local_dynamics():
    model=make_joint().eval()
    with torch.no_grad(): model.forecast[-1].weight.normal_(std=.01)
    x=torch.randn(2,20,2,4,64); mask=torch.ones(2,20,dtype=torch.bool); mask[0,4:]=False
    current=torch.randn(2,64)
    with torch.no_grad():
        a=model(x,mask,current)
        changed=x.clone(); changed[0,4:]*=100
        b=model(changed,mask,current)
        short=model(x[:1,:4],mask[:1,:4],current[:1])
        reset=model(x[1:,10:],mask[1:,10:],current[1:])
    torch.testing.assert_close(a['mean'],b['mean'])
    torch.testing.assert_close(a['logits'][:1],short['logits'])
    assert not torch.allclose(a['mean'][1:],reset['mean'])
    model.train(); x.requires_grad_()
    model(x[1:],mask[1:],current[1:])['logits'].sum().backward()
    assert x.grad[1,:4].abs().max()==0 and x.grad[1,4:].abs().max()>0
