from types import SimpleNamespace
import torch
from test_harrad_event import tiny_model


def make_reset():
    from continuous_mmwave.harrad.joint_model import JointModel
    from continuous_mmwave.harrad.event_semantic import RadarTokenAdapter
    from continuous_mmwave.harrad.reset_model import ResetModel
    torch.manual_seed(32)
    a=JointModel(SimpleNamespace(tokens=RadarTokenAdapter(16)),'A',42,dynamics_dim=8)
    with torch.no_grad(): a.forecast[-1].weight.normal_(std=.03)
    return ResetModel(a,tiny_model())


def test_reset_preserves_global_but_changes_local_event_and_future():
    model=make_reset().eval(); state=model.initial(1,'cpu')
    za=torch.randn(1,4,16); zb=torch.randn(1,4,16); m=torch.randn(1,64)
    for _ in range(3): state,_=model.step(za,zb,m,state,forced=torch.zeros(1))
    keep,a=model.step(za,zb,m,state,forced=torch.zeros(1))
    born,b=model.step(za,zb,m,state,forced=torch.ones(1))
    torch.testing.assert_close(keep['G'],born['G'],atol=0,rtol=0)
    assert not torch.equal(keep['L'],born['L']) and not torch.equal(keep['E'],born['E'])
    assert (a['mean']-b['mean']).abs().max()>0
    assert keep['age'].item()>born['age'].item()


def test_future_and_semantic_reach_policy_with_frozen_sources():
    model=make_reset().train(); model.phase('policy'); state=model.initial(1,'cpu')
    for _ in range(4): state,out=model.step(torch.randn(1,4,16),torch.randn(1,4,16),torch.randn(1,64),state)
    params=[p for p in model.policy.parameters() if p.requires_grad]
    future=torch.autograd.grad(out['mean'].square().mean(),params,retain_graph=True,allow_unused=True)
    semantic=torch.autograd.grad(out['logits'].square().mean(),params,allow_unused=True)
    assert sum(float(v.abs().sum()) for v in future if v is not None)>0
    assert sum(float(v.abs().sum()) for v in semantic if v is not None)>0
    assert all(not p.requires_grad and p.grad is None for n,p in model.named_parameters() if not n.startswith('policy.'))


def test_chunk_detach_keeps_values_and_state_inputs_remain_causal():
    from continuous_mmwave.harrad.reset_model import detached
    model=make_reset().eval(); state=model.initial(1,'cpu'); seq=[]
    for _ in range(6): seq.append((torch.randn(1,4,16),torch.randn(1,4,16),torch.randn(1,64)))
    outputs=[]
    for inputs in seq: state,out=model.step(*inputs,state); outputs.append(out['q'].detach())
    again=model.initial(1,'cpu')
    for i,inputs in enumerate(seq):
        if i==3: again=detached(again)
        again,out=model.step(*inputs,again)
        torch.testing.assert_close(out['q'],outputs[i],atol=0,rtol=0)
    for k in state: torch.testing.assert_close(state[k],again[k],atol=0,rtol=0)
