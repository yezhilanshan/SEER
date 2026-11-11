import copy
import numpy as np
import torch
from test_harrad_reset_model import make_reset
from test_harrad_reset_data import meta


def records():
    from continuous_mmwave.harrad.reset_data import reset_targets
    rng=np.random.default_rng(44); rows=[]
    for role,i in [('train',1),('train',2),('validation',3)]:
        m=meta(); m.update(role=role,subject='subject1' if role=='train' else 'subject7',file=f'{role}_{i}')
        rows.append({**m,'reset':reset_targets(m),'cells':rng.normal(size=(24,2,4,64)).astype('float32'),
                     'motion':rng.normal(size=(24,64)).astype('float32')})
    return rows


def test_tiny_head_policy_pipeline_carries_stream_and_preserves_sources(tmp_path):
    from continuous_mmwave.harrad.reset_training import cache_tokens,fit_heads,fit_policy,source_digest,working,load_working
    from continuous_mmwave.harrad.reset_evaluation import rollout,evaluate_record
    model=make_reset(); rows=records(); stats={'mean':np.zeros(64),'std':np.ones(64),
        'persistence_mse':np.ones(3),'logvar':np.zeros((3,64))}
    cache_tokens(model,rows,stats,'cpu'); before=source_digest(model)
    heads=fit_heads(model,rows,[0],stats,2,4,tmp_path/'heads','cpu')
    assert heads['successful_steps']==2
    initial=working(model)
    done=fit_policy(model,rows,[0],stats,3,4,42,tmp_path/'policy',False,False,'cpu',checkpoints=(0,3))
    assert done['successful_steps']==3 and done['attempts']==3
    assert source_digest(model)==before
    assert done['last_positions']=={'0':12}
    model.eval(); actual=rollout(model,rows[2],device='cpu')
    assert len(actual['q'])==24 and evaluate_record(actual,rows[2],stats)['n_updates']==24
    load_working(model,initial); paired=working(model)
    for group in initial:
        for key in initial[group]: torch.testing.assert_close(initial[group][key],paired[group][key],atol=0,rtol=0)


def test_control_removes_only_future_policy_gradient():
    from continuous_mmwave.harrad.reset_training import policy_losses
    model=make_reset().train(); row=records()[0]
    row['normalized']=row['motion']; state=model.initial(1,'cpu'); outputs=[]
    for _ in range(4):
        state,out=model.step(torch.randn(1,4,16),torch.randn(1,4,16),torch.randn(1,64),state); outputs.append(out)
    stats={'persistence_mse':np.ones(3)}
    full=policy_losses(outputs,row,0,stats,False,True)
    control=policy_losses(outputs,row,0,stats,False,False)
    p=[v for v in model.policy.parameters() if v.requires_grad]
    g0=torch.autograd.grad(control['total'],p,retain_graph=True,allow_unused=True)
    g1=torch.autograd.grad(full['total'],p,retain_graph=True,allow_unused=True)
    gf=torch.autograd.grad(full['future'],p,allow_unused=True)
    assert any(v is not None and v.abs().sum()>0 for v in gf)
    for a,b,c in zip(g1,g0,gf):
        if a is not None: torch.testing.assert_close(a-(b if b is not None else 0),c if c is not None else torch.zeros_like(a),atol=2e-6,rtol=1e-4)


def test_teacher_unknown_only_chunks_advance_state_without_spending_learning_budget(tmp_path):
    from continuous_mmwave.harrad.reset_data import reset_targets
    from continuous_mmwave.harrad.reset_training import cache_tokens,fit_policy
    model=make_reset(); rows=records(); m={'frames':48,'instances':[{'start':20,'stop':36,'label':1}]}
    rows[0]['reset']=reset_targets(m)
    stats={'mean':np.zeros(64),'std':np.ones(64),'persistence_mse':np.ones(3)}
    cache_tokens(model,rows,stats,'cpu')
    done=fit_policy(model,rows,[0],stats,1,4,42,tmp_path/'teacher',True,False,'cpu',checkpoints=(0,1))
    assert done['successful_steps']==1 and done['attempts']==3
    assert done['skipped_unknown_chunks']==2 and done['last_positions']=={'0':12}
