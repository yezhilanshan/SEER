import numpy as np
import torch
from test_harrad_joint_data import example
from test_harrad_joint_model import make_joint


def test_joint_losses_normalize_by_record_and_exclude_unknown_semantics():
    from continuous_mmwave.harrad.joint_training import joint_losses
    stats={'persistence_mse':np.ones(3,dtype='float32')}
    b={'future':torch.ones(3,3,64),'record_ids':torch.tensor([0,0,1]),'labels':torch.tensor([1,-100,2])}
    mean=torch.zeros_like(b['future'],requires_grad=True)
    logits=torch.zeros(3,12,requires_grad=True)
    loss=joint_losses({'mean':mean,'logits':logits},b,stats)
    torch.testing.assert_close(loss['motion'],torch.tensor(1.))
    torch.testing.assert_close(loss['semantic'],torch.tensor(1.))
    loss['total'].backward()
    assert not logits.grad[1].any()


def test_tiny_fixed_budget_saves_forecasts_logits_and_replays(tmp_path):
    from continuous_mmwave.harrad.joint_data import fit_stats, fixed_design
    from continuous_mmwave.harrad.joint_training import fit_job
    records=[example(),example('validation','subject7')]
    stats=fit_stats(records); design=fixed_design(records,[42],3,4)
    cfg={'steps':3,'learning_rate':.0001,'weight_decay':.01,'amp':False,
         'max_batch_attempts':3,'evaluation_batch_size':2,'checkpoints':[0,3]}
    result=fit_job(make_joint(),records,stats,design['schedule_42'],design['validation'],cfg,tmp_path/'job','cpu')
    assert result['successful_steps']==3 and result['attempts']==3
    assert result['frozen_unchanged'] and result['replay_max_abs']==0
    assert 'motion' in result['metrics'] and 'semantic' in result['metrics']
    with np.load(tmp_path/'job/validation-final.npz',allow_pickle=False) as a:
        assert a['mean'].shape[1:]==(3,64) and a['logits'].shape[1]==12
