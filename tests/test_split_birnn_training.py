from __future__ import annotations
import importlib
from pathlib import Path
import numpy as np
import torch

torch.set_num_threads(1)


def pool_fixture(tmp_path):
    rows=[]
    for i in range(4):
        p=tmp_path/f'r{i}.npz'
        np.savez(p,md=np.random.default_rng(i).normal(size=(20,128)).astype('float32'),
                 labels=np.array([-100]*2+[i%3]*18,dtype=np.int64))
        rows.append({'path':str(p),'recording_id':f'r{i}','subject_id':f's{i//2}'})
    return rows


def test_cached_uniform_sampler_restores_sequence_and_actual_counters(tmp_path):
    t=importlib.import_module('continuous_mmwave.split_birnn.training')
    rows=pool_fixture(tmp_path); a=t.RecordingSampler(rows,seed=9,min_length=5,max_length=12)
    for _ in range(3): a.sample(2)
    state=a.state_dict(); want=a.sample(2)
    b=t.RecordingSampler(rows,seed=123,min_length=5,max_length=12);b.load_state_dict(state)
    got=b.sample(2)
    torch.testing.assert_close(want[0],got[0],rtol=0,atol=0)
    torch.testing.assert_close(want[1],got[1],rtol=0,atol=0)
    assert want[2]==got[2]
    assert a.state_dict()['record_counts']==b.state_dict()['record_counts']
    assert a.expected_distribution()['subject_probabilities']=={'s0':.5,'s1':.5}
    assert sum(a.state_dict()['class_frame_counts'])+a.state_dict()['ignored_frames']==a.state_dict()['sampled_frames']


def test_checkpoint_restores_adam_scheduler_rng_and_next_updates_exactly(tmp_path):
    t=importlib.import_module('continuous_mmwave.split_birnn.training')
    t.seed_everything(7)
    net,opt,sched=t.make_training_state('split','cpu')
    sampler=t.RecordingSampler(pool_fixture(tmp_path),seed=42,min_length=5,max_length=8)
    for _ in range(2): t.optimize_step(net,opt,sched,sampler.sample(2),device='cpu')
    path=tmp_path/'state.pt'
    t.save_checkpoint(path,net,opt,sched,sampler,step=2,extra={'protocol':'test'})
    batch=sampler.sample(2); a=t.optimize_step(net,opt,sched,batch,device='cpu')
    expected={k:v.clone() for k,v in net.state_dict().items()}
    torch_random=torch.rand(4); numpy_random=np.random.rand(4)
    net2,opt2,sched2=t.make_training_state('split','cpu')
    sampler2=t.RecordingSampler(sampler.rows,seed=3,min_length=5,max_length=8)
    restored=t.load_checkpoint(path,net2,opt2,sched2,sampler2)
    assert restored['step']==2 and restored['extra']['protocol']=='test'
    batch2=sampler2.sample(2)
    assert batch[2]==batch2[2]
    b=t.optimize_step(net2,opt2,sched2,batch2,device='cpu')
    assert a['loss']==b['loss']
    assert all(torch.equal(v,net2.state_dict()[k]) for k,v in expected.items())
    assert sched.state_dict()==sched2.state_dict()
    torch.testing.assert_close(torch.rand(4),torch_random,rtol=0,atol=0)
    np.testing.assert_array_equal(np.random.rand(4),numpy_random)


def test_bn_diagnostic_does_not_modify_real_model_and_training_updates_statistics(tmp_path):
    t=importlib.import_module('continuous_mmwave.split_birnn.training')
    net,opt,sched=t.make_training_state('split','cpu')
    sampler=t.RecordingSampler(pool_fixture(tmp_path),seed=1,min_length=8,max_length=8)
    batch=sampler.sample(2); before={k:v.clone() for k,v in net.state_dict().items()}
    check=t.bn_feature_diagnostic(net,batch[0])
    assert check['finite'] and check['train_eval_difference_l2']>0
    assert all(torch.equal(before[k],v) for k,v in net.state_dict().items())
    result=t.optimize_step(net,opt,sched,batch,device='cpu')
    assert result['forward_loss']>0 and result['corrected_loss']>0
    assert abs(result['loss']-result['forward_loss']-result['corrected_loss'])<1e-6
    for k,v in net.state_dict().items():
        if k.endswith('num_batches_tracked'): assert v.item()==before[k].item()+1
    assert all(v>0 for v in result['gradient_group_sums'].values())


def test_fast_window_evaluation_agrees_with_original_reference():
    t=importlib.import_module('continuous_mmwave.split_birnn.evaluation')
    m=importlib.import_module('continuous_mmwave.split_birnn.model')
    s=importlib.import_module('continuous_mmwave.split_birnn.streaming')
    x=torch.randn(1,19,128)
    for net,reference in ((m.SplitBiGRU().eval(),s.replay_split),(m.BiGRUBaseline().eval(),s.replay_baseline)):
        want=reference(net,x,window=5,hop=2)
        got=t.fast_replay(net,x,window=5,hop=2,window_batch=3)
        for key in ('first_logits','final_logits'):
            torch.testing.assert_close(got[key],want[key],equal_nan=True)
        np.testing.assert_array_equal(got['correction_counts'],want['correction_counts'])
