import numpy as np
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA backend check runs on training server')
@pytest.mark.parametrize('arm',['md','rd'])
def test_cuda_forward_backward_with_determinism(arm):
    from continuous_mmwave.harrad.training import seed_everything
    from continuous_mmwave.harrad.model import SegmentClassifier
    seed_everything(42,1)
    model=SegmentClassifier(arm).cuda()
    shape=(256,) if arm=='md' else (40,64)
    x=torch.randn((2,4)+shape,device='cuda'); mask=torch.ones(2,4,dtype=torch.bool,device='cuda')
    loss=torch.nn.functional.cross_entropy(model(x,mask),torch.tensor([0,1],device='cuda'))
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


@pytest.mark.parametrize('arm',['md','rd'])
def test_padding_cannot_affect_predictions_or_gradients(arm):
    from continuous_mmwave.harrad.model import SegmentClassifier
    torch.manual_seed(3); torch.set_num_threads(1)
    model=SegmentClassifier(arm,dropout=0).eval()
    shape=(256,) if arm=='md' else (40,64)
    x=torch.randn((2,5)+shape); mask=torch.tensor([[1,1,0,0,0],[1,1,1,1,1]],dtype=torch.bool)
    a=model(x,mask); changed=x.clone(); changed[0,2:]=1e5
    b=model(changed,mask)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(a[:1],model(x[:1,:2],mask[:1,:2]),atol=2e-6,rtol=1e-5)
    torch.nn.functional.cross_entropy(a,torch.tensor([0,1])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad.abs().max()>0 for p in model.parameters())


def test_metrics_use_fixed_classes_and_report_events_gestures_separately():
    from continuous_mmwave.harrad.training import classification_metrics
    y=np.arange(12); logits=np.eye(12,dtype=np.float32)*10
    m=classification_metrics(logits,y,['subject7']*12)
    assert m['macro_f1']==m['events_macro_f1']==m['gestures_macro_f1']==1
    assert len(m['per_class_recall'])==12 and m['subjects']['subject7']['accuracy']==1
    m=classification_metrics(np.zeros((2,12)),np.array([0,1]),['s','s'])
    assert m['macro_f1']==pytest.approx((2/3)/12)


def test_minimal_training_checkpoint_replay(tmp_path):
    from continuous_mmwave.harrad.training import train_run
    rows=[]
    for role in ('train','validation'):
        for label in range(12):
            p=tmp_path/f'{role}{label}.npy'
            np.save(p,np.random.default_rng(label).normal(size=(4,256)).astype(np.float32))
            rows.append({'role':role,'uid':f'{role}{label}','subject':'s1' if role=='train' else 's2','start':0,'stop':4,'label':label,'instance_id':label,'feature_path':str(p)})
    cfg={'epochs':1,'learning_rate':.001,'weight_decay':.01,'grad_clip':1.,'batch_size':{'md':12,'rd':2},'evaluation_batch_size':12,'device':'cpu','dropout':0.,'threads':1}
    result=train_run('md',42,rows,{'mean':np.zeros(256),'scale':np.ones(256)},cfg,tmp_path/'run')
    assert result['epochs_completed']==1 and result['successful_steps']==1
    assert result['replay_max_abs']==0 and result['train_exposures']==12
    assert (tmp_path/'run'/'best.pt').exists() and (tmp_path/'run'/'validation-best.npz').exists()
