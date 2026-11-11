from pathlib import Path
import sys,math
import numpy as np
import torch,pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'.audit-staging'))
import proposal_window_peak as m
import proposal_window_ablation as worker
from continuous_mmwave.v25.stage3 import V25Stage3
from continuous_mmwave.v25.physical_proposal import ProposalBatches
from continuous_mmwave.v25.checkpoint import file_sha

def test_window_plan_excludes_overlap_touch_tail_invalid_and_chunk_cuts():
    t=np.arange(61)*.18;v=np.ones(61,bool)
    plan=m.record_windows(t,v,np.array([2.7,4.86,8.1]))
    assert [r['boundary_index'] for r in plan['windows']]==[2]
    assert plan['excluded']['overlap']==2
    assert m.chunk_windows(plan,39,51)==[]
    assert len(m.chunk_windows(plan,38,53))==1
    v[45]=False
    assert m.record_windows(t,v,np.array([8.1]))['windows']==[]
    assert m.record_windows(t,np.ones(61,bool),np.array([10.8]))['excluded']['record_edge']==1
    tg=t.copy();tg[46:]+=.18
    assert m.record_windows(tg,np.ones(61,bool),np.array([8.1]))['excluded']['gap']==1

def test_raw_logits_loss_is_mean_over_boundaries_and_shift_invariant():
    z=torch.tensor([[0.,math.log(3.),math.log(2.),0.]],requires_grad=True)
    y=torch.tensor([[1.,1.,1.,1.]])
    one=m.peak_loss(z,y,[(0,2)])
    assert one['ce'].item()==pytest.approx(math.log(4.)-.5*math.log(3.))
    two=m.peak_loss(z,y,[(0,2),(2,4)])
    expected=.5*((math.log(4.)-.5*math.log(3.))+(math.log(3.)-.5*math.log(2.)))
    assert two['ce'].item()==pytest.approx(expected)
    assert m.peak_loss(z+100,y,[(0,2),(2,4)])['ce'].item()==pytest.approx(expected,abs=1e-6)
    assert two['kl'].item()==pytest.approx(expected-math.log(2.))
    assert two['count']==2

def test_empty_window_loss_is_zero_and_does_not_backpropagate_auxiliary():
    z=torch.tensor([[1.,2.,3.]],requires_grad=True)
    r=m.peak_loss(z,torch.ones_like(z),[])
    assert r['count']==0 and r['ce'].item()==r['kl'].item()==0
    r['ce'].backward();assert torch.equal(z.grad,torch.zeros_like(z))

def source_batch():
    torch.manual_seed(8)
    return dict(tokens=torch.randn(1,8,2,4),mask=torch.ones(1,8,2,dtype=torch.bool),
        target=torch.tensor([[0.,.1,.5,1.,.8,.2,.1,.01]]),valid=torch.tensor([[False,True,True,True,True,True,True,True]]),
        class_weights=(3.,.6),peak_windows=[(1,6)])

def test_window_only_adds_loss_preserves_forward_rng_state_and_frozen_heads():
    torch.manual_seed(42);a=V25Stage3(radar_dim=4,event_dim=6,semantic_dim=3)
    b=m.WindowStage('window-peak',radar_dim=4,event_dim=6,semantic_dim=3);b.load_state_dict(a.state_dict())
    src=source_batch();rng=torch.get_rng_state().clone();ra=a.physical_stream_step(src);end_rng=torch.get_rng_state().clone()
    torch.set_rng_state(rng);rb=b.physical_stream_step(src)
    assert torch.equal(end_rng,torch.get_rng_state()) and torch.equal(rb['wbce'],ra['loss'])
    assert rb['loss'].item()==pytest.approx(ra['loss'].item()+.1*(rb['local_peak']['ce'].item()+rb['radar_peak']['ce'].item()))
    for k in ('fast','mean','m2'):assert torch.equal(getattr(ra['next_state'],k),getattr(rb['next_state'],k))
    frozen={k:v.clone() for k,v in b.state_dict().items() if not k.startswith(('heads.local.','heads.radar.'))}
    optimizer=torch.optim.AdamW([p for p in b.parameters() if p.requires_grad],lr=.001)
    rb['loss'].backward();optimizer.step()
    assert len(optimizer.state)==20 and all(torch.equal(v,b.state_dict()[k]) for k,v in frozen.items())
    assert not any('fusion' in name or 'or.' in name for name,_ in b.named_parameters())

def test_weighted_control_delegates_exactly_and_prefix_is_causal():
    torch.manual_seed(42);a=V25Stage3(radar_dim=4,event_dim=6,semantic_dim=3)
    c=m.WindowStage('weighted',radar_dim=4,event_dim=6,semantic_dim=3);c.load_state_dict(a.state_dict())
    src=source_batch();rng=torch.get_rng_state().clone();ra=a.physical_stream_step(src)
    torch.set_rng_state(rng);rc=c.physical_stream_step(src);assert torch.equal(ra['loss'],rc['loss'])
    b=m.WindowStage('window-peak',radar_dim=4,event_dim=6,semantic_dim=3);b.load_state_dict(a.state_dict());b.eval()
    full=b.physical_stream_step(src)
    prefix={k:(v[:,:4] if isinstance(v,torch.Tensor) and v.ndim>=2 else v) for k,v in src.items()};prefix['peak_windows']=[]
    short=b.physical_stream_step(prefix)
    assert torch.equal(full['local_logits'][:,:4],short['local_logits'])
    assert torch.equal(full['radar_logits'][:,:4],short['radar_logits'])

def test_invalid_peak_windows_rejected_before_loss():
    z=torch.ones(1,4);y=torch.ones_like(z)
    for windows in [[(0,5)],[(2,2)],[(-1,2)]]:
        with pytest.raises(ValueError):m.peak_loss(z,y,windows)

@pytest.mark.parametrize('missing',[[9,10],[20,21]])
def test_window_edge_missing_samples_are_not_complete(missing):
    t=np.delete(np.arange(31)*.18,missing)
    plan=m.record_windows(t,np.ones(len(t),bool),np.array([2.7]))
    assert plan['windows']==[] and plan['excluded']['gap']==1

def test_provenance_distinguishes_inherited_evidence_from_experiment_updates():
    spec=dict(initial_checkpoint_sha256='init',original_manifest_sha256='original',encoder_index_sha256='index')
    source=worker.arm_source('weighted',spec,'spec','digest',dict(physical=0,semantic=0,evidence=500))
    assert source['evidence_optimizer_steps']==500
    assert source['semantic_optimizer_steps']==0
    assert source['evidence_updates_this_experiment']==source['semantic_updates_this_experiment']==0

def test_64_update_tbptt_and_carried_state_backwards_match_control_rng():
    torch.manual_seed(42);a=V25Stage3(radar_dim=4,event_dim=6,semantic_dim=3)
    c=m.WindowStage('weighted',radar_dim=4,event_dim=6,semantic_dim=3);c.load_state_dict(a.state_dict())
    b=m.WindowStage('window-peak',radar_dim=4,event_dim=6,semantic_dim=3);b.load_state_dict(a.state_dict())
    stages=[a,c,b];opts=[torch.optim.AdamW([p for p in s.parameters() if p.requires_grad],lr=.001) for s in stages]
    previous=[None]*3;tokens=torch.randn(1,128,2,4);mask=torch.ones(1,128,2,dtype=torch.bool)
    for step in range(2):
        src=dict(tokens=tokens[:,step*64:(step+1)*64],mask=mask[:,step*64:(step+1)*64],target=torch.zeros(1,64),valid=torch.ones(1,64,dtype=torch.bool),class_weights=(3.,.6),peak_windows=[])
        if step==0:src['valid'][0,0]=False
        rng=torch.get_rng_state().clone();results=[];ends=[]
        for i,(stage,opt) in enumerate(zip(stages,opts)):
            torch.set_rng_state(rng);batch=dict(src)
            if step:batch.update(initial_state=previous[i],previous_tokens=tokens[:,63],previous_mask=mask[:,63])
            opt.zero_grad();r=stage.physical_stream_step(batch);r['loss'].backward();opt.step();results.append(r);ends.append(torch.get_rng_state().clone());previous[i]=r['next_state']
            assert all(not getattr(previous[i],k).requires_grad and getattr(previous[i],k).grad_fn is None for k in ('fast','mean','m2'))
        assert all(torch.equal(ends[0],end) for end in ends[1:])
        assert all(torch.equal(results[0]['loss'],r['loss']) for r in results[1:])
        assert all(torch.equal(a.state_dict()[k],s.state_dict()[k]) for s in stages[1:] for k in a.state_dict())

def test_real_sampler_keeps_original_draws_and_averages_zero_window_chunks(tmp_path):
    index={'records':[]}
    for j in range(2):
        labels=torch.zeros(20,2,dtype=torch.long)
        if j==0:labels[10:]=1
        payload=dict(radar=torch.randn(20,2,4),radar_mask=torch.ones(20,2,dtype=torch.bool),labels=labels,
            timestamps=torch.arange(40,dtype=torch.float64).reshape(20,2)*.09)
        p=tmp_path/f'cache{j}.pt';torch.save(payload,p)
        index['records'].append(dict(uid=f'r{j}',role='fit',path=str(p),sha256=file_sha(p),metadata=dict(subject_id='s',sensor_id='r')))
    a=V25Stage3(radar_dim=4,event_dim=6,semantic_dim=3);control=m.WindowStage('weighted',radar_dim=4,event_dim=6,semantic_dim=3);control.load_state_dict(a.state_dict())
    original=ProposalBatches(index,device='cpu');paired=m.WindowBatches(index,device='cpu');rng=torch.get_rng_state().clone()
    la=original(a);after=torch.get_rng_state().clone();torch.set_rng_state(rng);lb=paired(control)
    assert torch.equal(la,lb) and torch.equal(after,torch.get_rng_state()) and original.last==paired.last
    b=m.WindowStage('window-peak',radar_dim=4,event_dim=6,semantic_dim=3);b.load_state_dict(a.state_dict())
    builder=m.WindowBatches(index,device='cpu');total=builder(b);aux=builder.last_aux
    assert len(aux)==16 and sum(x['eligible_windows']>0 for x in aux)==8
    expected=sum(x['wbce']+.1*(x['local_peak']['ce']+x['radar_peak']['ce']) for x in aux)/16
    assert total.item()==pytest.approx(expected,abs=1e-6)
