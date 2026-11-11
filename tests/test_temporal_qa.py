from collections import Counter

import pytest
import torch

from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.models.temporal_qa import TemporalQAFusion
from continuous_mmwave.parrad_temporal_qa import (
    temporal_batches, temporal_context, temporal_loss, temporal_query, text_oracle_query,
)
from continuous_mmwave.parrad_event_boundary_tasks import build_boundary_tasks
from continuous_mmwave.models.event_former import TinyLLMAdapter
from tests.test_short_context_qa import events
from tests.test_stage4_qa import Tokenizer


def make_model():
    torch.manual_seed(42)
    return TemporalQAFusion(EventQAAdapter(16,8),bottleneck=8,encoding_dim=8)


def test_initial_parity_equal_arm_parameters_and_single_bypass_after_updates():
    model=make_model(); x=torch.randn(1,32,16)
    times=torch.tensor([[0.,1.,2.,3.]],dtype=torch.float64)
    ends=times+.9
    old=model.base(x).detach()
    for arm in ('none','order','time'):
        assert torch.equal(model(x,times,ends,arm=arm),old)
    with torch.no_grad(): model.output.weight.normal_(); model.output.bias.normal_()
    for arm in ('none','order','time'):
        assert torch.equal(model(x[:,:8],times[:,:1],ends[:,:1],arm=arm),model.base(x[:,:8]))
    assert all(not p.requires_grad for p in model.base.parameters())
    model.train()
    assert not model.base.training


def test_encoding_is_group_relative_event_shared_and_keeps_rank_separate_from_time():
    model=make_model()
    starts=torch.tensor([[1000.,1005.]],dtype=torch.float64); ends=starts+torch.tensor([[2.,7.]])
    none=model.features(starts,ends,arm='none')
    order=model.features(starts,ends,arm='order')
    timed=model.features(starts,ends,arm='time')
    assert none.shape==(1,16,24) and torch.count_nonzero(none)==0
    assert torch.equal(order[:,:,:8],timed[:,:,:8])
    assert torch.count_nonzero(order[:,:,8:])==0
    assert torch.equal(timed,model.features(starts+10000,ends+10000,arm='time'))
    assert torch.equal(timed[:,0],timed[:,7]) and torch.equal(timed[:,8],timed[:,15])
    assert not torch.equal(timed[:,0,:8],timed[:,8,:8])
    stretched=model.features(starts,ends+3,arm='time')
    assert torch.equal(timed[:,:,:16],stretched[:,:,:16])
    assert not torch.equal(timed[:,:,16:],stretched[:,:,16:])


@pytest.mark.parametrize('starts,ends',[
    ([[1.,0.]],[[1.9,.9]]), ([[0.,1.]],[[2.,1.9]]),
    ([[0.,1.]],[[0.,1.9]]), ([[0.,float('nan')]],[[.9,1.9]]),
])
def test_invalid_time_metadata_rejected(starts,ends):
    model=make_model()
    with pytest.raises(ValueError):
        model(torch.randn(1,16,16),torch.tensor(starts),torch.tensor(ends),arm='time')


def test_training_fusion_only_and_no_source_mutation_with_exact_budget():
    rows=events(); taskset=build_boundary_tasks(rows,42)
    batches=temporal_batches(taskset,42)
    assert len(batches)==500 and batches==temporal_batches(taskset,42)
    assert Counter(b[0]['size'] for b in batches)=={2:250,4:250}
    roles=Counter((b[2]['size'],b[2]['kind'],b[2].get('position')) for b in batches)
    assert roles=={(2,'position',1):125,(2,'position',2):125,(4,'sequence',None):125,
                  (4,'position',1):32,(4,'position',2):31,(4,'position',3):31,(4,'position',4):31}
    assert all(b[0]['answer']=='Yes' and b[1]['answer']=='No' and all(t['cohort']=='fit' for t in b) for b in batches)
    by_id={e['uid']:e for e in rows}; snapshots={u:e['tokens'].clone() for u,e in by_id.items()}
    model=make_model(); teacher=TinyLLMAdapter(128,16,2,4)
    batch=next(b for b in batches if b[0]['size']==4)
    context=temporal_context(model,batch[0],by_id,'cpu','time')
    assert context.shape==(1,32,16)
    # Tiny tokenizer is character based: only use short questions for gradient probe.
    probe=[dict(t,query='q') for t in batch]
    loss=temporal_loss(model,teacher,Tokenizer(),probe,by_id,'cpu','time'); loss.backward()
    assert model.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in model.base.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(e['tokens'],snapshots[u]) for u,e in by_id.items())
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001)
    optimizer.step(); optimizer.zero_grad()
    temporal_loss(model,teacher,Tokenizer(),probe,by_id,'cpu','time').backward()
    assert model.input.weight.grad.abs().sum()>0


def test_task_rephrasing_no_label_leak_and_oracle_keeps_repeated_actions():
    rows=events(); rows[0]['label']=rows[1]['label']=1
    taskset=build_boundary_tasks(rows,42); by_id={e['uid']:e for e in rows}
    sequence=next(t for t in taskset['tasks'] if t['kind']=='sequence' and t['source_ids'][0]==rows[0]['uid'])
    revised=temporal_query(sequence)
    assert 'exactly 4' in revised and 'repeated' in revised and 'Allowed activity' not in revised
    oracle=text_oracle_query(sequence,by_id)
    assert oracle.count('getting into bed')==2 and '1: getting into bed\n2: getting into bed' in oracle
    single=next(t for t in taskset['tasks'] if t['size']==1)
    assert temporal_query(single)==single['query']
    yes,no=[taskset['by_uid'][u] for u in next(p for p in taskset['pairs'] if p['size']==4)['task_ids']]
    assert temporal_query(yes)==temporal_query(no)


def test_oracle_gate_requires_actual_sequence_and_each_position_and_both_polarities():
    import continuous_mmwave.parrad_temporal_qa as runner
    assert hasattr(runner,'oracle_gate'), 'Missing pure-text gate before training'
    metric={}
    for n in ('2','4'):
        metric[n]=dict(presence_classes={'Yes':dict(accuracy=1),'No':dict(accuracy=1)},
                       positions={str(i):dict(accuracy=1) for i in range(1,int(n)+1)})
    metric['4']['sequence']=dict(accuracy=1)
    assert all(runner.oracle_gate(metric).values())
    metric['4']['sequence']['accuracy']=0
    metric['2']['presence_classes']['No']['accuracy']=.5
    metric['4']['positions']['3']['accuracy']=.9
    gate=runner.oracle_gate(metric)
    assert not gate['sequence'] and not gate['2_No'] and not gate['4_position_3']


def test_text_oracle_presence_has_one_unambiguous_answer_contract():
    rows=events(); taskset=build_boundary_tasks(rows,42)
    task=next(t for t in taskset['tasks'] if t['kind']=='presence' and t['size']==4)
    query=text_oracle_query(task,{e['uid']:e for e in rows})
    assert 'Answer only Yes or No.' in query
    assert 'Answer using only the activity names' not in query
    assert 'supplied memory' not in query
    assert 'evidence above' in query


def test_failed_text_preflight_persists_zero_updates_and_never_starts_optimizer(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import continuous_mmwave.parrad_temporal_qa as runner
    rows=events(); base=EventQAAdapter(16,8)
    initial=dict(model_settings=base.settings,model_state=base.state_dict())
    teacher=TinyLLMAdapter(128,16,2,4)
    runtime=SimpleNamespace(controller=SimpleNamespace(event_former=SimpleNamespace(adapter=teacher)))
    monkeypatch.setattr(runner,'load_boundary_sources',lambda settings,device:(runtime,Tokenizer(),initial,dict(events=rows),{}, {},'digest'))
    metric={}
    for n in ('2','4'):
        metric[n]=dict(presence_classes={'Yes':dict(accuracy=1),'No':dict(accuracy=1)},
                       positions={str(i):dict(accuracy=1) for i in range(1,int(n)+1)})
    metric['4']['sequence']=dict(accuracy=0)
    def evaluate(*args,**kwargs):
        assert args[-1]=='text'
        return dict(rows=[],metrics=metric)
    monkeypatch.setattr(runner,'evaluate',evaluate)
    monkeypatch.setattr(runner,'run_arm',lambda *args:pytest.fail('training after failed text gate'))
    config=tmp_path/'config.yaml'; config.write_text('test')
    output=tmp_path/'output'
    settings=dict(steps_per_arm=500,seed=42,output_dir=str(output),fusion=dict(bottleneck=8,encoding_dim=8))
    runner.run(settings,config)
    import json
    status=json.loads((output/'status.json').read_text())
    assert status['status']=='blocked' and status['actual_updates']==0
    assert not any((output/arm).exists() for arm in ('none','order','time'))


def test_export_settings_reconstruct_same_predictions_and_frozen_base():
    model=make_model()
    with torch.no_grad(): model.output.weight.normal_(std=.02)
    restored=TemporalQAFusion(EventQAAdapter(**model.base.settings),**model.settings)
    restored.load_state_dict(model.state_dict())
    raw=torch.randn(1,16,16); starts=torch.tensor([[0.,2.]]); ends=starts+1
    for arm in ('none','order','time'):
        assert torch.equal(restored(raw,starts,ends,arm=arm),model(raw,starts,ends,arm=arm))
    assert all(not p.requires_grad for p in restored.base.parameters())
