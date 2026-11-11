import importlib
from collections import Counter

import pytest
import torch


def module():
    name='continuous_mmwave.parrad_radar_alignment'
    assert importlib.util.find_spec(name), 'missing single-event semantic alignment'
    return importlib.import_module(name)


def events():
    return [dict(uid=f'{cohort}:{label}:{i}',label=label,cohort=cohort)
            for cohort in ('fit','dev') for label in range(14) for i in range(2)]


def test_sampler_balances_all_classes_and_never_uses_dev_evidence():
    m=module(); tasks=m.build_tasks(events(),42); batches=m.training_batches(tasks,42)
    assert len(batches)==500 and batches==m.training_batches(tasks,42)
    counts=Counter()
    for batch in batches:
        yes,no,identify=batch
        assert yes['query']==no['query'] and [yes['answer'],no['answer']]==['Yes','No']
        assert identify['event_uid']==yes['event_uid'] and identify['kind']=='identify'
        assert all(t['cohort']=='fit' and t['event_uid'].startswith('fit:') for t in batch)
        counts[yes['query_label']]+=1
    assert set(counts)==set(range(14)) and max(counts.values())-min(counts.values())<=1


def test_teacher_evidence_uses_actual_event_while_radar_query_has_no_label_evidence():
    m=module(); tasks=m.build_tasks(events(),42); by_event={e['uid']:e for e in events()}
    no=next(t for t in tasks if t['kind']=='presence' and t['answer']=='No')
    from continuous_mmwave.parrad_stage4_qa import ACTIVITIES
    assert by_event[no['event_uid']]['label']!=no['query_label']
    assert 'Observed activity: '+ACTIVITIES[no['label']] in m.teacher_query(no,by_event)
    assert 'Observed activity:' not in no['query']


def test_answer_positions_align_despite_different_context_and_prompt_lengths():
    m=module(); ids=torch.tensor([[1,2,3,4,5]]); mask=torch.tensor([[0,0,0,1,1]],dtype=torch.bool)
    raw=torch.randn(1,8+5,6,requires_grad=True)
    student,labels=m.answer_positions(raw,ids,torch.ones_like(ids),mask,8)
    teacher_ids=torch.tensor([[1,1,1,1,1,1,4,5]])
    teacher_mask=torch.tensor([[0,0,0,0,0,0,1,1]],dtype=torch.bool)
    teacher_raw=torch.randn(1,8,6,requires_grad=True)
    teacher,teacher_labels=m.answer_positions(teacher_raw,teacher_ids,torch.ones_like(teacher_ids),teacher_mask,0)
    assert torch.equal(student,raw[0,[10,11]]) and torch.equal(labels,torch.tensor([4,5]))
    assert torch.equal(teacher,teacher_raw[0,[5,6]]) and torch.equal(labels,teacher_labels)
    result=m.answer_objective(student,labels,teacher,teacher_labels,weight=.5,temperature=2.)
    result['loss'].backward()
    assert raw.grad.abs().sum()>0 and teacher_raw.grad is None
    assert raw.grad[0,:10].abs().sum()==0 and raw.grad[0,12:].abs().sum()==0


def test_distillation_rejects_different_teacher_target_and_zero_when_equal():
    m=module(); logits=torch.tensor([[2.,-1.,0.]],requires_grad=True); labels=torch.tensor([0])
    result=m.answer_objective(logits,labels,logits.clone(),labels,weight=.5,temperature=2.)
    assert abs(float(result['kl'].detach()))<1e-6
    with pytest.raises(ValueError,match='target'):
        m.answer_objective(logits,labels,logits,torch.tensor([1]),weight=.5,temperature=2.)


def test_new_reader_evaluation_never_dispatches_to_original_single_reader():
    m=module()
    class Reader(torch.nn.Module):
        compute_dtype=torch.float32
        def __init__(self): super().__init__(); self.bias=torch.nn.Parameter(torch.zeros(()),requires_grad=False)
        def caption_logits(self,context,ids,attention):
            assert context.shape==(1,8,4)
            return context.sum()*torch.ones(1,8+ids.shape[1],6)+self.bias
    from continuous_mmwave.models.qa_adapter import EventQAAdapter
    adapter=EventQAAdapter(dim=4,bottleneck=2); reader=Reader()
    ids=torch.tensor([[1,2,3]]); mask=torch.tensor([[0,0,1]],dtype=torch.bool)
    logits,labels=m.radar_answer_positions(adapter,reader,torch.randn(1,8,4),ids,torch.ones_like(ids),mask)
    logits.sum().backward()
    assert adapter.output.weight.grad is not None and reader.bias.grad is None


def test_development_gate_requires_every_locked_metric():
    m=module(); metrics=dict(identify_macro_recall=.85,presence_balanced_accuracy=.9,pair_both_correct=.8)
    assert all(m.development_gates(metrics).values())
    for key in metrics:
        bad=dict(metrics);bad[key]-=.01
        assert not all(m.development_gates(bad).values())


@pytest.mark.parametrize('mutation',['missing','reader','source','evidence'])
def test_saved_predictions_reject_missing_old_reader_and_wrong_actual_evidence(mutation):
    m=module(); taskset=m.build_tasks(events(),42); by_event={e['uid']:e for e in events()}
    tasks=[t for t in taskset if t['cohort']=='dev' and t['kind']=='presence']
    lookup={t['uid']:t for t in tasks}; rows=[]
    for t in tasks:
        other=lookup[t['pair_id']+(':No' if t['answer']=='Yes' else ':Yes')]
        rows.append(dict(t,reference=t['answer'],generated=t['answer'],correct=True,reader='text-qa-500',
                         actual_event_uid=other['event_uid'],context_tokens=8,evidence_reference=other['answer']))
    result=dict(rows=rows,mode='swapped')
    m.validate_predictions(result,tasks,by_event)
    if mutation=='missing': rows.pop()
    elif mutation=='reader': rows[0]['reader']='original'
    elif mutation=='source': rows[0]['actual_event_uid']=tasks[0]['event_uid']
    else: rows[0]['evidence_reference']=tasks[0]['answer']
    with pytest.raises(ValueError): m.validate_predictions(result,tasks,by_event)


def test_generation_evaluation_directly_calls_new_reader_and_preserves_exact_task_set():
    from unittest.mock import patch
    name='continuous_mmwave.parrad_radar_alignment_train'
    assert importlib.util.find_spec(name), 'missing native single-event runner'
    runner=importlib.import_module(name)
    task=module().build_tasks(events(),42)[-1];event=dict(label=task['label'],tokens=torch.ones(8,4))
    from continuous_mmwave.models.qa_adapter import EventQAAdapter
    adapter=EventQAAdapter(dim=4,bottleneck=2);reader=object()
    with patch.object(runner,'generate_from_memory',return_value=task['answer']) as generation:
        result=runner.evaluate(adapter,reader,None,[task],{task['event_uid']:event},torch.device('cpu'),'raw')
    assert generation.call_args.args[0] is reader
    assert len(result['rows'])==1 and result['rows'][0]['reader']=='text-qa-500'


def test_native_loss_backward_reaches_connector_but_not_teacher():
    transformers=pytest.importorskip('transformers');pytest.importorskip('peft')
    runner=importlib.import_module('continuous_mmwave.parrad_radar_alignment_train')
    from continuous_mmwave.parrad_qa_text_decoder import create_reader
    from continuous_mmwave.models.qa_adapter import EventQAAdapter
    config=transformers.Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=64,num_hidden_layers=2,
        num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=256,attention_dropout=0.)
    config._attn_implementation='eager'
    reader=create_reader(transformers.Qwen2ForCausalLM(config),rank=2).eval().requires_grad_(False)
    class Tokenizer:
        eos_token_id=2
        def apply_chat_template(self,messages,**kwargs): return messages[0]['content']+' assistant:'
        def __call__(self,text,**kwargs): return {'input_ids':torch.tensor([[3]*len(text.split())])}
        def encode(self,text,**kwargs): return [4,5]
    tasks=module().build_tasks(events(),42);batch=module().training_batches(tasks,42)[0]
    by_event={e['uid']:dict(e,tokens=torch.randn(8,32)) for e in events()}
    before={k:v.clone() for k,v in reader.state_dict().items()}
    model=EventQAAdapter(dim=32,bottleneck=8);optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    scaler=torch.amp.GradScaler('cpu',enabled=False)
    values=runner.backward_batch(model,reader,Tokenizer(),batch,by_event,torch.device('cpu'),scaler,.5,2.)
    assert values['ce']>0 and values['loss']>=values['ce']-1e-6
    assert model.output.weight.grad.abs().sum()>0 and all(p.grad is None for p in reader.parameters())
    optimizer.step()
    assert len(optimizer.state)==6 and {int(v['step']) for v in optimizer.state.values()}=={1}
    assert all(torch.equal(v,before[k]) for k,v in reader.state_dict().items())
