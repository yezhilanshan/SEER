import torch
from collections import Counter
from continuous_mmwave.parrad_short_context_tasks import build_short_tasks, short_training_batches, score_short
from continuous_mmwave.parrad_short_context_qa import short_context, short_loss, build_l1_match
from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.models.memory import QueryCompressor
from continuous_mmwave.models.event_former import TinyLLMAdapter
from tests.test_stage4_qa import Tokenizer
from continuous_mmwave.parrad_stage4_qa import ACTIVITIES


def events():
    rows=[]
    for cohort in ('fit','dev'):
        for recording in ('a','b'):
            for i in range(17):
                rows.append(dict(uid=f'{cohort}-{recording}-{i}',cohort=cohort,recording=cohort+recording,
                    start=float(i),end=i+.9,label=i%14,tokens=torch.randn(8,16)))
    return rows


def test_grouping_keeps_recording_chronology_tails_and_opposite_pairs():
    rows=events(); taskset=build_short_tasks(rows,42); by_id={e['uid']:e for e in rows}
    for task in taskset['tasks']:
        sources=[by_id[uid] for uid in task['source_ids']]
        assert len(sources)==task['size'] and len({s['recording'] for s in sources})==1
        assert [s['start'] for s in sources]==sorted(s['start'] for s in sources)
        assert all(s['cohort']==task['cohort'] for s in sources)
    for pair in taskset['pairs']:
        yes,no=[taskset['by_uid'][uid] for uid in pair['task_ids']]
        assert yes['query']==no['query'] and yes['answer']=='Yes' and no['answer']=='No'
        assert pair['activity'] in [by_id[u]['label'] for u in yes['source_ids']]
        assert pair['activity'] not in [by_id[u]['label'] for u in no['source_ids']]
    assert all(len(row['source_ids'])==1 for row in taskset['tails'])
    assert all(all(name in t['query'] for name in ACTIVITIES)
               for t in taskset['tasks'] if t['kind'] in ('identify','first','last','sequence'))


def test_sampler_exact_budget_and_separate_type_clocks():
    taskset=build_short_tasks(events(),42); batches=short_training_batches(taskset,42)
    assert batches==short_training_batches(taskset,42)
    assert [sum(batch[0]['size']==size for batch in batches) for size in (1,2,4)]==[250,125,125]
    assert all(batch[0]['answer']=='Yes' and batch[1]['answer']=='No' for batch in batches)
    assert all(task['cohort']=='fit' for batch in batches for task in batch)
    assert {batch[2]['kind'] for batch in batches if batch[0]['size']==2}=={'first','last'}
    assert {batch[2]['kind'] for batch in batches if batch[0]['size']==4}=={'first','last','sequence'}
    assert Counter(batch[2]['kind'] for batch in batches if batch[0]['size']==2)=={'first':63,'last':62}
    assert Counter(batch[2]['kind'] for batch in batches if batch[0]['size']==4)=={'first':42,'last':42,'sequence':41}


def test_real_short_loss_learns_only_qa_and_l1_control_uses_same_sources():
    torch.manual_seed(5); rows=events(); tasks=build_short_tasks(rows,42); by_id={e['uid']:e for e in rows}
    model=EventQAAdapter(16,8); teacher=TinyLLMAdapter(128,16,2,4)
    batch=next(b for b in short_training_batches(tasks,42) if b[0]['size']==4)
    loss=short_loss(model,teacher,Tokenizer(),batch,by_id,torch.device('cpu')); loss.backward()
    assert model.output.weight.grad.abs().sum()>0 and all(p.grad is None for p in teacher.parameters())
    compressor=QueryCompressor(16,internal_dim=16,heads=4,dropout=0)
    for p in compressor.parameters(): p.requires_grad_(False)
    raw=short_context(model,batch[0],by_id,torch.device('cpu'))
    compressed=short_context(model,batch[0],by_id,torch.device('cpu'),compressor)
    assert raw.shape==(1,32,16) and compressed.shape==(1,4,16)
    assert all(p.grad is None for p in compressor.parameters())


def test_metrics_keep_sequence_failures_and_pair_failures():
    rows=[dict(uid='a',kind='presence',size=4,pair_id='p',reference='Yes',generated='Yes',label=1),
          dict(uid='b',kind='presence',size=4,pair_id='p',reference='No',generated='Yes',label=2),
          dict(uid='c',kind='sequence',size=4,pair_id=None,reference='a; b; c; d',generated='a; b; c; d; d',label=1)]
    scores=score_short(rows)
    assert scores['4']['presence_balanced_accuracy']==.5 and scores['4']['pair_both_correct']==0
    assert scores['4']['sequence']['accuracy']==0 and scores['2']['first'] is None


def test_l1_match_requires_identical_source_query_and_reference():
    import pytest
    ev=events(); by_id={e['uid']:e for e in ev}; ids=[e['uid'] for e in ev[:4]]
    row=dict(uid='q',kind='first',size=4,query='first?',reference='getting into bed',
             source_ids=ids,actual_source_ids=ids,generated='getting into bed',correct=True,context_tokens=32)
    raw=dict(rows=[row]); compressed=dict(rows=[dict(row,generated='lying in bed',correct=False,context_tokens=4)])
    result=build_l1_match(raw,compressed,by_id)
    assert result['transitions']['first']['raw_only_correct']==1
    assert len(result['rows'][0]['source_sha256'])==64
    compressed['rows'][0]['source_ids']=ids[:3]
    with pytest.raises(ValueError): build_l1_match(raw,compressed,by_id)
