from collections import Counter
import torch
from tests.test_short_context_qa import events
from tests.test_stage4_qa import Tokenizer
from continuous_mmwave.models.event_former import TinyLLMAdapter
from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.parrad_stage4_qa import ACTIVITIES
from continuous_mmwave.parrad_event_boundary_tasks import build_boundary_tasks, boundary_batches, score_boundary
from continuous_mmwave.parrad_event_boundary_qa import boundary_context, boundary_loss, development_gates


def test_positions_keep_source_order_repeats_and_old_tasks():
    rows=events(); rows[0]['label']=rows[1]['label']=1
    tasks=build_boundary_tasks(rows,42)
    group=next(g for g in tasks['groups'] if g['size']==4 and g['source_ids'][0]==rows[0]['uid'])
    by_uid=tasks['by_uid']
    assert by_uid[group['uid']+':position:1']['answer']=='getting into bed'
    assert by_uid[group['uid']+':position:2']['answer']=='getting into bed'
    for t in tasks['tasks']:
        if t['kind']=='position':
            source={e['uid']:e for e in rows}[t['source_ids'][t['position']-1]]
            assert t['answer']==ACTIVITIES[source['label']]
            assert all(name in t['query'] for name in ACTIVITIES)
    assert by_uid[group['uid']+':sequence']['answer'].startswith('getting into bed; getting into bed;')


def test_matched_batches_exact_budget_and_class_role_coverage():
    taskset=build_boundary_tasks(events(),42); batches=boundary_batches(taskset,42)
    assert len(batches)==500 and batches==boundary_batches(taskset,42)
    assert Counter(b[0]['size'] for b in batches)=={1:200,2:100,4:200}
    counts=Counter((b[0]['size'],b[2]['kind'],b[2].get('position')) for b in batches)
    assert counts=={(1,'identify',None):200,(2,'position',1):50,(2,'position',2):50,
        (4,'sequence',None):100,(4,'position',1):25,(4,'position',2):25,(4,'position',3):25,(4,'position',4):25}
    assert all(b[0]['answer']=='Yes' and b[1]['answer']=='No' and all(t['cohort']=='fit' for t in b) for b in batches)
    pair_activity={p['task_ids'][0]:p['activity'] for p in taskset['pairs']}
    for role in counts:
        selected=[b for b in batches if (b[0]['size'],b[2]['kind'],b[2].get('position'))==role]
        coverage=Counter(pair_activity[b[0]['uid']] for b in selected)
        available={p['activity'] for p in taskset['pairs'] if p['cohort']=='fit' and p['size']==role[0]}
        assert set(coverage)==available and max(coverage.values())-min(coverage.values())<=1


def test_markers_keep_raw_source_order_single_parity_and_frozen_gradients():
    torch.manual_seed(7); rows=events(); by_id={r['uid']:r for r in rows}
    taskset=build_boundary_tasks(rows,42); batch=next(b for b in boundary_batches(taskset,42) if b[0]['size']==4 and b[2]['kind']=='position')
    model=EventQAAdapter(16,8); teacher=TinyLLMAdapter(128,16,2,4); tokenizer=Tokenizer()
    snapshots={uid:r['tokens'].clone() for uid,r in by_id.items()}
    raw=boundary_context(model,teacher,tokenizer,batch[0],by_id,'cpu',False)
    marked=boundary_context(model,teacher,tokenizer,batch[0],by_id,'cpu',True)
    offset=0
    for index in range(4):
        ids=tokenizer(f'Event {index+1}:\n',return_tensors='pt')['input_ids']
        marker=teacher.embed_input_ids(ids)
        torch.testing.assert_close(marked[:,offset:offset+ids.shape[1]],marker,rtol=0,atol=0)
        offset+=ids.shape[1]
        torch.testing.assert_close(marked[:,offset:offset+8],raw[:,index*8:(index+1)*8],rtol=0,atol=0)
        offset+=8
    single=next(t for t in taskset['tasks'] if t['size']==1)
    torch.testing.assert_close(boundary_context(model,teacher,tokenizer,single,by_id,'cpu',True),
        boundary_context(model,teacher,tokenizer,single,by_id,'cpu',False),rtol=0,atol=0)
    loss=boundary_loss(model,teacher,tokenizer,batch,by_id,'cpu',True); loss.backward()
    assert model.output.weight.grad.abs().sum()>0 and all(p.grad is None for p in teacher.parameters())
    assert all(torch.equal(r['tokens'],snapshots[uid]) for uid,r in by_id.items())


def test_position_metric_keeps_wrong_order_and_missing_position():
    rows=[dict(kind='position',size=4,position=1,label=1,reference='getting into bed',generated='getting into bed'),
          dict(kind='position',size=4,position=2,label=2,reference='getting out of bed',generated='getting into bed'),
          dict(kind='sequence',size=4,label=1,reference='a; b; b; c',generated='a; b; c')]
    m=score_boundary(rows)['4']
    assert m['positions']['1']['accuracy']==1 and m['positions']['2']['accuracy']==0
    assert m['positions']['3'] is None and m['position_mean_accuracy']==.5
    assert m['sequence']['accuracy']==0


def test_gates_require_single_retention_all_positions_and_full_sequence():
    before={'1':dict(identify_macro_recall=.805,presence_balanced_accuracy=.843,pair_both_correct=.747)}
    after={'1':dict(identify_macro_recall=.81,presence_balanced_accuracy=.84,pair_both_correct=.74)}
    for size in ('2','4'):
        after[size]=dict(presence_balanced_accuracy=.85,pair_both_correct=.75,
            first=dict(accuracy=.85),last=dict(accuracy=.85),positions={str(i):dict(accuracy=.85) for i in range(1,int(size)+1)})
    after['4']['sequence']=dict(accuracy=.5)
    assert all(development_gates(before,after).values())
    after['4']['positions']['2']['accuracy']=.5
    after['1']['identify_macro_recall']=.77
    gates=development_gates(before,after)
    assert not gates['single_retained'] and not gates['4_position_2']
