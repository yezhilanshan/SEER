import torch
from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.parrad_single_event_qa import build_single_tasks, score_predictions, batch_loss, training_batches
from continuous_mmwave.models.event_former import TinyLLMAdapter
from tests.test_stage4_qa import Tokenizer


def events():
    return [dict(uid=f'{cohort}-{label}-{n}',cohort=cohort,label=label,recording=cohort,tokens=torch.randn(8,16))
            for cohort in ('fit','dev') for label in (1,2,5,6) for n in range(2)]


def test_pairs_same_question_opposite_events_and_no_cohort_leakage():
    rows=events(); tasks=build_single_tasks(rows,42); by_id={e['uid']:e for e in rows}
    assert tasks==build_single_tasks(rows,42)
    for pair in tasks['pairs']:
        yes,no=pair['yes'],pair['no']
        assert yes['query']==no['query'] and yes['answer']=='Yes' and no['answer']=='No'
        a,b=by_id[yes['event_uid']],by_id[no['event_uid']]
        assert a['cohort']==b['cohort']==pair['cohort'] and a['label']!=b['label']
        assert b['label']=={1:2,2:1,5:6,6:5}[a['label']]


def test_adapter_starts_identity_does_not_mutate_retrieval_inputs_and_learns():
    model=EventQAAdapter(16,8); x=torch.randn(1,8,16); saved=x.clone()
    torch.testing.assert_close(model(x),x,rtol=0,atol=0)
    opt=torch.optim.AdamW(model.parameters(),lr=.01)
    model(x).square().mean().backward(); opt.step()
    assert not torch.equal(model(x),x)
    torch.testing.assert_close(x,saved,rtol=0,atol=0)


def test_real_loss_backpropagates_only_into_qa_adapter():
    teacher=TinyLLMAdapter(128,16,2,4); model=EventQAAdapter(16,8)
    rows=events(); tasks=build_single_tasks(rows,42); pair=tasks['pairs'][0]
    loss=batch_loss(model,teacher,Tokenizer(),[pair['yes'],pair['no'],tasks['identify'][0]],
                    {e['uid']:e for e in rows},'cpu')
    loss.backward()
    assert model.output.weight.grad.abs().sum()>0
    assert all(p.grad is None for p in teacher.parameters())


def test_metrics_require_both_pair_answers_and_report_missing_classes():
    rows=[dict(uid='a',kind='presence',reference='Yes',generated='Yes',pair_id='p'),
          dict(uid='b',kind='presence',reference='No',generated='Yes',pair_id='p'),
          dict(uid='c',kind='identify',reference='getting into bed',generated='getting into bed',label=1)]
    scores=score_predictions(rows)
    assert scores['presence_balanced_accuracy']==.5 and scores['pair_both_correct']==0
    assert scores['identify_macro_recall']==1 and scores['identify_classes']['2'] is None


def test_training_identification_and_binary_questions_cover_each_fit_class():
    tasks=build_single_tasks(events(),42)
    batches=training_batches(tasks,8,42)
    assert batches==training_batches(tasks,8,42)
    assert [batch[2]['label'] for batch in batches]==[1,2,5,6]*2
    assert all([task['answer'] for task in batch[:2]]==['Yes','No'] for batch in batches)
    assert all(task['cohort']=='fit' for batch in batches for task in batch)
