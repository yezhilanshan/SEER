import importlib
from copy import deepcopy

import pytest


def checks():
    name='continuous_mmwave.parrad_qa_readback'
    assert importlib.util.find_spec(name), 'missing complete prediction readback checks'
    return importlib.import_module(name)


def fixture():
    tasks=[dict(uid='yes',kind='presence',size=2,label=0,position=None,pair_id='p',
        query='Does it include falling to the floor?',source_ids=['a','b'],answer='Yes',cohort='dev'),
        dict(uid='no',kind='presence',size=2,label=0,position=None,pair_id='p',
        query='Does it include falling to the floor?',source_ids=['c','d'],answer='No',cohort='dev')]
    events={u:dict(label=v) for u,v in [('a',0),('b',1),('c',2),('d',3)]}
    rows=[dict(uid=t['uid'],kind=t['kind'],size=2,label=0,position=None,pair_id='p',query=t['query'],
        source_ids=t['source_ids'],actual_source_ids=t['source_ids'],reference=t['answer'],
        generated=t['answer'],correct=True,context_tokens=16,arm='none',mode='raw',reader='text-qa-500',cohort='dev') for t in tasks]
    return dict(rows=rows,mode='raw',arm='none'),tasks,events


@pytest.mark.parametrize('mutation',['delete','duplicate','wrong_field','wrong_source','wrong_reader'])
def test_readback_rejects_incomplete_duplicate_or_misbound_predictions(mutation):
    r=checks(); result,tasks,events=fixture()
    r.validate_rows(result,tasks,events,'raw','none',new_reader=True)
    if mutation=='delete': result['rows'].pop()
    if mutation=='duplicate': result['rows'][1]=deepcopy(result['rows'][0])
    if mutation=='wrong_field': result['rows'][0]['label']=3
    if mutation=='wrong_source': result['rows'][0]['actual_source_ids']=['c','d']
    if mutation=='wrong_reader': result['rows'][0]['reader']='original'
    with pytest.raises(ValueError): r.validate_rows(result,tasks,events,'raw','none',new_reader=True)


@pytest.mark.parametrize('mutation',['source','reference'])
def test_swapped_readback_requires_other_pair_evidence_and_answer(mutation):
    r=checks(); result,tasks,events=fixture(); swaps={'yes':tasks[1],'no':tasks[0]}
    result['mode']='swapped'
    for row in result['rows']:
        row['mode']='swapped'; row['actual_source_ids']=swaps[row['uid']]['source_ids']
        row['evidence_reference']='No' if row['uid']=='yes' else 'Yes'
    r.validate_rows(result,tasks,events,'swapped','none',swaps=swaps,query_labels={'p':0},new_reader=True)
    if mutation=='source': result['rows'][0]['actual_source_ids']=tasks[0]['source_ids']
    else: result['rows'][0]['evidence_reference']='Yes'
    with pytest.raises(ValueError): r.validate_rows(result,tasks,events,'swapped','none',swaps=swaps,query_labels={'p':0},new_reader=True)


def test_swapped_presence_checks_queried_activity_not_source_event_label():
    r=checks(); result,tasks,events=fixture(); swaps={'yes':tasks[1],'no':tasks[0]}
    tasks[1]['label']=2; result['rows'][1]['label']=2
    result['mode']='swapped'
    for row in result['rows']:
        row['mode']='swapped'; row['actual_source_ids']=swaps[row['uid']]['source_ids']
        row['evidence_reference']='No' if row['uid']=='yes' else 'Yes'
    r.validate_rows(result,tasks,events,'swapped','none',swaps=swaps,query_labels={'p':0},new_reader=True)
    result['rows'][1]['evidence_reference']='No'
    with pytest.raises(ValueError,match='evidence'):
        r.validate_rows(result,tasks,events,'swapped','none',swaps=swaps,query_labels={'p':0},new_reader=True)


def test_gate_readback_rejects_claimed_pass_when_recomputed_metrics_fail():
    r=checks()
    single=dict(identify_macro_recall=.9,presence_balanced_accuracy=.9,pair_both_correct=.8)
    multi=dict(presence_balanced_accuracy=.9,pair_both_correct=.8,first=dict(accuracy=.9),
        last=dict(accuracy=.9),positions={str(i):dict(accuracy=.9) for i in range(1,5)},sequence=dict(accuracy=.6))
    before={'1':single,'2':multi,'4':multi}; after=deepcopy(before)
    after['4']['sequence']['accuracy']=0.
    with pytest.raises(ValueError,match='gate'):
        r.validate_gates(before,after,dict(gates={},development_pass=True))
