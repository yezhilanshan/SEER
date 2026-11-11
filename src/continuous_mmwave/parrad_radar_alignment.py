"""Single radar event to the frozen text-QA reader: tasks and aligned losses."""
from collections import Counter
import random

import torch
from torch.nn import functional as F

from continuous_mmwave.parrad_single_event_qa import build_single_tasks
from continuous_mmwave.parrad_stage4_qa import ACTIVITIES,grade_answer


def build_tasks(events,seed):
    native=build_single_tasks(events,seed); tasks=[]
    for pair in native['pairs']:
        for polarity in ('yes','no'):
            tasks.append(dict(pair[polarity],query_label=pair['activity']))
    tasks.extend(dict(t,query_label=None,pair_id=None) for t in native['identify'])
    if len({t['uid'] for t in tasks})!=len(tasks): raise ValueError('duplicate tasks')
    return tasks


def training_batches(tasks,seed):
    by_uid={t['uid']:t for t in tasks}; pools={}
    for t in tasks:
        if t['cohort']=='fit' and t['kind']=='presence' and t['answer']=='Yes':
            pools.setdefault(t['query_label'],[]).append(t)
    if set(pools)!=set(range(14)): raise ValueError('all 14 fit classes required')
    rng=random.Random(seed); batches=[]
    for step in range(500):
        yes=rng.choice(pools[step%14])
        no=by_uid[yes['pair_id']+':No']; identify=by_uid[yes['event_uid']+':identify']
        batch=[yes,no,identify]
        if any(t['cohort']!='fit' for t in batch): raise ValueError('dev training evidence')
        batches.append(batch)
    return batches


def teacher_query(task,events):
    return ('Use the following correct event description as evidence.\nObserved activity: '
            +ACTIVITIES[events[task['event_uid']]['label']]+'.\n\n'+task['query'])


def answer_positions(logits,ids,attention,target,offset):
    if ids.shape!=attention.shape or ids.shape!=target.shape or logits.shape[:2]!=(ids.shape[0],offset+ids.shape[1]):
        raise ValueError('answer masks/logit length mismatch')
    if offset:
        prediction=logits[:,offset-1:offset+ids.shape[1]-1]; labels=ids; valid=target.bool()&attention.bool()
    else:
        prediction=logits[:,:-1];labels=ids[:,1:];valid=target[:,1:].bool()&attention[:,1:].bool()
    if not valid.any(): raise ValueError('missing assistant targets')
    return prediction[valid],labels[valid]


def radar_answer_positions(model,reader,raw,ids,attention,target):
    if any(p.requires_grad for p in reader.parameters()): raise ValueError('QA reader must be frozen')
    if raw.ndim!=3 or raw.shape[1]!=8: raise ValueError('one eight-token radar event required')
    context=model(raw).to(reader.compute_dtype)
    return answer_positions(reader.caption_logits(context,ids,attention),ids,attention,target,8)


def answer_objective(student,labels,teacher=None,teacher_labels=None,*,weight=0.,temperature=2.):
    if weight<0 or temperature<=0: raise ValueError('invalid distillation settings')
    ce=F.cross_entropy(student.float(),labels); kl=ce.new_zeros(())
    if weight:
        if teacher is None or teacher.shape!=student.shape or not torch.equal(labels,teacher_labels):
            raise ValueError('teacher/student assistant target mismatch')
        kl=F.kl_div(F.log_softmax(student.float()/temperature,dim=-1),
                    F.softmax(teacher.detach().float()/temperature,dim=-1),reduction='batchmean')*temperature**2
    return dict(loss=ce+weight*kl,ce=ce,kl=kl)


def development_gates(metrics):
    return {key:metrics[key] is not None and metrics[key]>=floor for key,floor in
        (('identify_macro_recall',.85),('presence_balanced_accuracy',.9),('pair_both_correct',.8))}


def text_gates(metrics):
    return dict(identify=metrics['identify_macro_recall'] is not None and metrics['identify_macro_recall']>=.95,
        **{a:metrics['presence_classes'][a]['accuracy']>=.95 for a in ('Yes','No')})


def text_preflight_decision(metrics,policy='required'):
    if policy not in ('required','diagnostic-only'):
        raise ValueError('unknown text preflight policy')
    gates=text_gates(metrics);passed=all(gates.values())
    return dict(gates=gates,passed=passed,policy=policy,
        proceed=passed or policy=='diagnostic-only',selection='fit only; locked prompt')


def validate_predictions(result,tasks,events):
    rows=result['rows'];mode=result['mode']
    if mode not in ('raw','none','text','swapped'): raise ValueError('unknown mode')
    if [r['uid'] for r in rows]!=[t['uid'] for t in tasks] or len({r['uid'] for r in rows})!=len(rows):
        raise ValueError('incomplete or duplicate prediction set')
    paired={t['uid']:t for t in tasks}
    for r,t in zip(rows,tasks):
        if any(r[k]!=t.get(k) for k in ('uid','cohort','kind','label','query_label','pair_id','query','event_uid')) or r['reference']!=t['answer']:
            raise ValueError('prediction task binding mismatch')
        if r['reader']!='text-qa-500': raise ValueError('new single-event reader bypassed')
        actual=t['event_uid']
        if mode=='swapped':
            if t['kind']!='presence': raise ValueError('swap only presence')
            counterpart=paired[t['pair_id']+(':No' if t['answer']=='Yes' else ':Yes')]
            actual=counterpart['event_uid']
            evidence='Yes' if events[actual]['label']==t['query_label'] else 'No'
            if r['evidence_reference']!=evidence: raise ValueError('incorrect swap evidence answer')
        if r['actual_event_uid']!=actual or r['context_tokens']!=(0 if mode in ('none','text') else 8):
            raise ValueError('actual radar evidence mismatch')
        if mode=='text' and r['oracle_query']!=teacher_query(t,events): raise ValueError('teacher evidence mismatch')
        if r['correct']!=grade_answer(r['generated'],t['answer'],'presence' if t['kind']=='presence' else 'first'):
            raise ValueError('prediction grade mismatch')
