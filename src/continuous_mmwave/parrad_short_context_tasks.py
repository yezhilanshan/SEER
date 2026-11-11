"""Chronological single/two/four-event tasks and an exact-budget sampler."""
from collections import Counter
import random

from continuous_mmwave.parrad_single_event_qa import build_single_tasks, CONFUSIONS, score_predictions
from continuous_mmwave.parrad_stage4_qa import ACTIVITIES, grade_answer


def short_query(kind,size,activity=None):
    prefix=f'The memory represents {size} events in chronological order. '
    choices='Allowed activity names: '+'; '.join(ACTIVITIES)+'. '
    if kind=='presence': return prefix+f'Does it include {ACTIVITIES[activity]}? Answer only Yes or No.'
    if kind in ('first','last'): return prefix+choices+f'What is the {kind} activity? Answer only with one activity name.'
    if kind=='sequence': return prefix+choices+'List all activities in order. Answer only with a semicolon-separated list of activity names.'
    raise ValueError('unknown short QA kind')


def build_short_tasks(events,seed):
    rng=random.Random(seed); groups=[]; tails=[]; tasks=[]; pairs=[]
    by_event={e['uid']:e for e in events}
    if len(by_event)!=len(events): raise ValueError('duplicate events')
    single=build_single_tasks(events,seed)
    for pair in single['pairs']:
        ids=[]
        for old in (pair['yes'],pair['no']):
            task=dict(old,size=1,source_ids=[old['event_uid']],group_uid=old['event_uid'])
            tasks.append(task); ids.append(task['uid'])
        pairs.append(dict(uid=pair['uid'],size=1,cohort=pair['cohort'],activity=pair['activity'],
                          group_uid=pair['yes']['event_uid'],task_ids=ids))
    for old in single['identify']:
        tasks.append(dict(old,size=1,source_ids=[old['event_uid']],group_uid=old['event_uid']))
    for recording in sorted({e['recording'] for e in events}):
        rows=sorted((e for e in events if e['recording']==recording),key=lambda e:e['start'])
        if len({e['cohort'] for e in rows})!=1 or any(a['end']>b['start']+1e-6 for a,b in zip(rows,rows[1:])):
            raise ValueError('recording must have one cohort and nonoverlapping events')
        for size in (2,4):
            complete=len(rows)//size*size
            if complete<len(rows): tails.append(dict(recording=recording,size=size,source_ids=[e['uid'] for e in rows[complete:]]))
            for start in range(0,complete,size):
                group=rows[start:start+size]; uid=f'{recording}:group{size}:{start}'
                groups.append(dict(uid=uid,size=size,recording=recording,cohort=group[0]['cohort'],
                    source_ids=[e['uid'] for e in group],labels=[e['label'] for e in group],
                    query_activity=group[(start//size)%size]['label'],start=group[0]['start'],end=group[-1]['end']))
    for group in groups:
        size=group['size']; uid=group['uid']; activity=group['query_activity']
        eligible=[n for n in groups if n['size']==size and n['cohort']==group['cohort'] and activity not in n['labels']]
        if not eligible: raise ValueError('no opposite short-context group')
        for confusion in CONFUSIONS[activity]:
            hard=[n for n in eligible if confusion in n['labels']]
            if hard: eligible=hard; break
        negative=rng.choice(eligible); pair_id=uid+':pair'; query=short_query('presence',size,activity)
        ids=[]
        for source,answer in ((group,'Yes'),(negative,'No')):
            task=dict(uid=pair_id+':'+answer,size=size,kind='presence',label=source['labels'][0],
                cohort=source['cohort'],source_ids=source['source_ids'],group_uid=source['uid'],
                pair_id=pair_id,query=query,answer=answer)
            tasks.append(task); ids.append(task['uid'])
        pairs.append(dict(uid=pair_id,size=size,cohort=group['cohort'],activity=activity,group_uid=uid,task_ids=ids))
        for kind in ('first','last')+(('sequence',) if size==4 else ()):
            answer='; '.join(ACTIVITIES[label] for label in group['labels']) if kind=='sequence' else ACTIVITIES[group['labels'][0 if kind=='first' else -1]]
            tasks.append(dict(uid=uid+':'+kind,size=size,kind=kind,label=group['labels'][0 if kind!='last' else -1],
                cohort=group['cohort'],source_ids=group['source_ids'],group_uid=uid,pair_id=None,
                query=short_query(kind,size),answer=answer))
    return dict(tasks=tasks,pairs=pairs,groups=groups,tails=tails,by_uid={t['uid']:t for t in tasks})


def short_training_batches(taskset,seed):
    rng=random.Random(seed); by_uid=taskset['by_uid']; pools={size:{} for size in (1,2,4)}
    for pair in taskset['pairs']:
        if pair['cohort']=='fit': pools[pair['size']].setdefault(pair['activity'],[]).append(pair)
    kinds={1:['identify'],2:['first','last'],4:['first','last','sequence']}
    group_tasks={(t['group_uid'],t['kind']):t for t in taskset['tasks'] if t['cohort']=='fit' and t['kind']!='presence'}
    counters=Counter(); batches=[]
    for size in [1,2,1,4]*125:
        classes=sorted(pools[size]); index=counters[size]
        if not classes: raise ValueError('missing fit size pool')
        # Every class sees a complete role cycle; role counts differ by at most one.
        pair=rng.choice(pools[size][classes[(index//len(kinds[size]))%len(classes)]])
        kind=kinds[size][index%len(kinds[size])]
        batches.append([by_uid[uid] for uid in pair['task_ids']]+[group_tasks[pair['group_uid'],kind]])
        counters[size]+=1
    return batches


def score_short(rows):
    result={}
    for size in (1,2,4):
        selected=[r for r in rows if r['size']==size]
        base=score_predictions([r for r in selected if r['kind'] in ('identify','presence')])
        for kind in ('first','last','sequence'):
            group=[r for r in selected if r['kind']==kind]
            correct=sum(grade_answer(r['generated'],r['reference'],'summary' if kind=='sequence' else 'first') for r in group)
            base[kind]=None if not group else dict(n=len(group),correct=correct,accuracy=correct/len(group))
        result[str(size)]=base
    return result
