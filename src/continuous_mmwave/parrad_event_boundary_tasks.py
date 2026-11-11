"""Position tasks and identical fixed-budget batches for boundary A/B."""
from collections import Counter
import random
from continuous_mmwave.parrad_short_context_tasks import build_short_tasks, score_short
from continuous_mmwave.parrad_stage4_qa import ACTIVITIES, grade_answer


def build_boundary_tasks(events,seed):
    result=build_short_tasks(events,seed)
    for group in result['groups']:
        for position,label in enumerate(group['labels'],1):
            query=(f'The memory represents {group["size"]} events in chronological order. '
                   +'Allowed activity names: '+'; '.join(ACTIVITIES)+'. '
                   +f'What activity occurs in event {position}? Answer only with one activity name.')
            result['tasks'].append(dict(uid=f'{group["uid"]}:position:{position}',kind='position',
                position=position,size=group['size'],cohort=group['cohort'],label=label,
                source_ids=group['source_ids'],group_uid=group['uid'],pair_id=None,query=query,answer=ACTIVITIES[label]))
    result['by_uid']={t['uid']:t for t in result['tasks']}
    return result


def boundary_batches(taskset,seed):
    rng=random.Random(seed); pools={size:{} for size in (1,2,4)}
    for pair in taskset['pairs']:
        if pair['cohort']=='fit': pools[pair['size']].setdefault(pair['activity'],[]).append(pair)
    roles={1:[('identify',None)],2:[('position',1),('position',2)],
           4:[('sequence',None),('position',1),('sequence',None),('position',2),
              ('sequence',None),('position',3),('sequence',None),('position',4)]}
    content={(t['group_uid'],t['kind'],t.get('position')):t for t in taskset['tasks']
             if t['cohort']=='fit' and t['kind']!='presence'}
    sizes=Counter(); classes_by_role=Counter(); batches=[]
    for size in [1,4,1,2,4]*100:
        kind,position=roles[size][sizes[size]%len(roles[size])]; role=(size,kind,position)
        classes=sorted(pools[size])
        if not classes: raise ValueError('missing fit size pool')
        activity=classes[classes_by_role[role]%len(classes)]
        pair=rng.choice(pools[size][activity])
        batches.append([taskset['by_uid'][u] for u in pair['task_ids']]+[content[(pair['group_uid'],kind,position)]])
        sizes[size]+=1; classes_by_role[role]+=1
    return batches


def score_boundary(rows):
    result=score_short(rows)
    for size,m in result.items():
        m['positions']={}
        for position in range(1,int(size)+1):
            subset=[r for r in rows if r['size']==int(size) and r['kind']=='position' and r['position']==position]
            count=sum(grade_answer(r['generated'],r['reference'],'first') for r in subset)
            m['positions'][str(position)]=None if not subset else dict(n=len(subset),correct=count,accuracy=count/len(subset))
        present=[p['accuracy'] for p in m['positions'].values() if p is not None]
        m['position_mean_accuracy']=None if not present else sum(present)/len(present)
    return result
