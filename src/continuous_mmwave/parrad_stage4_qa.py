"""Chat-aligned mixed-memory QA training with explicit selected-event evidence."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import time

import torch
from torch.nn import functional as F
import yaml

from continuous_mmwave.parrad_memory import EventMemory, MemoryUnit
from continuous_mmwave.parrad_stage4_runner import (
    QueryCompressor, sha, atomic_json, validate_provenance, load_frontend, upstream_digest,
    compress_example, replay)
from continuous_mmwave.stages.parrad_stage4 import encode_query, generate_from_memory, chat_supervision

ACTIVITIES = ('falling to the floor','getting into bed','getting out of bed','lying in bed',
              'rolling over in bed','sitting down on the bed','sitting down on a chair','sitting in bed',
              'standing up from bed','standing up from a chair','standing up from the floor',
              'walking toward the bed','walking toward the chair','walking into the room')


def task_query(kind, activity=None):
    if kind=='presence':
        return f'Does the retrieved event memory include {ACTIVITIES[activity]}? Answer only Yes or No.'
    if kind=='summary':
        return 'List the activities in the retrieved event memory in chronological order. Answer only with a semicolon-separated list of activity names.'
    if kind in ('first','last'):
        return f'What is the {kind} activity in the retrieved event memory? Answer only with one activity name.'
    raise ValueError('unknown QA kind')


def activity_answer(kind, evidence, activity=None):
    labels=[int(e['label']) for e in evidence]
    if not labels or any(label not in range(14) for label in labels):
        raise ValueError('QA needs annotated selected evidence')
    if kind=='presence': return 'Yes' if activity in labels else 'No'
    if kind=='first': return ACTIVITIES[labels[0]]
    if kind=='last': return ACTIVITIES[labels[-1]]
    if kind=='summary': return '; '.join(ACTIVITIES[label] for label in labels)
    raise ValueError('unknown QA kind')


def grade_answer(generated, reference, kind):
    def normalized(text): return re.sub(r'\s+', ' ', text.strip().lower().rstrip('.! '))
    if kind=='presence':
        return bool(re.fullmatch(r'(yes|no)[.!]?',generated.strip(),flags=re.I)) and normalized(generated)==normalized(reference)
    return normalized(generated)==normalized(reference)


def presence_pairs(tasks):
    by_query={}
    for row in tasks:
        if row['kind']=='presence': by_query.setdefault(row['query'],{}).setdefault(row['answer'],row['uid'])
    return [(answers['Yes'],answers['No']) for _,answers in sorted(by_query.items()) if {'Yes','No'}.issubset(answers)]


def render_context(model, projection, selected, events, device):
    if not selected or [u['level'] for u in selected]!=sorted([u['level'] for u in selected],reverse=True):
        raise ValueError('context must be ordered L2/L1/allL0')
    tokens=[]; retained=[]
    for unit in selected:
        source=[events[uid] for uid in unit['source_ids']]
        if unit['level']==0:
            if len(source)!=1: raise ValueError('raw L0 requires one event')
            tokens.append(source[0]['tokens'][None].to(device)); continue
        summary,raw=compress_example(model,(unit['level'],source),device)
        with torch.no_grad(): source_key=projection(raw)
        key=projection(summary)
        retained.append((1-F.cosine_similarity(key,source_key)).mean())
        tokens.append(summary)
    if not retained: raise ValueError('QA training context requires a trainable summary')
    return torch.cat(tokens,1),torch.stack(retained).mean()


def qa_loss(adapter, context, ids, attention, target, retention, retain_weight):
    if any(p.requires_grad for p in adapter.parameters()): raise ValueError('FullLM/LoRA must be frozen')
    logits=adapter.caption_logits(context.to(adapter.compute_dtype),ids,attention)
    offset=context.shape[1]
    if offset:
        prediction=logits[:,offset-1:offset+ids.shape[1]-1]; labels=ids; valid=target.bool()&attention.bool()
    else:
        prediction=logits[:,:-1]; labels=ids[:,1:]; valid=target[:,1:].bool()&attention[:,1:].bool()
    ce=F.cross_entropy(prediction[valid].float(),labels[valid])
    return dict(loss=ce+retain_weight*retention,answer_ce=ce,retain_loss=retention)


@torch.no_grad()
def build_tasks(model,projection,adapter,tokenizer,payload,settings,device):
    model.eval(); tasks=[]; query_keys={}
    for recording in sorted({e['recording'] for e in payload['events']}):
        rows=[e for e in payload['events'] if e['recording']==recording]
        cohort=rows[0]['cohort']
        for count in sorted(set([n for n in (12,24,40,56,72,88) if n<=len(rows)]+[len(rows)])):
            memory=EventMemory(model,projection,**settings['memory'])
            for e in rows[:count]:
                memory.add(MemoryUnit(e['uid'],e['tokens'].to(device),e['start'],e['end'],e['end'],1.,e['key'].to(device)))
            for kind,activity in [('summary',None),('first',None),('last',None)]+[('presence',i) for i in range(14)]:
                query=task_query(kind,activity)
                if query not in query_keys: query_keys[query]=encode_query(adapter,projection,tokenizer,query,device)
                key=query_keys[query]
                _,units=memory.context(key,now=rows[count-1]['end'])
                selected=[dict(level=u.level,source_ids=list(u.source_ids or (u.uid,))) for u in units]
                ids=[uid for unit in selected for uid in unit['source_ids']]
                if len(set(ids))!=len(ids): raise ValueError('overlapping selected evidence')
                events={e['uid']:e for e in rows[:count]}
                evidence=[events[uid] for uid in ids]
                if [e['start'] for e in evidence]!=sorted(e['start'] for e in evidence):
                    raise ValueError('selected evidence is not chronological')
                answer=activity_answer(kind,evidence,activity)
                tasks.append(dict(uid=f'{recording}:{count}:{kind}:{activity}',recording=recording,
                                  cohort=cohort,prefix_count=count,kind=kind,activity=activity,
                                  query=query,answer=answer,selected=selected,evidence_ids=ids,
                                  evidence_labels=[e['label'] for e in evidence]))
    return tasks


def summarize_predictions(rows):
    result={}
    for kind in ('presence','summary','first','last'):
        selected=[r for r in rows if r['kind']==kind]
        result[kind]=None if not selected else dict(n=len(selected),
            accuracy=sum(r['correct'] for r in selected)/len(selected))
    binary=[r for r in rows if r['kind']=='presence']
    if binary:
        rates=[sum(r['correct'] for r in binary if r['reference']==answer)/sum(r['reference']==answer for r in binary)
               for answer in ('Yes','No') if any(r['reference']==answer for r in binary)]
        result['binary_balanced_accuracy']=sum(rates)/len(rates) if len(rates)==2 else None
    return result


@torch.no_grad()
def evaluate_qa(model,adapter,projection,tokenizer,tasks,events,settings,device,controls=True):
    model.eval(); rows=[]; stored={}
    for task in tasks:
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            context,retention=render_context(model,projection,task['selected'],events,device)
            ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
            metrics=qa_loss(adapter,context,ids,attention,target,retention,settings['retain_weight'])
        answer=generate_from_memory(adapter,tokenizer,context,task['query'],max_new_tokens=384 if task['kind']=='summary' else 24)
        rows.append(dict(uid=task['uid'],kind=task['kind'],query=task['query'],reference=task['answer'],
                         generated=answer,correct=grade_answer(answer,task['answer'],task['kind']),
                         context_tokens=context.shape[1],**{k:float(v) for k,v in metrics.items()}))
        stored[task['uid']]=context.detach()
    by_id={r['uid']:r for r in rows}; pairs=presence_pairs(tasks)
    pair_rows=[dict(yes_uid=yes,no_uid=no,both_correct=by_id[yes]['correct'] and by_id[no]['correct'],
                    answers_differ=by_id[yes]['generated'].strip()!=by_id[no]['generated'].strip()) for yes,no in pairs]
    blind=[]
    if controls:
        by_task={t['uid']:t for t in tasks}
        for uid in dict.fromkeys(uid for pair in pairs for uid in pair):
            task=by_task[uid]
            answer=generate_from_memory(adapter,tokenizer,stored[uid][:,:0],task['query'],max_new_tokens=24)
            blind.append(dict(uid=uid,kind=task['kind'],reference=task['answer'],generated=answer,
                              correct=grade_answer(answer,task['answer'],task['kind'])))
    return dict(rows=rows,metrics=summarize_predictions(rows),
                mean_answer_ce=sum(r['answer_ce'] for r in rows)/len(rows),
                mean_summary_source_cosine=1-sum(r['retain_loss'] for r in rows)/len(rows),
                paired_presence=pair_rows,
                paired_both_correct=None if not pairs else sum(r['both_correct'] for r in pair_rows)/len(pairs),
                no_memory_controls=blind,no_memory_metrics=summarize_predictions(blind))


def train_qa(model,runtime,tokenizer,tasks,events,settings,output,device,source):
    adapter=runtime.controller.event_former.adapter; projection=runtime.controller.semantic_projection
    fit=[t for t in tasks if t['cohort']=='fit']; dev=[t for t in tasks if t['cohort']=='dev']
    print(json.dumps(dict(phase='dev_before',tasks=len(dev))),flush=True)
    before=evaluate_qa(model,adapter,projection,tokenizer,dev,events,settings,device)
    atomic_json(output/'dev-before.json',before)
    digest=upstream_digest(runtime)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=settings['amp_initial_scale'],enabled=device.type=='cuda')
    pools={}
    for t in fit:
        category=t['kind'] if t['kind']!='presence' else t['answer'].lower()
        pools.setdefault(category,[]).append(t)
    categories=['summary','first','last','yes','no']
    if any(not pools.get(c) for c in categories): raise ValueError('missing balanced task pool')
    rng=random.Random(settings['seed']); training=[]
    for step in range(1,settings['steps']+1):
        task=rng.choice(pools[categories[(step-1)%len(categories)]])
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            context,retention=render_context(model,projection,task['selected'],events,device)
            ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
            metrics=qa_loss(adapter,context,ids,attention,target,retention,settings['retain_weight'])
        if not torch.isfinite(metrics['loss']): raise RuntimeError('nonfinite QA loss')
        old_scale=scaler.get_scale(); scaler.scale(metrics['loss']).backward(); scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('nonfinite QA gradient, no update counted')
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale()<old_scale: raise RuntimeError('skipped QA optimizer update')
        row=dict(step=step,kind=task['kind'],category=categories[(step-1)%5],uid=task['uid'],
                 gradient_norm=float(norm),**{k:float(v.detach()) for k,v in metrics.items()})
        training.append(row)
        with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if step%10==0: print(json.dumps(dict(phase='train',**row)),flush=True)
        if step in (100,250,settings['steps']):
            torch.save(dict(step=step,model_settings=model.settings,model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict(),
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                source=source,settings=settings,task_sha256=sha(output/'tasks.json'),frozen_upstream_digest=digest),
                output/f'compressor-step{step}.pt')
    print(json.dumps(dict(phase='dev_after',tasks=len(dev))),flush=True)
    after=evaluate_qa(model,adapter,projection,tokenizer,dev,events,settings,device)
    if upstream_digest(runtime)!=digest: raise RuntimeError('frozen upstream changed')
    result=dict(updates=len(training),init_step=100,total_compressor_updates=100+len(training),
                before=before,after=after,upstream_unchanged=True,frozen_upstream_digest=digest,
                categories=dict(Counter(r['category'] for r in training)),
                checkpoint_sha256=sha(output/f"compressor-step{settings['steps']}.pt"))
    atomic_json(output/'qa-result.json',result); model.eval()
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv); settings=yaml.safe_load(args.config.read_text())['stage4']
    output=Path(settings['output_dir'])
    if output.exists(): raise FileExistsError('refuse duplicate QA output')
    output.mkdir(parents=True)
    torch.manual_seed(settings['seed']); torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    started=time.time(); status=dict(status='running',settings=settings,started_at=started,
                                    config_sha256=sha(args.config),source_code_sha256=sha(__file__),formal_go=False)
    atomic_json(output/'status.json',status)
    try:
        runtime,tokenizer,frontend=load_frontend(settings,device)
        root=Path(settings['source_output']); manifest=json.loads((root/'cache-manifest.json').read_text())
        provenance={name:sha(settings[name]) for name in ('frontend_config','stage3_checkpoint','threshold_artifact')}
        provenance.update(stage1_checkpoint=sha(frontend['stage']['stage1_checkpoint']),
            stage2_checkpoint=sha(frontend['stage3']['stage2_checkpoint']),
            event_former_code=sha(Path(__file__).parent/'models/event_former.py'),
            projection_code=sha(Path(__file__).parent/'models/semantic.py'))
        validate_provenance(manifest['provenance'],provenance)
        if sha(root/'events.pt')!=settings['cache_sha256'] or sha(settings['init_checkpoint'])!=settings['init_sha256']:
            raise ValueError('cache or initial checkpoint changed')
        payload=torch.load(root/'events.pt',map_location='cpu',weights_only=False)
        if payload['upstream_digest']!=upstream_digest(runtime): raise ValueError('cached teacher changed')
        state=torch.load(settings['init_checkpoint'],map_location='cpu',weights_only=False)
        if state['step']!=100 or state['cache_sha256']!=settings['cache_sha256']: raise ValueError('invalid initialization')
        model=QueryCompressor(**state['model_settings']).to(device); model.load_state_dict(state['model_state'])
        adapter=runtime.controller.event_former.adapter; projection=runtime.controller.semantic_projection
        tasks=build_tasks(model,projection,adapter,tokenizer,payload,settings,device)
        atomic_json(output/'tasks.json',dict(tasks=tasks,selection='fixed initial100step native retrieval',
            target_source='selected TRAIN GT event evidence only; no annotation in model input'))
        events={e['uid']:e for e in payload['events']}
        print(json.dumps(dict(phase='tasks_ready',counts=dict(Counter(t['cohort'] for t in tasks)))),flush=True)
        source=dict(provenance=provenance,init_sha256=settings['init_sha256'],cache_sha256=settings['cache_sha256'])
        result=train_qa(model,runtime,tokenizer,tasks,events,settings,output,device,source)
        print(json.dumps(dict(phase='online_replay')),flush=True)
        replay(settings,runtime,tokenizer,model,frontend,output)
        # Independent checkpoint readback validates saved optimization count.
        checkpoint=output/f"compressor-step{settings['steps']}.pt"
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        actual_steps=sorted({int(s['step']) for s in saved['optimizer_state']['state'].values()})
        if actual_steps!=[settings['steps']] or not all(torch.isfinite(t).all() for t in saved['model_state'].values()):
            raise RuntimeError('saved checkpoint failed readback')
        torch.save(dict(model_settings=model.settings,model_state=saved['model_state'],source=source,
                        step=settings['steps'],init_step=100),output/'compressor-inference.pt')
        if upstream_digest(runtime)!=result['frozen_upstream_digest']:
            raise RuntimeError('frozen upstream changed during replay')
        atomic_json(output/'readback.json',dict(status='passed',actual_steps=actual_steps,
            checkpoint_sha256=sha(checkpoint),inference_sha256=sha(output/'compressor-inference.pt'),
            frozen_upstream_unchanged=upstream_digest(runtime)==result['frozen_upstream_digest']))
        status.update(status='complete',duration_s=time.time()-started)
    except Exception as error:
        status.update(status='failed',error=repr(error),duration_s=time.time()-started)
        atomic_json(output/'status.json',status); raise
    atomic_json(output/'status.json',status); print(json.dumps(dict(phase='complete',duration_s=status['duration_s'])),flush=True)


if __name__=='__main__': main()
