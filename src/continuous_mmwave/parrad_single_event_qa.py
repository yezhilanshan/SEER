"""Fixed-budget independent single-event QA adapter with matched controls."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import time

import torch
import yaml

from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.parrad_stage4_qa import ACTIVITIES, grade_answer, qa_loss
from continuous_mmwave.parrad_stage4_runner import sha, atomic_json, load_frontend, upstream_digest, validate_provenance
from continuous_mmwave.stages.parrad_stage4 import chat_supervision, generate_from_memory

CONFUSIONS={0:(10,8),1:(2,5),2:(1,8),3:(7,4),4:(3,7),5:(6,7),6:(5,9),7:(3,5),
            8:(9,2),9:(8,6),10:(8,0),11:(12,13),12:(11,13),13:(11,12)}
IDENTIFY_QUERY=('What activity occurs in this single event? Choose one of: '+
                '; '.join(ACTIVITIES)+'. Answer only with one activity name.')


def presence_query(label):
    return f'Does this single event show {ACTIVITIES[label]}? Answer only Yes or No.'


def build_single_tasks(events,seed):
    rng=random.Random(seed); pairs=[]; identify=[]
    if len({e['uid'] for e in events})!=len(events): raise ValueError('duplicate source event')
    for e in events:
        if e['cohort'] not in ('fit','dev') or e['label'] not in range(14):
            raise ValueError('unexpected cohort/activity')
        eligible=[n for n in events if n['cohort']==e['cohort'] and n['label']!=e['label']]
        if not eligible: raise ValueError('cohort needs at least two activities')
        for label in CONFUSIONS[e['label']]:
            hard=[n for n in eligible if n['label']==label]
            if hard: eligible=hard; break
        negative=rng.choice(eligible); pair_id=e['uid']+':pair'; query=presence_query(e['label'])
        def task(source,answer):
            return dict(uid=pair_id+':'+answer,event_uid=source['uid'],query=query,answer=answer,
                        kind='presence',label=source['label'],cohort=e['cohort'],pair_id=pair_id)
        pairs.append(dict(uid=pair_id,cohort=e['cohort'],activity=e['label'],
                          yes=task(e,'Yes'),no=task(negative,'No')))
        identify.append(dict(uid=e['uid']+':identify',event_uid=e['uid'],query=IDENTIFY_QUERY,
            answer=ACTIVITIES[e['label']],kind='identify',label=e['label'],cohort=e['cohort']))
    return dict(pairs=pairs,identify=identify)


def score_predictions(rows):
    correct=lambda r:grade_answer(r['generated'],r['reference'],'presence' if r['kind']=='presence' else 'first')
    binary=[r for r in rows if r['kind']=='presence']; identified=[r for r in rows if r['kind']=='identify']
    classes={str(label):None for label in range(14)}
    for label in range(14):
        group=[r for r in identified if r['label']==label]
        if group: classes[str(label)]=dict(n=len(group),correct=sum(correct(r) for r in group),
                                          recall=sum(correct(r) for r in group)/len(group))
    presence={}
    for answer in ('Yes','No'):
        group=[r for r in binary if r['reference']==answer]
        presence[answer]=None if not group else dict(n=len(group),correct=sum(correct(r) for r in group),
                                                   accuracy=sum(correct(r) for r in group)/len(group))
    grouped={}
    for r in binary: grouped.setdefault(r['pair_id'],[]).append(r)
    if any(len(group)!=2 or {r['reference'] for r in group}!={'Yes','No'} for group in grouped.values()):
        raise ValueError('each evaluation pair requires exactly one Yes and one No')
    pair_correct=sum(all(correct(r) for r in group) for group in grouped.values())
    return dict(n=len(rows),identify_n=len(identified),
        identify_accuracy=None if not identified else sum(correct(r) for r in identified)/len(identified),
        identify_macro_recall=None if not identified else sum(v['recall'] for v in classes.values() if v)/sum(v is not None for v in classes.values()),
        identify_classes=classes,presence_classes=presence,
        presence_balanced_accuracy=None if any(v is None for v in presence.values()) else sum(v['accuracy'] for v in presence.values())/2,
        pair_n=len(grouped),pair_correct=pair_correct,pair_both_correct=None if not grouped else pair_correct/len(grouped))


def batch_loss(model,teacher,tokenizer,tasks,events,device):
    losses=[]
    for task in tasks:
        context=model(events[task['event_uid']]['tokens'][None].to(device))
        ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
        losses.append(qa_loss(teacher,context,ids,attention,target,context.new_zeros(()),0.)['answer_ce'])
    return torch.stack(losses).mean()


def dev_tasks(tasks):
    return ([task for pair in tasks['pairs'] if pair['cohort']=='dev' for task in (pair['yes'],pair['no'])]+
            [t for t in tasks['identify'] if t['cohort']=='dev'])


def training_batches(tasks,steps,seed):
    pools={}
    for pair in tasks['pairs']:
        if pair['cohort']=='fit': pools.setdefault(pair['activity'],[]).append(pair)
    if not pools: raise ValueError('no fit task pairs')
    classes=sorted(pools); rng=random.Random(seed)
    by_event={t['event_uid']:t for t in tasks['identify'] if t['cohort']=='fit'}
    batches=[]
    for step in range(steps):
        pair=rng.choice(pools[classes[step%len(classes)]])
        batches.append([pair['yes'],pair['no'],by_event[pair['yes']['event_uid']]])
    return batches


@torch.no_grad()
def evaluate(model,teacher,tokenizer,tasks,events,device,mode='adapted',swapped=None):
    if model is not None: model.eval()
    rows=[]
    for i,task in enumerate(tasks):
        event=events[task['event_uid']]; query=task['query']; context=event['tokens'][None].to(device)
        if mode=='text':
            query=f'Observed event: {ACTIVITIES[event["label"]]}.\n'+query; context=context[:,:0]
        elif mode=='none': context=context[:,:0]
        elif mode=='swapped': context=events[swapped[task['event_uid']]]['tokens'][None].to(device)
        if mode in ('adapted','swapped'):
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'): context=model(context)
        answer=generate_from_memory(teacher,tokenizer,context,query,max_new_tokens=24)
        rows.append(dict(uid=task['uid'],kind=task['kind'],label=task['label'],pair_id=task.get('pair_id'),
            event_uid=task['event_uid'],query=task['query'],reference=task['answer'],generated=answer,
            actual_event_uid=swapped[task['event_uid']] if mode=='swapped' else task['event_uid'],
            correct=grade_answer(answer,task['answer'],'presence' if task['kind']=='presence' else 'first')))
        if (i+1)%50==0: print(json.dumps(dict(phase='evaluate',mode=mode,done=i+1,total=len(tasks))),flush=True)
    return dict(mode=mode,rows=rows,metrics=score_predictions(rows))


@torch.no_grad()
def evaluate_swapped(model,teacher,tokenizer,tasks,events,device):
    # Per-task opposite event: one event may be a negative for several questions.
    rows=[]
    for pair in tasks['pairs']:
        if pair['cohort']!='dev': continue
        for task,other in ((pair['yes'],pair['no']),(pair['no'],pair['yes'])):
            context=events[other['event_uid']]['tokens'][None].to(device)
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'): context=model(context)
            generated=generate_from_memory(teacher,tokenizer,context,task['query'],max_new_tokens=24)
            flipped='No' if task['answer']=='Yes' else 'Yes'
            rows.append(dict(uid=task['uid'],kind='presence',label=other['label'],pair_id=task['pair_id'],
                query=task['query'],event_uid=task['event_uid'],actual_event_uid=other['event_uid'],
                reference=task['answer'],evidence_reference=flipped,generated=generated,
                correct=grade_answer(generated,task['answer'],'presence'),
                evidence_correct=grade_answer(generated,flipped,'presence')))
    flipped=[dict(r,reference=r['evidence_reference']) for r in rows]
    return dict(rows=rows,against_original_reference=score_predictions(rows),
                against_swapped_evidence=score_predictions(flipped))


def run(settings,config_path):
    output=Path(settings['output_dir'])
    if output.exists(): raise FileExistsError('refuse existing single-event QA output')
    output.mkdir(parents=True); started=time.time()
    status=dict(status='running',started_at=started,settings=settings,config_sha256=sha(config_path),
                source_code_sha256=sha(__file__),formal_go=False)
    code_paths=[Path(__file__),Path(__file__).parent/'models/qa_adapter.py',
                Path(__file__).parent/'parrad_stage4_qa.py',Path(__file__).parent/'parrad_stage4_runner.py',
                Path(__file__).parent/'stages/parrad_stage4.py']
    status['code_sha256']={str(p):sha(p) for p in code_paths}
    atomic_json(output/'status.json',status)
    try:
        torch.manual_seed(settings['seed']); torch.set_num_threads(1)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        runtime,tokenizer,frontend=load_frontend(settings,device)
        teacher=runtime.controller.event_former.adapter; teacher.eval()
        if any(p.requires_grad for p in teacher.parameters()): raise ValueError('teacher not frozen')
        root=Path(settings['source_output']); manifest=json.loads((root/'cache-manifest.json').read_text())
        provenance={name:sha(settings[name]) for name in ('frontend_config','stage3_checkpoint','threshold_artifact')}
        provenance.update(stage1_checkpoint=sha(frontend['stage']['stage1_checkpoint']),
            stage2_checkpoint=sha(frontend['stage3']['stage2_checkpoint']),
            event_former_code=sha(Path(__file__).parent/'models/event_former.py'),
            projection_code=sha(Path(__file__).parent/'models/semantic.py'))
        validate_provenance(manifest['provenance'],provenance)
        if sha(root/'events.pt')!=settings['cache_sha256']: raise ValueError('source cache changed')
        if sha(settings['protected_compressor'])!=settings['compressor_sha256']: raise ValueError('compressor changed')
        payload=torch.load(root/'events.pt',map_location='cpu',weights_only=False)
        digest=upstream_digest(runtime)
        if payload['upstream_digest']!=digest: raise ValueError('cached frontend changed')
        events={e['uid']:e for e in payload['events']}
        for e in events.values():
            if not e['recording'].startswith('subject_02__') or not e['recording'].endswith('__TI_xWR14xx'):
                raise ValueError('source outside approved subject/sensor')
            session=e['recording'].split('__')[1]
            if session not in ('set000','set001','set002') or e['cohort']!=('dev' if session=='set002' else 'fit'):
                raise ValueError('source session/cohort changed')
        tasks=build_single_tasks(payload['events'],settings['seed']); atomic_json(output/'tasks.json',tasks)
        dev=dev_tasks(tasks)
        print(json.dumps(dict(phase='tasks_ready',dev=len(dev),fit_events=sum(e['cohort']=='fit' for e in events.values()))),flush=True)
        baselines={}
        for mode in ('raw','text','none'):
            print(json.dumps(dict(phase='baseline',mode=mode)),flush=True)
            baselines[mode]=evaluate(None,teacher,tokenizer,dev,events,device,mode)
            atomic_json(output/f'baseline-{mode}.json',baselines[mode])
        model=EventQAAdapter(**settings['adapter']).to(device)
        params=list(model.parameters()); optimizer=torch.optim.AdamW(params,lr=settings['learning_rate'],weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=settings['amp_initial_scale'],enabled=device.type=='cuda')
        batches=training_batches(tasks,settings['steps'],settings['seed'])
        print(json.dumps(dict(phase='train_start',parameters=sum(p.numel() for p in params))),flush=True)
        source=dict(cache_sha256=settings['cache_sha256'],compressor_sha256=settings['compressor_sha256'],
                    frozen_upstream_digest=digest,provenance=provenance,task_sha256=sha(output/'tasks.json'))
        for step,batch in enumerate(batches,1):
            identification=batch[2]
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                loss=batch_loss(model,teacher,tokenizer,batch,events,device)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite single-event loss')
            old_scale=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(params,1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient; update not counted')
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()<old_scale: raise RuntimeError('skipped optimizer update')
            row=dict(step=step,pair_uid=batch[0]['pair_id'],activity=identification['label'],identify_event=identification['event_uid'],
                     loss=float(loss.detach()),gradient_norm=float(norm),yes=1,no=1,identify=1)
            with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            if step%25==0: print(json.dumps(dict(phase='train',**row)),flush=True)
            if step in (100,250,settings['steps']):
                torch.save(dict(step=step,model_settings=model.settings,model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict(),settings=settings,source=source,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),
                    output/f'qa-adapter-step{step}.pt')
        after=evaluate(model,teacher,tokenizer,dev,events,device)
        atomic_json(output/'dev-after.json',after)
        swapped=evaluate_swapped(model,teacher,tokenizer,tasks,events,device)
        atomic_json(output/'dev-swapped.json',swapped)
        if upstream_digest(runtime)!=digest: raise RuntimeError('original frozen model changed')
        if sha(root/'events.pt')!=settings['cache_sha256'] or sha(settings['protected_compressor'])!=settings['compressor_sha256']:
            raise RuntimeError('protected source changed during QA fitting')
        # Original event tokens and keys remain byte-identical in memory too.
        original=torch.load(root/'events.pt',map_location='cpu',weights_only=False)
        if not all(torch.equal(a[name],b[name]) for a,b in zip(original['events'],payload['events']) for name in ('tokens','key')):
            raise RuntimeError('original event/retrieval tensor modified')
        checks=[]
        for step in (100,250,settings['steps']):
            path=output/f'qa-adapter-step{step}.pt'; saved=torch.load(path,map_location='cpu',weights_only=False)
            actual=sorted({int(s['step']) for s in saved['optimizer_state']['state'].values()})
            finite=all(bool(torch.isfinite(t).all()) for t in saved['model_state'].values())
            if actual!=[step] or not finite or saved['source']['task_sha256']!=sha(output/'tasks.json'):
                raise RuntimeError('checkpoint readback failed')
            checks.append(dict(step=step,actual_steps=actual,finite=finite,sha256=sha(path)))
        loaded=EventQAAdapter(**saved['model_settings']); loaded.load_state_dict(saved['model_state'])
        if not all(torch.equal(v.cpu(),loaded.state_dict()[k]) for k,v in model.state_dict().items()):
            raise RuntimeError('final checkpoint differs from trained model')
        torch.save(dict(model_settings=model.settings,model_state=saved['model_state'],step=settings['steps'],source=source),
                   output/'qa-adapter-inference.pt')
        blind=baselines['none']['metrics']['presence_balanced_accuracy']; metrics=after['metrics']
        gates=dict(identify_macro=metrics['identify_macro_recall']>=.8,
                   presence_balanced=metrics['presence_balanced_accuracy']>=.8,
                   paired=metrics['pair_both_correct']>=.7,
                   memory_gain=metrics['presence_balanced_accuracy']-blind>=.2)
        atomic_json(output/'result.json',dict(updates=settings['steps'],parameters=sum(p.numel() for p in params),
            baselines={k:v['metrics'] for k,v in baselines.items()},after=metrics,
            swapped_original=swapped['against_original_reference'],swapped_evidence=swapped['against_swapped_evidence'],
            proposed_development_gates=gates,development_pass=all(gates.values()),formal_go=False,
            source=source,checkpoints=checks,frozen_upstream_unchanged=True,event_tokens_keys_unchanged=True,
            compressor_unchanged=True,inference_sha256=sha(output/'qa-adapter-inference.pt')))
        status.update(status='complete',duration_s=time.time()-started)
    except Exception as error:
        status.update(status='failed',error=repr(error),duration_s=time.time()-started)
        atomic_json(output/'status.json',status); raise
    atomic_json(output/'status.json',status); print(json.dumps(dict(phase='complete',duration_s=status['duration_s'])),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv); run(yaml.safe_load(args.config.read_text())['stage4'],args.config)


if __name__=='__main__': main()
