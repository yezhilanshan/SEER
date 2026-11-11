"""Matched raw/numbered event prefixes; only the QA adapter learns."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time
import torch
import yaml
from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.parrad_event_boundary_tasks import build_boundary_tasks, boundary_batches, score_boundary
from continuous_mmwave.parrad_stage4_qa import ACTIVITIES, grade_answer, qa_loss
from continuous_mmwave.parrad_short_context_qa import group_digest
from continuous_mmwave.parrad_stage4_runner import atomic_json, sha, load_frontend, upstream_digest, validate_provenance
from continuous_mmwave.stages.parrad_stage4 import chat_supervision, generate_from_memory


def boundary_context(model,teacher,tokenizer,task,events,device,marked):
    raw=torch.cat([events[uid]['tokens'].to(device) for uid in task['source_ids']])[None]
    if len(task['source_ids'])!=task['size'] or raw.shape[1]!=8*task['size']:
        raise ValueError('expected complete eight-token events')
    context=model(raw)
    if not marked or task['size']==1: return context
    if any(p.requires_grad for p in teacher.parameters()): raise ValueError('marker teacher must be frozen')
    pieces=[]
    for index in range(task['size']):
        ids=tokenizer(f'Event {index+1}:\n',add_special_tokens=False,return_tensors='pt')['input_ids'].to(device)
        with torch.no_grad(): marker=teacher.embed_input_ids(ids).to(context.dtype)
        pieces.extend((marker,context[:,index*8:(index+1)*8]))
    return torch.cat(pieces,dim=1)


def boundary_loss(model,teacher,tokenizer,batch,events,device,marked):
    losses=[]
    for task in batch:
        context=boundary_context(model,teacher,tokenizer,task,events,device,marked)
        ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
        losses.append(qa_loss(teacher,context,ids,attention,target,context.new_zeros(()),0.)['answer_ce'])
    return torch.stack(losses).mean()


def development_gates(before,after):
    a=before['1']; b=after['1']
    gates=dict(single_retained=all(b[key]>=max(floor,a[key]-.03) for key,floor in
        (('identify_macro_recall',.8),('presence_balanced_accuracy',.8),('pair_both_correct',.7))))
    for size in ('2','4'):
        m=after[size]
        gates.update({f'{size}_presence':m['presence_balanced_accuracy']>=.8,
            f'{size}_pair':m['pair_both_correct']>=.7,
            f'{size}_first':m['first']['accuracy']>=.8,f'{size}_last':m['last']['accuracy']>=.8})
    for position,value in after['4']['positions'].items():
        gates[f'4_position_{position}']=value is not None and value['accuracy']>=.8
    gates['4_sequence']=after['4']['sequence']['accuracy']>=.5
    return gates


@torch.no_grad()
def evaluate_boundary(model,teacher,tokenizer,tasks,events,device,marked=False,mode='raw',swaps=None):
    model.eval(); rows=[]
    for index,task in enumerate(tasks):
        actual=dict(task)
        if mode=='swapped': actual['source_ids']=swaps[task['uid']]['source_ids']
        query=task['query']
        if mode in ('none','text'):
            context=torch.empty(1,0,model.settings['dim'],device=device)
            if mode=='text':
                evidence='\n'.join(f'Event {i}: {ACTIVITIES[events[uid]["label"]]}' for i,uid in enumerate(actual['source_ids'],1))
                query='The following correct activity text is a diagnostic oracle, not radar input:\n'+evidence+'\n'+query
        else:
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                context=boundary_context(model,teacher,tokenizer,actual,events,device,marked)
        answer=generate_from_memory(teacher,tokenizer,context,query,max_new_tokens=64 if task['kind']=='sequence' else 24)
        row=dict(uid=task['uid'],kind=task['kind'],size=task['size'],label=task['label'],position=task.get('position'),
            pair_id=task.get('pair_id'),query=task['query'],source_ids=task['source_ids'],actual_source_ids=actual['source_ids'],
            reference=task['answer'],generated=answer,context_tokens=context.shape[1],marked=marked and task['size']>1 and mode not in ('none','text'),
            correct=grade_answer(answer,task['answer'],'summary' if task['kind']=='sequence' else 'presence' if task['kind']=='presence' else 'first'))
        if mode=='text': row['oracle_query']=query
        if mode=='swapped': row['evidence_reference']='No' if task['answer']=='Yes' else 'Yes'
        rows.append(row)
        if (index+1)%100==0: print(json.dumps(dict(phase='evaluate',mode=mode,marked=marked,done=index+1,total=len(tasks))),flush=True)
    result=dict(mode=mode,marked=marked,rows=rows,metrics=score_boundary(rows))
    if mode=='swapped': result['evidence_metrics']=score_boundary([dict(r,reference=r['evidence_reference']) for r in rows])
    return result


def load_boundary_sources(settings,device):
    runtime,tokenizer,frontend=load_frontend(settings,device)
    for module in (runtime.radar_encoder,runtime.controller.event_former,runtime.controller.semantic_projection,
                   runtime.controller.local_change,runtime.controller.radar_transition,runtime.controller.semantic_difference,runtime.controller.verify):
        module.eval(); module.requires_grad_(False)
    root=Path(settings['source_output'])
    protected={root/'events.pt':settings['cache_sha256'],Path(settings['init_qa']):settings['init_qa_sha256'],
        Path(settings['protected_compressor']):settings['compressor_sha256']}
    if any(sha(p)!=h for p,h in protected.items()): raise ValueError('protected source SHA mismatch')
    provenance={name:sha(settings[name]) for name in ('frontend_config','stage3_checkpoint','threshold_artifact')}
    provenance.update(stage1_checkpoint=sha(frontend['stage']['stage1_checkpoint']),stage2_checkpoint=sha(frontend['stage3']['stage2_checkpoint']),
        event_former_code=sha(Path(__file__).parent/'models/event_former.py'),projection_code=sha(Path(__file__).parent/'models/semantic.py'))
    validate_provenance(json.loads((root/'cache-manifest.json').read_text())['provenance'],provenance)
    payload=torch.load(root/'events.pt',map_location='cpu',weights_only=False); digest=upstream_digest(runtime)
    if payload['upstream_digest']!=digest or len(payload['events'])!=295: raise ValueError('cache teacher/event count mismatch')
    for e in payload['events']:
        pieces=e['recording'].split('__')
        if len(pieces)!=3 or pieces[0]!='subject_02' or pieces[2]!='TI_xWR14xx' or pieces[1] not in ('set000','set001','set002'):
            raise ValueError('outside approved TRAIN sources')
        if e['cohort']!=('dev' if pieces[1]=='set002' else 'fit') or tuple(e['tokens'].shape)!=(8,896):
            raise ValueError('invalid cohort or event shape')
    initial=torch.load(settings['init_qa'],map_location='cpu',weights_only=False)
    if initial['step']+initial.get('init_step',0)!=1000 or initial['source']['protected_sha256'][str(root/'events.pt')]!=settings['cache_sha256']:
        raise ValueError('initial QA cumulative steps/cache mismatch')
    if initial['source']['frozen_upstream_digest']!=digest: raise ValueError('initial QA teacher mismatch')
    return runtime,tokenizer,initial,payload,protected,provenance,digest


def checkpoint_readback(folder,model,source):
    checks=[]
    for step in (100,250,500):
        path=folder/f'qa-adapter-step{step}.pt'; saved=torch.load(path,map_location='cpu',weights_only=False)
        states=saved['optimizer_state']['state']; actual=sorted({int(v['step']) for v in states.values()})
        if actual!=[step] or saved['source']!=source or not all(bool(torch.isfinite(v).all()) for v in saved['model_state'].values()):
            raise RuntimeError('checkpoint readback failed')
        if not all(bool(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all()) for v in states.values()):
            raise RuntimeError('nonfinite optimizer moments')
        checks.append(dict(step=step,actual_steps=actual,optimizer_parameters=len(states),sha256=sha(path)))
    if not all(torch.equal(v.cpu(),saved['model_state'][k]) for k,v in model.state_dict().items()): raise RuntimeError('final saved mismatch')
    torch.save(dict(step=500,init_step=1000,marked=saved['marked'],model_settings=model.settings,
        model_state=saved['model_state'],source=source),folder/'qa-adapter-inference.pt')
    exported=torch.load(folder/'qa-adapter-inference.pt',map_location='cpu',weights_only=False)
    if set(exported['model_state'])!=set(saved['model_state']) or not all(torch.equal(v,exported['model_state'][k]) for k,v in saved['model_state'].items()):
        raise RuntimeError('inference export mismatch')
    return checks


def run_arm(name,marked,folder,initial,teacher,tokenizer,taskset,batches,events,device,settings,source,before,blind):
    torch.manual_seed(settings['seed']); model=EventQAAdapter(**initial['model_settings']).to(device)
    model.load_state_dict(initial['model_state'])
    atomic_json(folder/'arm-status.json',dict(status='training',arm=name,marked=marked))
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=settings['amp_initial_scale'],enabled=device.type=='cuda')
    for step,batch in enumerate(batches,1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            loss=boundary_loss(model,teacher,tokenizer,batch,events,device,marked)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
        scale=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient; no update counted')
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale()<scale: raise RuntimeError('skipped optimizer update')
        row=dict(step=step,size=batch[0]['size'],role=batch[2]['kind'],position=batch[2].get('position'),
            task_uids=[t['uid'] for t in batch],yes=1,no=1,loss=float(loss.detach()),gradient_norm=float(norm))
        with (folder/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if step%25==0: print(json.dumps(dict(phase='train',arm=name,**row)),flush=True)
        if step in (100,250,500):
            torch.save(dict(step=step,init_step=1000,marked=marked,model_settings=model.settings,model_state=model.state_dict(),
                optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict(),source=source,settings=settings,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),folder/f'qa-adapter-step{step}.pt')
    atomic_json(folder/'arm-status.json',dict(status='evaluating',arm=name,marked=marked,updates=500))
    dev=[t for t in taskset['tasks'] if t['cohort']=='dev']
    after=evaluate_boundary(model,teacher,tokenizer,dev,events,device,marked); atomic_json(folder/'dev-after.json',after)
    swaps={}
    for pair in taskset['pairs']:
        if pair['cohort']=='dev':
            a,b=[taskset['by_uid'][uid] for uid in pair['task_ids']]; swaps[a['uid']]=b; swaps[b['uid']]=a
    swapped=evaluate_boundary(model,teacher,tokenizer,[t for t in dev if t['kind']=='presence'],events,device,marked,'swapped',swaps)
    atomic_json(folder/'dev-swapped.json',swapped)
    checks=checkpoint_readback(folder,model,source)
    gates=development_gates(before['metrics'],after['metrics'])
    result=dict(updates=500,total_qa_updates=1500,marked=marked,source=source,
        size_updates=dict(Counter(b[0]['size'] for b in batches)),before=before['metrics'],after=after['metrics'],none=blind['metrics'],
        gates=gates,development_pass=all(gates.values()),formal_go=False,checkpoints=checks,inference_sha256=sha(folder/'qa-adapter-inference.pt'))
    atomic_json(folder/'result.json',result)
    atomic_json(folder/'arm-status.json',dict(status='complete',arm=name,marked=marked,updates=500))
    return result


def run(settings,config_path):
    output=Path(settings['output_dir'])
    if output.exists(): raise FileExistsError('refuse existing boundary output')
    output.mkdir(parents=True); started=time.time()
    modules=['parrad_event_boundary_qa.py','parrad_event_boundary_tasks.py','parrad_short_context_qa.py','parrad_short_context_tasks.py',
        'parrad_single_event_qa.py','parrad_stage4_qa.py','parrad_stage4_runner.py','models/qa_adapter.py','stages/parrad_stage4.py']
    status=dict(status='running',phase='load',started_at=started,settings=settings,config_sha256=sha(config_path),
        code_sha256={p:sha(Path(__file__).parent/p) for p in modules},formal_go=False)
    atomic_json(output/'status.json',status)
    try:
        if settings['steps_per_arm']!=500: raise ValueError('locked budget500per arm')
        torch.manual_seed(settings['seed']); torch.set_num_threads(1)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        runtime,tokenizer,initial,payload,protected,provenance,digest=load_boundary_sources(settings,device)
        teacher=runtime.controller.event_former.adapter; events={e['uid']:e for e in payload['events']}
        taskset=build_boundary_tasks(payload['events'],settings['seed'])
        for group in taskset['groups']: group['source_sha256']=group_digest(events,group['source_ids'])
        atomic_json(output/'tasks.json',{k:v for k,v in taskset.items() if k!='by_uid'})
        batches=boundary_batches(taskset,settings['seed'])
        atomic_json(output/'batches.json',[[t['uid'] for t in b] for b in batches])
        dev=[t for t in taskset['tasks'] if t['cohort']=='dev']; assert len(dev)==807 and len(batches)==500
        source=dict(provenance=provenance,protected_sha256={str(p):h for p,h in protected.items()},frozen_upstream_digest=digest,
            task_sha256=sha(output/'tasks.json'),batch_sha256=sha(output/'batches.json'),init_total_updates=1000)
        print(json.dumps(dict(phase='tasks_ready',dev=len(dev),fit=sum(t['cohort']=='fit' for t in taskset['tasks']))),flush=True)
        model=EventQAAdapter(**initial['model_settings']).to(device); model.load_state_dict(initial['model_state'])
        before={}
        for name,marked in (('A-raw',False),('B-numbered',True)):
            folder=output/name; folder.mkdir()
            status.update(phase='baseline',arm=name); atomic_json(output/'status.json',status)
            before[name]=evaluate_boundary(model,teacher,tokenizer,dev,events,device,marked)
            atomic_json(folder/'dev-before.json',before[name])
        singles=[{r['uid']:(r['query'],r['reference'],r['generated']) for r in value['rows'] if r['size']==1} for value in before.values()]
        if singles[0]!=singles[1]: raise RuntimeError('initial single renderer mismatch')
        previous=json.loads((Path(settings['previous_output'])/'dev-after.json').read_text())['rows']
        current={r['uid']:r for r in before['A-raw']['rows']}
        if len(previous)!=613 or any(any(r[k]!=current[r['uid']][k] for k in ('query','reference','generated','source_ids')) for r in previous):
            raise RuntimeError('previous raw QA baseline did not reproduce')
        status.update(phase='diagnostic'); atomic_json(output/'status.json',status)
        oracle=evaluate_boundary(model,teacher,tokenizer,[t for t in dev if t['size']>1],events,device,mode='text')
        atomic_json(output/'dev-text-oracle.json',oracle)
        blind=evaluate_boundary(model,teacher,tokenizer,dev,events,device,mode='none'); atomic_json(output/'dev-none.json',blind)
        del model; results={}
        for name,marked in (('A-raw',False),('B-numbered',True)):
            status.update(phase='train-evaluate',arm=name); atomic_json(output/'status.json',status)
            results[name]=run_arm(name,marked,output/name,initial,teacher,tokenizer,taskset,batches,events,device,settings,source,before[name],blind)
            if upstream_digest(runtime)!=digest or any(sha(p)!=h for p,h in protected.items()): raise RuntimeError('frozen sources changed')
        fresh=torch.load(Path(settings['source_output'])/'events.pt',map_location='cpu',weights_only=False)
        if len(fresh['events'])!=len(payload['events']) or not all(torch.equal(a[n],b[n]) for a,b in zip(fresh['events'],payload['events']) for n in ('tokens','key')):
            raise RuntimeError('original event tokens/keys changed')
        a={r['uid']:r for r in json.loads((output/'A-raw/dev-after.json').read_text())['rows']}
        b={r['uid']:r for r in json.loads((output/'B-numbered/dev-after.json').read_text())['rows']}
        if set(a)!=set(b): raise RuntimeError('unmatched final tasks')
        paired=[]
        for uid,x in a.items():
            y=b[uid]
            if any(x[k]!=y[k] for k in ('query','reference','source_ids','actual_source_ids','kind','position')): raise RuntimeError('unmatched final inputs')
            paired.append(dict(uid=uid,kind=x['kind'],size=x['size'],position=x['position'],source_ids=x['source_ids'],
                source_sha256=group_digest(events,x['source_ids']),raw_generated=x['generated'],numbered_generated=y['generated'],
                raw_correct=x['correct'],numbered_correct=y['correct']))
        atomic_json(output/'paired-outcomes.json',dict(rows=paired))
        atomic_json(output/'result.json',dict(arms=results,source=source,initial_single_exact=True,previous613_exact=True,
            frozen_upstream_unchanged=True,event_tokens_keys_unchanged=True,protected_files_unchanged=True,formal_go=False))
        status.update(status='complete',phase='complete',duration_s=time.time()-started)
    except Exception as error:
        status.update(status='failed',error=repr(error),duration_s=time.time()-started); atomic_json(output/'status.json',status); raise
    atomic_json(output/'status.json',status); print(json.dumps(dict(phase='complete',duration_s=status['duration_s'])),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv); run(yaml.safe_load(args.config.read_text())['stage4'],args.config)


if __name__=='__main__': main()
