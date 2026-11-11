"""Fixed-budget none/order/time event QA with a frozen original QA branch."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path
import random
import time

import torch
import yaml

from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.models.temporal_qa import TemporalQAFusion
from continuous_mmwave.parrad_qa_reader import QAReaders, reader_for, load_multi_reader, reader_digest, retrieval_snapshot
from continuous_mmwave.parrad_event_boundary_qa import load_boundary_sources,development_gates
from continuous_mmwave.parrad_event_boundary_tasks import build_boundary_tasks,score_boundary
from continuous_mmwave.parrad_short_context_qa import group_digest
from continuous_mmwave.parrad_stage4_qa import ACTIVITIES,qa_loss,grade_answer
from continuous_mmwave.parrad_stage4_runner import atomic_json,sha,upstream_digest
from continuous_mmwave.stages.parrad_stage4 import chat_supervision,generate_from_memory


def temporal_query(task):
    if task['size']==1: return task['query']
    n=task['size']; prefix=f'The supplied memory contains exactly {n} events, in chronological order. '
    if task['kind']=='presence':
        if 'Does it include ' not in task['query']: raise ValueError('unexpected presence question')
        return prefix+'Does it include '+task['query'].split('Does it include ',1)[1]
    if task['kind']=='sequence':
        return prefix+f'List exactly {n} activity names, one for each event, in that same order. Keep repeated activities. Separate names with "; ". Output the list only.'
    position=task.get('position',1 if task['kind']=='first' else n)
    return prefix+f'What is the activity of event {position}? Output only its activity name.'


def text_oracle_query(task,events):
    evidence='\n'.join(f'{i}: {ACTIVITIES[events[u]["label"]]}' for i,u in enumerate(task['source_ids'],1))
    question=temporal_query(task).replace('The supplied memory contains', 'The numbered evidence above contains').replace('Does it include ', 'Does the numbered evidence above include ')
    if task['kind']=='sequence': instruction='Copy the activity names from the numbered evidence below in order. Keep every row, including repetitions. Do not add or replace activities.'
    elif task['kind']=='presence': instruction='Check whether the requested activity name appears in the numbered evidence below. Answer only Yes or No.'
    else: instruction='Copy the activity name from the requested evidence row below. Do not infer a different activity.'
    return instruction+'\nEvidence:\n'+evidence+'\n\nQuestion: '+question


def temporal_batches(taskset,seed):
    rng=random.Random(seed); pools={size:{} for size in (2,4)}
    for pair in taskset['pairs']:
        if pair['cohort']=='fit' and pair['size'] in pools: pools[pair['size']].setdefault(pair['activity'],[]).append(pair)
    roles={2:[('position',1),('position',2)],4:[('sequence',None),('position',1),('sequence',None),('position',2),('sequence',None),('position',3),('sequence',None),('position',4)]}
    content={(t['group_uid'],t['kind'],t.get('position')):t for t in taskset['tasks'] if t['cohort']=='fit' and t['kind']!='presence'}
    counts=Counter(); classes_by_role=Counter(); batches=[]
    for size in [2,4]*250:
        kind,position=roles[size][counts[size]%len(roles[size])]; role=(size,kind,position)
        classes=sorted(pools[size])
        if not classes: raise ValueError('missing multi-event fit pool')
        activity=classes[classes_by_role[role]%len(classes)]
        pair=rng.choice(pools[size][activity])
        batch=[taskset['by_uid'][u] for u in pair['task_ids']]+[content[(pair['group_uid'],kind,position)]]
        batches.append([dict(t,query=temporal_query(t)) for t in batch])
        counts[size]+=1; classes_by_role[role]+=1
    return batches


def temporal_context(model,task,events,device,arm):
    source=[events[u] for u in task['source_ids']]
    if len(source)!=task['size'] or len({s['recording'] for s in source})!=1:
        raise ValueError('expected a complete group from one recording')
    raw=torch.cat([e['tokens'].to(device) for e in source])[None]
    starts=torch.tensor([[e['start'] for e in source]],device=device,dtype=torch.float64)
    ends=torch.tensor([[e['end'] for e in source]],device=device,dtype=torch.float64)
    return model(raw,starts,ends,arm=arm)


def temporal_loss(model,teacher,tokenizer,batch,events,device,arm):
    losses=[]
    for task in batch:
        context=temporal_context(model,task,events,device,arm)
        ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
        losses.append(qa_loss(reader_for(teacher,task),context,ids,attention,target,context.new_zeros(()),0.)['answer_ce'])
    return torch.stack(losses).mean()


def temporal_backward(model,teacher,tokenizer,batch,events,device,arm,scaler):
    if not batch: raise ValueError('empty temporal batch')
    value=0.
    for task in batch:
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            context=temporal_context(model,task,events,device,arm)
            ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
            loss=qa_loss(reader_for(teacher,task),context,ids,attention,target,context.new_zeros(()),0.)['answer_ce']/len(batch)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite temporal loss')
        value+=float(loss.detach()); scaler.scale(loss).backward()
    return value


def finite_optimizer_step(parameters,optimizer,scaler,backward,device,*,max_backoffs=12,retry_sink=None):
    cpu_rng=torch.get_rng_state()
    cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None
    for attempt in range(max_backoffs+1):
        if attempt:
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None: torch.cuda.set_rng_state_all(cuda_rng)
        optimizer.zero_grad(set_to_none=True); scale=scaler.get_scale()
        try: loss=backward()
        except Exception:
            optimizer.zero_grad(set_to_none=True); raise
        if not math.isfinite(loss):
            optimizer.zero_grad(set_to_none=True); raise RuntimeError('nonfinite temporal loss')
        scaler.unscale_(optimizer)
        if any(p.grad is None for p in parameters):
            optimizer.zero_grad(set_to_none=True); raise RuntimeError('missing temporal parameter gradients')
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if torch.isfinite(norm):
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()<scale: raise RuntimeError('unexpected skipped temporal update')
            return dict(loss=loss,gradient_norm=float(norm),loss_scale=scale,gradient_backoffs=attempt)
        optimizer.zero_grad(set_to_none=True)
        if retry_sink is not None: retry_sink(dict(attempt=attempt+1,loss_scale=scale,reason='nonfinite-gradient'))
        if not scaler.is_enabled() or attempt==max_backoffs:
            raise RuntimeError('nonfinite temporal gradient after bounded scaling retries')
        # No optimizer step or sampler advance: retry exactly the same batch.
        scaler.update(new_scale=scale*.5)


def oracle_gate(metrics):
    gates={'sequence':metrics['4']['sequence'] is not None and metrics['4']['sequence']['accuracy']>=.95}
    for size in ('2','4'):
        for polarity in ('Yes','No'):
            value=metrics[size]['presence_classes'].get(polarity)
            gates[f'{size}_{polarity}']=value is not None and value['accuracy']>=.95
        for position in range(1,int(size)+1):
            value=metrics[size]['positions'].get(str(position))
            gates[f'{size}_position_{position}']=value is not None and value['accuracy']>=.95
    return gates


@torch.no_grad()
def evaluate(model,teacher,tokenizer,tasks,events,device,arm,mode='raw',swaps=None):
    model.eval(); rows=[]
    for index,task in enumerate(tasks):
        actual=dict(task)
        if mode=='swapped': actual['source_ids']=swaps[task['uid']]['source_ids']
        query=temporal_query(task)
        if mode in ('text','none'):
            context=torch.empty(1,0,model.settings['dim'],device=device)
            if mode=='text': query=text_oracle_query(actual,events)
        else:
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                context=temporal_context(model,actual,events,device,arm)
        answer=generate_from_memory(reader_for(teacher,task),tokenizer,context,query,max_new_tokens=96 if task['kind']=='sequence' else 24)
        row=dict(uid=task['uid'],cohort=task['cohort'],kind=task['kind'],size=task['size'],label=task['label'],position=task.get('position'),
                 pair_id=task.get('pair_id'),query=temporal_query(task),source_ids=task['source_ids'],
                 actual_source_ids=actual['source_ids'],reference=task['answer'],generated=answer,
                 context_tokens=context.shape[1],arm=arm,mode=mode,
                 reader='original' if not isinstance(teacher,QAReaders) or task['size']==1 else 'text-qa-500',
                 correct=grade_answer(answer,task['answer'],'summary' if task['kind']=='sequence' else 'presence' if task['kind']=='presence' else 'first'))
        if mode=='text': row['oracle_query']=query
        if mode=='swapped': row['evidence_reference']='No' if task['answer']=='Yes' else 'Yes'
        rows.append(row)
        if (index+1)%100==0: print(json.dumps(dict(phase='evaluate',arm=arm,mode=mode,done=index+1,total=len(tasks))),flush=True)
    result=dict(mode=mode,arm=arm,rows=rows,metrics=score_boundary(rows))
    if mode=='swapped': result['evidence_metrics']=score_boundary([dict(r,reference=r['evidence_reference']) for r in rows])
    return result


def new_model(initial,settings,device):
    torch.manual_seed(settings['seed'])
    base=EventQAAdapter(**initial['model_settings']); base.load_state_dict(initial['model_state'])
    return TemporalQAFusion(base,**settings['fusion']).to(device)


def readback(folder,model,source,arm):
    checks=[]
    base_state={f'base.{k}':v.detach().cpu() for k,v in model.base.state_dict().items()}
    for step in (100,250,500):
        path=folder/f'fusion-step{step}.pt'; saved=torch.load(path,map_location='cpu',weights_only=False)
        states=saved['optimizer_state']['state']
        actual=sorted({int(v['step']) for v in states.values()})
        if saved['step']!=step or actual!=[step] or len(states)!=6 or saved['source']!=source or saved['arm']!=arm:
            raise RuntimeError('temporal optimizer/source readback failed')
        if not all(torch.equal(saved['model_state'][k],v) for k,v in base_state.items()): raise RuntimeError('base QA changed')
        if not all(torch.isfinite(v).all() for v in saved['model_state'].values()): raise RuntimeError('nonfinite saved weights')
        if not all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in states.values()):
            raise RuntimeError('nonfinite saved moments')
        checks.append(dict(step=step,actual_steps=actual,optimizer_parameters=len(states),base_unchanged=True,sha256=sha(path)))
    if set(saved['model_state'])!=set(model.state_dict()) or not all(torch.equal(v.cpu(),saved['model_state'][k]) for k,v in model.state_dict().items()):
        raise RuntimeError('saved final weights differ')
    exported={k:saved[k] for k in ('step','arm','model_settings','base_settings','model_state','source')}
    torch.save(exported,folder/'fusion-inference.pt')
    reread=torch.load(folder/'fusion-inference.pt',map_location='cpu',weights_only=False)
    if not all(torch.equal(v,reread['model_state'][k]) for k,v in saved['model_state'].items()): raise RuntimeError('export mismatch')
    return checks


def run_arm(arm,folder,initial,teacher,tokenizer,taskset,batches,events,device,settings,source,before,blind):
    folder.mkdir(); model=new_model(initial,settings,device)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=settings['amp_initial_scale'],enabled=device.type=='cuda')
    atomic_json(folder/'arm-status.json',dict(status='training',arm=arm,trainable_parameters=sum(p.numel() for p in parameters)))
    for step,batch in enumerate(batches,1):
        model.train()
        def retry_sink(row):
            with (folder/'numerical-retries.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step,task_uids=[t['uid'] for t in batch],**row))+'\n')
        update=finite_optimizer_step(parameters,optimizer,scaler,
            lambda:temporal_backward(model,teacher,tokenizer,batch,events,device,arm,scaler),device,retry_sink=retry_sink)
        row=dict(step=step,size=batch[0]['size'],role=batch[2]['kind'],position=batch[2].get('position'),
                 task_uids=[t['uid'] for t in batch],yes=1,no=1,**update)
        with (folder/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
        if step%25==0: print(json.dumps(dict(phase='train',arm=arm,**row)),flush=True)
        if step in (100,250,500):
            torch.save(dict(step=step,arm=arm,model_settings=model.settings,base_settings=model.base.settings,
                       model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict(),
                       source=source,settings=settings,torch_rng=torch.get_rng_state(),
                       cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),folder/f'fusion-step{step}.pt')
    if not all(torch.equal(model.base.state_dict()[k].cpu(),v) for k,v in initial['model_state'].items()):
        raise RuntimeError('initial QA weights changed')
    atomic_json(folder/'arm-status.json',dict(status='evaluating',arm=arm,updates=500))
    dev=[t for t in taskset['tasks'] if t['cohort']=='dev']
    after=evaluate(model,teacher,tokenizer,dev,events,device,arm); atomic_json(folder/'dev-after.json',after)
    single=lambda result:{r['uid']:(r['query'],r['reference'],r['generated']) for r in result['rows'] if r['size']==1}
    if single(before)!=single(after): raise RuntimeError('single-event predictions changed')
    swaps={}
    for pair in taskset['pairs']:
        if pair['cohort']=='dev':
            a,b=[taskset['by_uid'][u] for u in pair['task_ids']]; swaps[a['uid']]=b; swaps[b['uid']]=a
    swapped=evaluate(model,teacher,tokenizer,[t for t in dev if t['kind']=='presence'],events,device,arm,'swapped',swaps)
    atomic_json(folder/'dev-swapped.json',swapped)
    checks=readback(folder,model,source,arm)
    gates=development_gates(before['metrics'],after['metrics'])
    result=dict(arm=arm,updates=500,base_qa_updates=1000,new_fusion_updates=500,single_predictions_exact=True,
                trainable_parameters=sum(p.numel() for p in parameters),source=source,size_updates=dict(Counter(b[0]['size'] for b in batches)),
                before=before['metrics'],after=after['metrics'],none=blind['metrics'],gates=gates,
                development_pass=all(gates.values()),formal_go=False,checkpoints=checks,inference_sha256=sha(folder/'fusion-inference.pt'))
    atomic_json(folder/'result.json',result); atomic_json(folder/'arm-status.json',dict(status='complete',arm=arm,updates=500))
    return result


def run(settings,config_path):
    if settings['steps_per_arm']!=500 or settings['seed']!=42: raise ValueError('locked 500-step seed42 experiment')
    output=Path(settings['output_dir'])
    if output.exists(): raise FileExistsError('refuse duplicate temporal output')
    output.mkdir(parents=True); started=time.time()
    modules=['parrad_temporal_qa.py','models/temporal_qa.py','models/qa_adapter.py','parrad_event_boundary_qa.py',
             'parrad_event_boundary_tasks.py','parrad_short_context_qa.py','parrad_short_context_tasks.py',
             'parrad_single_event_qa.py','parrad_stage4_qa.py','parrad_stage4_runner.py','stages/parrad_stage4.py',
             'parrad_qa_reader.py','parrad_qa_text_decoder.py','models/event_former.py','parrad_qa_readback.py']
    status=dict(status='running',phase='load',settings=settings,started_at=started,config_sha256=sha(config_path),
                code_sha256={p:sha(Path(__file__).parent/p) for p in modules},formal_go=False)
    atomic_json(output/'status.json',status)
    try:
        torch.set_num_threads(1); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        runtime,tokenizer,initial,payload,protected,provenance,digest=load_boundary_sources(settings,device)
        teacher=runtime.controller.event_former.adapter; events={e['uid']:e for e in payload['events']}
        reader_source=None
        if 'qa_reader' in settings:
            multi,reader_source=load_multi_reader(settings,teacher,protected,digest,device)
            teacher=QAReaders(teacher,multi)
            protected[Path(settings['qa_reader'])]=settings['qa_reader_sha256']
        taskset=build_boundary_tasks(payload['events'],settings['seed'])
        for task in taskset['tasks']: task['query']=temporal_query(task)
        for group in taskset['groups']: group['source_sha256']=group_digest(events,group['source_ids'])
        atomic_json(output/'tasks.json',{k:v for k,v in taskset.items() if k!='by_uid'})
        batches=temporal_batches(taskset,settings['seed']); atomic_json(output/'batches.json',[[t['uid'] for t in b] for b in batches])
        source=dict(provenance=provenance,protected_sha256={str(p):h for p,h in protected.items()},frozen_upstream_digest=digest,
                    task_sha256=sha(output/'tasks.json'),batch_sha256=sha(output/'batches.json'),base_qa_updates=1000)
        if reader_source is not None: source['multi_reader']=reader_source
        source['numerical_policy']=dict(initial_scale=settings.get('amp_initial_scale',256),backoff=.5,max_backoffs=12,
            retry_same_batch=True,restore_rng=True,advance_only_after_optimizer_step=True)
        atomic_json(output/'source.json',source)
        model=new_model(initial,settings,device)
        # Exact zero-residual parity across all three arms, before any optimization.
        for group in taskset['groups']:
            task=dict(source_ids=group['source_ids'],size=group['size'])
            with torch.no_grad(),torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                outputs=[temporal_context(model,task,events,device,arm) for arm in ('none','order','time')]
                raw=torch.cat([events[u]['tokens'].to(device) for u in task['source_ids']])[None]
                base=model.base(raw)
            if not all(torch.equal(v,base) for v in outputs): raise RuntimeError('zero-residual parity failed')
        retrieval_before=None
        if settings.get('baseline_source'):
            from continuous_mmwave.parrad_qa_readback import restore_baseline
            status.update(phase='baseline-reuse'); atomic_json(output/'status.json',status)
            before,blind,retrieval_before,binding=restore_baseline(settings['baseline_source'],output,settings,taskset,events,source)
            if retrieval_snapshot(runtime,tokenizer,events,device)!=retrieval_before:
                raise RuntimeError('reused original retrieval baseline changed')
            source['baseline_reuse']=binding; atomic_json(output/'source.json',source)
        else:
            status.update(phase='text-preflight'); atomic_json(output/'status.json',status)
            fit=[t for t in taskset['tasks'] if t['cohort']=='fit' and t['size']>1]
            oracle=evaluate(model,teacher,tokenizer,fit,events,device,'none','text'); atomic_json(output/'fit-text-preflight.json',oracle)
            gate=oracle_gate(oracle['metrics']); atomic_json(output/'text-preflight-gates.json',dict(gates=gate,passed=all(gate.values()),selection='locked template; fit only'))
            if not all(gate.values()):
                status.update(status='blocked',phase='text-preflight-failed',reason='Frozen decoder did not pass locked pure-text gate; no training was started',actual_updates=0)
                return
            dev=[t for t in taskset['tasks'] if t['cohort']=='dev']
            oracle_dev=evaluate(model,teacher,tokenizer,[t for t in dev if t['size']>1],events,device,'none','text')
            atomic_json(output/'dev-text-oracle.json',oracle_dev)
            status.update(phase='baseline'); atomic_json(output/'status.json',status)
            before=evaluate(model,teacher,tokenizer,dev,events,device,'none'); atomic_json(output/'dev-before.json',before)
            if isinstance(teacher,QAReaders):
                old_multi=evaluate(model,teacher.single,tokenizer,[t for t in dev if t['size']>1],events,device,'none')
                atomic_json(output/'dev-original-reader.json',old_multi)
            blind=evaluate(model,teacher,tokenizer,dev,events,device,'none','none'); atomic_json(output/'dev-none.json',blind)
            if isinstance(teacher,QAReaders):
                retrieval_before=retrieval_snapshot(runtime,tokenizer,events,device)
                atomic_json(output/'retrieval-before.json',retrieval_before)
        previous=json.loads((Path(settings['previous_output'])/'dev-after.json').read_text())['rows']
        old={r['uid']:(r['query'],r['reference'],r['generated']) for r in previous if r['size']==1}
        current={r['uid']:(r['query'],r['reference'],r['generated']) for r in before['rows'] if r['size']==1}
        if old!=current or len(current)!=297: raise RuntimeError('previous single-event predictions not reproduced')
        results={}
        for arm in ('none','order','time'):
            status.update(phase='train-evaluate',arm=arm); atomic_json(output/'status.json',status)
            results[arm]=run_arm(arm,output/arm,initial,teacher,tokenizer,taskset,batches,events,device,settings,source,before,blind)
            if upstream_digest(runtime)!=digest or any(sha(p)!=h for p,h in protected.items()): raise RuntimeError('protected sources changed')
            if reader_source is not None and reader_digest(teacher.multi)!=reader_source['frozen_digest']:
                raise RuntimeError('frozen multi reader changed')
            if retrieval_before is not None:
                retrieval_after=retrieval_snapshot(runtime,tokenizer,events,device)
                atomic_json(output/arm/'retrieval-after.json',retrieval_after)
                if retrieval_after!=retrieval_before: raise RuntimeError('original retrieval scores/rankings changed')
        fresh=torch.load(Path(settings['source_output'])/'events.pt',map_location='cpu',weights_only=False)
        if not all(torch.equal(a[k],b[k]) for a,b in zip(payload['events'],fresh['events']) for k in ('tokens','key')):
            raise RuntimeError('original event tokens/keys changed')
        atomic_json(output/'result.json',dict(arms=results,source=source,zero_residual_parity=True,single_predictions_exact=True,
                    frozen_upstream_unchanged=True,event_tokens_keys_unchanged=True,formal_go=False))
        status.update(status='complete',phase='complete')
    except Exception as error:
        status.update(status='failed',reason=repr(error)); raise
    finally:
        status['duration_s']=time.time()-started; atomic_json(output/'status.json',status)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); args=parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['stage4'],args.config)


if __name__=='__main__': main()
