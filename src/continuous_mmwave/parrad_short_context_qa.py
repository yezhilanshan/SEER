"""Fixed500update short-context QA and frozen L1 matched evaluation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
import torch
import yaml

from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.models.memory import QueryCompressor
from continuous_mmwave.parrad_short_context_tasks import build_short_tasks,short_training_batches,score_short
from continuous_mmwave.parrad_stage4_qa import grade_answer,qa_loss
from continuous_mmwave.parrad_stage4_runner import sha,atomic_json,load_frontend,upstream_digest,validate_provenance,compress_example
from continuous_mmwave.stages.parrad_stage4 import chat_supervision,generate_from_memory


def short_context(model,task,events,device,compressor=None):
    source=[events[uid] for uid in task['source_ids']]
    if compressor is None: tokens=torch.cat([e['tokens'].to(device) for e in source])[None]
    else:
        if task['size']!=4 or any(p.requires_grad for p in compressor.parameters()):
            raise ValueError('matched compression requires frozen four-event compressor')
        with torch.no_grad(): tokens,_=compress_example(compressor,(1,source),device)
    return model(tokens)


def short_loss(model,teacher,tokenizer,batch,events,device):
    losses=[]
    for task in batch:
        context=short_context(model,task,events,device)
        ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
        losses.append(qa_loss(teacher,context,ids,attention,target,context.new_zeros(()),0.)['answer_ce'])
    return torch.stack(losses).mean()


def group_digest(events,source_ids):
    digest=hashlib.sha256()
    for uid in source_ids:
        event=events[uid]
        digest.update(json.dumps(dict(uid=uid,start=event['start'],end=event['end']),sort_keys=True).encode())
        tokens=event['tokens'].detach().cpu().contiguous()
        digest.update(f'{tuple(tokens.shape)}:{tokens.dtype}'.encode())
        digest.update(tokens.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_l1_match(raw_result,compressed_result,events):
    raw={r['uid']:r for r in raw_result['rows'] if r['size']==4}
    compressed={r['uid']:r for r in compressed_result['rows']}
    if set(raw)!=set(compressed) or not raw: raise ValueError('missing matched four-event tasks')
    rows=[]; transitions={}
    for uid,a in raw.items():
        b=compressed[uid]
        if any(a[key]!=b[key] for key in ('source_ids','actual_source_ids','query','reference','kind')):
            raise ValueError('L1 comparison source/query/reference mismatch')
        if a['context_tokens']!=32 or b['context_tokens']!=4: raise ValueError('L1 comparison token counts mismatch')
        outcome=('both_correct' if a['correct'] and b['correct'] else 'raw_only_correct' if a['correct']
                 else 'compressed_only_correct' if b['correct'] else 'both_wrong')
        counts=transitions.setdefault(a['kind'],dict(both_correct=0,raw_only_correct=0,compressed_only_correct=0,both_wrong=0))
        counts[outcome]+=1
        rows.append(dict(uid=uid,kind=a['kind'],source_ids=a['source_ids'],query=a['query'],reference=a['reference'],
            source_sha256=group_digest(events,a['source_ids']),raw_generated=a['generated'],compressed_generated=b['generated'],
            raw_correct=a['correct'],compressed_correct=b['correct']))
    return dict(rows=rows,transitions=transitions)


@torch.no_grad()
def evaluate_short(model,teacher,tokenizer,tasks,events,device,mode='raw',compressor=None,swaps=None):
    model.eval(); rows=[]
    for index,task in enumerate(tasks):
        actual=dict(task)
        if mode=='swapped': actual['source_ids']=swaps[task['uid']]['source_ids']
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            context=short_context(model,actual,events,device,compressor if mode=='compressed' else None)
        if mode=='none': context=context[:,:0]
        answer=generate_from_memory(teacher,tokenizer,context,task['query'],max_new_tokens=64 if task['kind']=='sequence' else 24)
        row=dict(uid=task['uid'],kind=task['kind'],size=task['size'],label=task['label'],pair_id=task.get('pair_id'),
            query=task['query'],source_ids=task['source_ids'],actual_source_ids=actual['source_ids'],
            reference=task['answer'],generated=answer,context_tokens=context.shape[1],
            correct=grade_answer(answer,task['answer'],'summary' if task['kind']=='sequence' else 'presence' if task['kind']=='presence' else 'first'))
        if mode=='swapped': row['evidence_reference']='No' if task['answer']=='Yes' else 'Yes'
        rows.append(row)
        if (index+1)%100==0: print(json.dumps(dict(phase='evaluate',mode=mode,done=index+1,total=len(tasks))),flush=True)
    result=dict(mode=mode,rows=rows,metrics=score_short(rows))
    if mode=='swapped': result['evidence_metrics']=score_short([dict(r,reference=r['evidence_reference']) for r in rows])
    return result


def load_sources(settings,device):
    runtime,tokenizer,frontend=load_frontend(settings,device)
    root=Path(settings['source_output']); manifest=json.loads((root/'cache-manifest.json').read_text())
    provenance={name:sha(settings[name]) for name in ('frontend_config','stage3_checkpoint','threshold_artifact')}
    provenance.update(stage1_checkpoint=sha(frontend['stage']['stage1_checkpoint']),stage2_checkpoint=sha(frontend['stage3']['stage2_checkpoint']),
        event_former_code=sha(Path(__file__).parent/'models/event_former.py'),projection_code=sha(Path(__file__).parent/'models/semantic.py'))
    validate_provenance(manifest['provenance'],provenance)
    protected={root/'events.pt':settings['cache_sha256'],Path(settings['init_qa']):settings['init_qa_sha256'],
               Path(settings['protected_compressor']):settings['compressor_sha256']}
    if any(sha(p)!=h for p,h in protected.items()): raise ValueError('protected source SHA mismatch')
    payload=torch.load(root/'events.pt',map_location='cpu',weights_only=False); digest=upstream_digest(runtime)
    if payload['upstream_digest']!=digest: raise ValueError('cached teacher changed')
    for e in payload['events']:
        pieces=e['recording'].split('__')
        if len(pieces)!=3 or pieces[0]!='subject_02' or pieces[2]!='TI_xWR14xx' or pieces[1] not in ('set000','set001','set002'):
            raise ValueError('outside approved TRAIN subject/sensor/recording')
        if e['cohort']!=('dev' if pieces[1]=='set002' else 'fit'): raise ValueError('cohort mismatch')
    initial=torch.load(settings['init_qa'],map_location='cpu',weights_only=False)
    if initial['step']!=500 or initial['source']['cache_sha256']!=settings['cache_sha256']:
        raise ValueError('QA initialization mismatch')
    model=EventQAAdapter(**initial['model_settings']).to(device); model.load_state_dict(initial['model_state'])
    cstate=torch.load(settings['protected_compressor'],map_location='cpu',weights_only=False)
    compressor=QueryCompressor(**cstate['model_settings']).to(device); compressor.load_state_dict(cstate['model_state']); compressor.eval()
    for p in compressor.parameters(): p.requires_grad_(False)
    if any(p.requires_grad for p in runtime.controller.event_former.adapter.parameters()): raise ValueError('LLM not frozen')
    return runtime,tokenizer,model,compressor,payload,protected,provenance,digest


def run(settings,config_path):
    output=Path(settings['output_dir'])
    if output.exists(): raise FileExistsError('refuse existing short QA output')
    output.mkdir(parents=True); started=time.time()
    code=['parrad_short_context_qa.py','parrad_short_context_tasks.py','parrad_single_event_qa.py','parrad_stage4_qa.py',
          'parrad_stage4_runner.py','models/qa_adapter.py','models/memory.py','stages/parrad_stage4.py']
    status=dict(status='running',started_at=started,settings=settings,config_sha256=sha(config_path),
                code_sha256={p:sha(Path(__file__).parent/p) for p in code},formal_go=False)
    atomic_json(output/'status.json',status)
    try:
        torch.manual_seed(settings['seed']); torch.set_num_threads(1)
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        runtime,tokenizer,model,compressor,payload,protected,provenance,digest=load_sources(settings,device)
        teacher=runtime.controller.event_former.adapter; teacher.eval(); events={e['uid']:e for e in payload['events']}
        taskset=build_short_tasks(payload['events'],settings['seed'])
        for group in taskset['groups']: group['source_sha256']=group_digest(events,group['source_ids'])
        atomic_json(output/'tasks.json',{k:v for k,v in taskset.items() if k!='by_uid'})
        dev=[t for t in taskset['tasks'] if t['cohort']=='dev']; fit_single=[t for t in taskset['tasks'] if t['cohort']=='fit' and t['size']==1]
        print(json.dumps(dict(phase='tasks_ready',dev=len(dev),fit_single=len(fit_single),groups=dict(Counter(f'{g["cohort"]}:{g["size"]}' for g in taskset['groups'])))),flush=True)
        for name,tasks in (('fit-single-before',fit_single),('dev-before',dev)):
            print(json.dumps(dict(phase=name)),flush=True)
            atomic_json(output/(name+'.json'),evaluate_short(model,teacher,tokenizer,tasks,events,device))
        before=json.loads((output/'dev-before.json').read_text())
        batches=short_training_batches(taskset,settings['seed']); assert len(batches)==settings['steps']==500
        optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=settings['amp_initial_scale'],enabled=device.type=='cuda')
        source=dict(provenance=provenance,protected_sha256={str(p):h for p,h in protected.items()},
                    frozen_upstream_digest=digest,task_sha256=sha(output/'tasks.json'))
        for step,batch in enumerate(batches,1):
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                loss=short_loss(model,teacher,tokenizer,batch,events,device)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
            scale=scaler.get_scale(); scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient; no update counted')
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()<scale: raise RuntimeError('skipped optimizer update')
            row=dict(step=step,size=batch[0]['size'],pair_id=batch[0]['pair_id'],task_uids=[t['uid'] for t in batch],
                role=batch[2]['kind'],yes=1,no=1,loss=float(loss.detach()),gradient_norm=float(norm))
            with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            if step%25==0: print(json.dumps(dict(phase='train',**row)),flush=True)
            if step in (100,250,500):
                torch.save(dict(step=step,init_step=500,model_settings=model.settings,model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict(),source=source,settings=settings,
                    torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else []),
                    output/f'qa-adapter-step{step}.pt')
        after=evaluate_short(model,teacher,tokenizer,dev,events,device); atomic_json(output/'dev-after.json',after)
        blind=evaluate_short(model,teacher,tokenizer,dev,events,device,'none'); atomic_json(output/'dev-none.json',blind)
        swaps={}
        for pair in taskset['pairs']:
            if pair['cohort']=='dev':
                a,b=[taskset['by_uid'][u] for u in pair['task_ids']]; swaps[a['uid']]=b; swaps[b['uid']]=a
        swapped=evaluate_short(model,teacher,tokenizer,[t for t in dev if t['kind']=='presence'],events,device,'swapped',swaps=swaps)
        atomic_json(output/'dev-swapped.json',swapped)
        compressed=evaluate_short(model,teacher,tokenizer,[t for t in dev if t['size']==4],events,device,'compressed',compressor)
        atomic_json(output/'dev-l1.json',compressed)
        atomic_json(output/'l1-matched.json',build_l1_match(after,compressed,events))
        # Per-group semantics and exact event source manifest accompany QA comparison.
        retention=[]; projection=runtime.controller.semantic_projection
        with torch.no_grad():
            for group in taskset['groups']:
                if group['cohort']=='dev' and group['size']==4:
                    with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                        summary,raw=compress_example(compressor,(1,[events[u] for u in group['source_ids']]),device)
                    cosine=float(torch.nn.functional.cosine_similarity(projection(summary),projection(raw)))
                    retention.append(dict(group_uid=group['uid'],source_ids=group['source_ids'],source_sha256=group['source_sha256'],cosine=cosine))
        atomic_json(output/'l1-retention.json',dict(groups=retention))
        if upstream_digest(runtime)!=digest or any(sha(p)!=h for p,h in protected.items()): raise RuntimeError('protected source changed')
        fresh=torch.load(Path(settings['source_output'])/'events.pt',map_location='cpu',weights_only=False)
        if not all(torch.equal(a[name],b[name]) for a,b in zip(fresh['events'],payload['events']) for name in ('tokens','key')):
            raise RuntimeError('original tokens or keys changed')
        stored_compressor=torch.load(settings['protected_compressor'],map_location='cpu',weights_only=False)['model_state']
        if set(compressor.state_dict())!=set(stored_compressor) or not all(torch.equal(v.cpu(),stored_compressor[k]) for k,v in compressor.state_dict().items()):
            raise RuntimeError('compressor memory weights changed')
        checks=[]
        for step in (100,250,500):
            path=output/f'qa-adapter-step{step}.pt'; saved=torch.load(path,map_location='cpu',weights_only=False)
            actual=sorted({int(s['step']) for s in saved['optimizer_state']['state'].values()})
            if actual!=[step] or saved['source']['task_sha256']!=sha(output/'tasks.json') or not all(bool(torch.isfinite(v).all()) for v in saved['model_state'].values()):
                raise RuntimeError('checkpoint readback failed')
            checks.append(dict(step=step,actual_steps=actual,sha256=sha(path)))
        if not all(torch.equal(v.cpu(),saved['model_state'][k]) for k,v in model.state_dict().items()): raise RuntimeError('final saved mismatch')
        torch.save(dict(step=500,init_step=500,model_settings=model.settings,model_state=saved['model_state'],source=source),output/'qa-adapter-inference.pt')
        exported=torch.load(output/'qa-adapter-inference.pt',map_location='cpu',weights_only=False)
        if not all(torch.equal(v,exported['model_state'][k]) for k,v in saved['model_state'].items()): raise RuntimeError('export readback mismatch')
        gates=dict(single_retained=after['metrics']['1']['identify_macro_recall']>=before['metrics']['1']['identify_macro_recall']-.03)
        for size in ('2','4'):
            m=after['metrics'][size]
            gates.update({f'{size}_presence':m['presence_balanced_accuracy']>=.8,f'{size}_pair':m['pair_both_correct']>=.7,
                          f'{size}_first':m['first']['accuracy']>=.8,f'{size}_last':m['last']['accuracy']>=.8})
        atomic_json(output/'result.json',dict(updates=500,total_qa_updates=1000,size_updates=dict(Counter(b[0]['size'] for b in batches)),
            before=before['metrics'],after=after['metrics'],none=blind['metrics'],compressed=compressed['metrics'],
            gates=gates,development_pass=all(gates.values()),formal_go=False,source=source,checkpoints=checks,
            frozen_upstream_unchanged=True,event_tokens_keys_unchanged=True,compressor_unchanged=True,
            inference_sha256=sha(output/'qa-adapter-inference.pt')))
        status.update(status='complete',duration_s=time.time()-started)
    except Exception as error:
        status.update(status='failed',error=repr(error),duration_s=time.time()-started); atomic_json(output/'status.json',status); raise
    atomic_json(output/'status.json',status); print(json.dumps(dict(phase='complete',duration_s=status['duration_s'])),flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args(argv); run(yaml.safe_load(args.config.read_text())['stage4'],args.config)


if __name__=='__main__': main()
