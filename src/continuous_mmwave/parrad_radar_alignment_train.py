"""Fixed-budget single-event connector; new QA reader is used in every branch."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import torch
import yaml

from continuous_mmwave.models.qa_adapter import EventQAAdapter
from continuous_mmwave.parrad_event_boundary_qa import load_boundary_sources
from continuous_mmwave.parrad_qa_reader import load_multi_reader,reader_digest,retrieval_snapshot
from continuous_mmwave.parrad_radar_alignment import (build_tasks,training_batches,teacher_query,answer_positions,
    radar_answer_positions,answer_objective,development_gates,text_preflight_decision,validate_predictions)
from continuous_mmwave.parrad_single_event_qa import score_predictions
from continuous_mmwave.parrad_stage4_qa import grade_answer
from continuous_mmwave.parrad_stage4_runner import atomic_json,sha,upstream_digest
from continuous_mmwave.parrad_temporal_qa import finite_optimizer_step
from continuous_mmwave.stages.parrad_stage4 import chat_supervision,generate_from_memory


def new_connector(settings,device):
    torch.manual_seed(settings['seed'])
    return EventQAAdapter(dim=896,bottleneck=256).to(device)


def backward_batch(model,reader,tokenizer,batch,events,device,scaler,weight,temperature):
    totals=dict(loss=0.,ce=0.,kl=0.)
    for task in batch:
        with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
            ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
            # Teacher has no radar context and sees correct fit-only text evidence.
            teacher=teacher_labels=None
            if weight:
                if task['cohort']!='fit' or events[task['event_uid']]['cohort']!='fit':
                    raise ValueError('distillation evidence must be fit-only')
                tids,tattention,ttarget=chat_supervision(tokenizer,teacher_query(task,events),task['answer'],device)
                with torch.no_grad():
                    empty=torch.empty(1,0,model.settings['dim'],device=device,dtype=reader.compute_dtype)
                    teacher,teacher_labels=answer_positions(reader.caption_logits(empty,tids,tattention),tids,tattention,ttarget,0)
            raw=events[task['event_uid']]['tokens'][None].to(device)
            student,labels=radar_answer_positions(model,reader,raw,ids,attention,target)
            values=answer_objective(student,labels,teacher,teacher_labels,weight=weight,temperature=temperature)
        if not all(torch.isfinite(v) for v in values.values()): raise RuntimeError('nonfinite alignment loss')
        for k,v in values.items(): totals[k]+=float(v.detach())/len(batch)
        scaler.scale(values['loss']/len(batch)).backward()
    return totals


@torch.no_grad()
def evaluate(model,reader,tokenizer,tasks,events,device,mode):
    model.eval();lookup={t['uid']:t for t in tasks};rows=[]
    for index,t in enumerate(tasks):
        actual=t['event_uid'];query=t['query']
        if mode=='swapped': actual=lookup[t['pair_id']+(':No' if t['answer']=='Yes' else ':Yes')]['event_uid']
        if mode in ('text','none'):
            context=torch.empty(1,0,model.settings['dim'],device=device)
            if mode=='text': query=teacher_query(t,events)
        else:
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                context=model(events[actual]['tokens'][None].to(device))
        answer=generate_from_memory(reader,tokenizer,context,query,max_new_tokens=24)
        row=dict(t,reference=t['answer'],generated=answer,reader='text-qa-500',actual_event_uid=actual,
            context_tokens=context.shape[1],correct=grade_answer(answer,t['answer'],'presence' if t['kind']=='presence' else 'first'))
        if mode=='text': row['oracle_query']=query
        if mode=='swapped': row['evidence_reference']='Yes' if events[actual]['label']==t['query_label'] else 'No'
        rows.append(row)
        if (index+1)%100==0: print(json.dumps(dict(phase='evaluate',mode=mode,done=index+1,total=len(tasks))),flush=True)
    result=dict(mode=mode,rows=rows,metrics=score_predictions(rows))
    if mode=='swapped': result['evidence_metrics']=score_predictions([dict(r,reference=r['evidence_reference']) for r in rows])
    validate_predictions(result,tasks,events)
    return result


def save_checkpoint(path,model,optimizer,scaler,step,source,arm):
    torch.save(dict(step=step,source=source,arm=arm,model_settings=model.settings,
        model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        optimizer_state=optimizer.state_dict(),scaler_state=scaler.state_dict()),path)


def readback_checkpoints(folder,model,source,arm):
    checks=[]
    for step in (100,250,500):
        path=folder/f'connector-step{step}.pt'; saved=torch.load(path,map_location='cpu',weights_only=False)
        state=saved['optimizer_state']['state']
        if saved['step']!=step or saved['source']!=source or saved['arm']!=arm or len(state)!=6 or {int(v['step']) for v in state.values()}!={step}:
            raise RuntimeError('connector actual optimizer steps mismatch')
        if not all(torch.isfinite(v).all() for v in saved['model_state'].values()) or not all(
            torch.isfinite(v[k]).all() for v in state.values() for k in ('exp_avg','exp_avg_sq')):
            raise RuntimeError('nonfinite saved connector')
        checks.append(dict(step=step,sha256=sha(path),actual_updates=step))
    if set(model.state_dict())!=set(saved['model_state']) or not all(torch.equal(v.cpu(),saved['model_state'][k]) for k,v in model.state_dict().items()):
        raise RuntimeError('final connector readback mismatch')
    exported={k:saved[k] for k in ('step','source','arm','model_settings','model_state')}
    torch.save(exported,folder/'connector-inference.pt')
    fresh=torch.load(folder/'connector-inference.pt',map_location='cpu',weights_only=False)
    if not all(torch.equal(v,fresh['model_state'][k]) for k,v in saved['model_state'].items()): raise RuntimeError('inference export mismatch')
    return checks


def run(settings,config_path):
    if (settings['seed'],settings['steps_per_arm'],settings['learning_rate'],settings['distill_weight'],settings['temperature'])!=(42,500,.0001,.5,2.):
        raise ValueError('locked single-event experiment')
    output=Path(settings['output_dir']);output.mkdir(exist_ok=False)
    start=time.time();package=Path(__file__).parent
    modules=['parrad_radar_alignment.py','parrad_radar_alignment_train.py','models/qa_adapter.py',
        'parrad_qa_reader.py','parrad_qa_text_decoder.py','parrad_temporal_qa.py','parrad_event_boundary_qa.py',
        'parrad_single_event_qa.py','parrad_stage4_qa.py','parrad_stage4_runner.py','stages/parrad_stage4.py','models/event_former.py']
    status=dict(status='running',phase='load',settings=settings,started_at=start,formal_go=False,
        config_sha256=sha(config_path),code_sha256={p:sha(package/p) for p in modules},actual_updates={a:0 for a in ('ce','distill')})
    atomic_json(output/'status.json',status)
    try:
        torch.set_num_threads(1);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        runtime,tokenizer,initial,payload,protected,provenance,digest=load_boundary_sources(settings,device)
        reader,reader_source=load_multi_reader(settings,runtime.controller.event_former.adapter,protected,digest,device)
        protected[Path(settings['qa_reader'])]=settings['qa_reader_sha256']
        events={e['uid']:e for e in payload['events']};tasks=build_tasks(payload['events'],42);batches=training_batches(tasks,42)
        fit=[t for t in tasks if t['cohort']=='fit'];dev=[t for t in tasks if t['cohort']=='dev']
        if len(fit)!=588 or len(dev)!=297: raise ValueError('unexpected fit/dev task count')
        atomic_json(output/'tasks.json',tasks);atomic_json(output/'batches.json',[[t['uid'] for t in b] for b in batches])
        source=dict(provenance=provenance,protected_sha256={str(p):h for p,h in protected.items()},
            upstream_digest=digest,reader=reader_source,task_sha256=sha(output/'tasks.json'),batch_sha256=sha(output/'batches.json'),
            initial='new seed42 identity residual; raw 8-token event; no old QA weights',teacher='fit-only correct text; detached answer logits',
            text_preflight_policy=settings.get('text_preflight_policy','required'),
            numerical_policy=dict(initial_scale=256,max_backoffs=12,retry_same_batch=True),final_selection=500)
        atomic_json(output/'source.json',source)
        model=new_connector(settings,device)
        torch.save(dict(model_settings=model.settings,model_state=model.state_dict(),source=source),output/'connector-initial.pt')
        status.update(phase='text-preflight');atomic_json(output/'status.json',status)
        oracle=evaluate(model,reader,tokenizer,fit,events,device,'text');atomic_json(output/'fit-text.json',oracle)
        decision=text_preflight_decision(oracle['metrics'],source['text_preflight_policy'])
        atomic_json(output/'text-gates.json',decision);status['text_preflight']=decision
        atomic_json(output/'dev-text.json',evaluate(model,reader,tokenizer,dev,events,device,'text'))
        if not decision['proceed']:
            status.update(status='blocked',phase='text-preflight',reason='single-event text teacher failed 95% fit gates; no optimization')
            return
        status.update(phase='baseline');atomic_json(output/'status.json',status)
        for cohort,selected in (('fit',fit),('dev',dev)):
            atomic_json(output/f'{cohort}-before.json',evaluate(model,reader,tokenizer,selected,events,device,'raw'))
        atomic_json(output/'dev-none.json',evaluate(model,reader,tokenizer,dev,events,device,'none'))
        # Independently reproduce the original QA route without changing new-reader dispatch.
        original=EventQAAdapter(**initial['model_settings']).to(device);original.load_state_dict(initial['model_state']);original.requires_grad_(False)
        original_rows=[]
        for t in dev:
            with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'): context=original(events[t['event_uid']]['tokens'][None].to(device))
            answer=generate_from_memory(runtime.controller.event_former.adapter,tokenizer,context,t['query'],max_new_tokens=24)
            original_rows.append(dict(uid=t['uid'],query=t['query'],reference=t['answer'],generated=answer))
        previous=json.loads((Path(settings['previous_output'])/'dev-after.json').read_text())['rows']
        exact=lambda rows:{r['uid']:(r['query'],r['reference'],r['generated']) for r in rows if r.get('size',1)==1}
        if len(original_rows)!=297 or exact(original_rows)!=exact(previous): raise RuntimeError('original single QA baseline changed')
        atomic_json(output/'original-single.json',dict(rows=original_rows,previous_output=settings['previous_output'],predictions_exact=True))
        del original
        retrieval=retrieval_snapshot(runtime,tokenizer,events,device);atomic_json(output/'retrieval-before.json',retrieval)
        results={}
        for arm,weight in (('ce',0.),('distill',settings['distill_weight'])):
            folder=output/arm;folder.mkdir();model=new_connector(settings,device)
            initial_saved=torch.load(output/'connector-initial.pt',map_location='cpu',weights_only=False)
            if not all(torch.equal(v.cpu(),initial_saved['model_state'][k]) for k,v in model.state_dict().items()): raise RuntimeError('arm initialization changed')
            parameters=list(model.parameters());optimizer=torch.optim.AdamW(parameters,lr=settings['learning_rate'],weight_decay=.01)
            scaler=torch.amp.GradScaler('cuda',init_scale=256,enabled=device.type=='cuda')
            status.update(phase='training',arm=arm);atomic_json(output/'status.json',status)
            for step,batch in enumerate(batches,1):
                model.train();pending={}
                def backward():
                    pending.update(backward_batch(model,reader,tokenizer,batch,events,device,scaler,weight,settings['temperature']))
                    return pending['loss']
                def retry(row):
                    with (folder/'numerical-retries.jsonl').open('a') as f: f.write(json.dumps(dict(row,step=step,task_uids=[t['uid'] for t in batch]))+'\n')
                outcome=finite_optimizer_step(parameters,optimizer,scaler,backward,device,retry_sink=retry)
                row=dict(outcome,step=step,ce=pending['ce'],kl=pending['kl'],weight=weight,query_label=batch[0]['query_label'],
                    task_uids=[t['uid'] for t in batch],yes=1,no=1,identify=1)
                with (folder/'training.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                status['actual_updates'][arm]=step
                if step%25==0: atomic_json(output/'status.json',status);print(json.dumps(dict(phase='training',arm=arm,**row)),flush=True)
                if step in (100,250,500):save_checkpoint(folder/f'connector-step{step}.pt',model,optimizer,scaler,step,source,arm)
            checks=readback_checkpoints(folder,model,source,arm)
            status.update(phase='evaluate',arm=arm);atomic_json(output/'status.json',status)
            after={}
            for cohort,selected in (('fit',fit),('dev',dev)):
                value=evaluate(model,reader,tokenizer,selected,events,device,'raw');atomic_json(folder/f'{cohort}-after.json',value);after[cohort]=value['metrics']
            atomic_json(folder/'dev-swapped.json',evaluate(model,reader,tokenizer,[t for t in dev if t['kind']=='presence'],events,device,'swapped'))
            if upstream_digest(runtime)!=digest or reader_digest(reader)!=reader_source['frozen_digest'] or any(sha(p)!=h for p,h in protected.items()):
                raise RuntimeError('frozen model/source changed')
            fresh=retrieval_snapshot(runtime,tokenizer,events,device);atomic_json(folder/'retrieval-after.json',fresh)
            if fresh!=retrieval: raise RuntimeError('original retrieval changed')
            gate=development_gates(after['dev']);results[arm]=dict(arm=arm,updates=500,weight=weight,source=source,
                after=after,gates=gate,development_pass=all(gate.values()),checkpoints=checks,
                inference_sha256=sha(folder/'connector-inference.pt'),frozen_reader_exact=True,original_retrieval_exact=True,formal_go=False)
            atomic_json(folder/'result.json',results[arm])
        atomic_json(output/'result.json',dict(arms=results,source=source,original_single_exact=True,formal_go=False))
        status.update(status='complete',phase='complete')
    except Exception as e:
        status.update(status='failed',reason=repr(e));raise
    finally:
        status['duration_s']=time.time()-start;atomic_json(output/'status.json',status)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);args=parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['stage4'],args.config)


if __name__=='__main__':main()
