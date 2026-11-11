"""Independent, fixed-budget text QA LoRA. Original radar/retrieval stays frozen."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
import yaml

from continuous_mmwave.models.event_former import Qwen2Adapter
from continuous_mmwave.parrad_event_boundary_qa import load_boundary_sources
from continuous_mmwave.parrad_event_boundary_tasks import build_boundary_tasks,score_boundary
from continuous_mmwave.parrad_temporal_qa import temporal_batches,temporal_query,text_oracle_query,oracle_gate
from continuous_mmwave.parrad_stage4_qa import grade_answer
from continuous_mmwave.parrad_stage4_runner import atomic_json,sha,upstream_digest
from continuous_mmwave.stages.parrad_stage4 import chat_supervision,generate_from_memory


def create_reader(causal_lm,rank=8):
    reader=Qwen2Adapter(causal_lm,num_layers=causal_lm.config.num_hidden_layers,lora_rank=rank)
    # Text-only inputs otherwise have no trainable prefix. Checkpointed layers
    # need input gradients to propagate into the new LoRA tensors.
    reader.model.enable_input_require_grads()
    return reader


def frozen_base_digest(reader):
    digest=hashlib.sha256()
    for name,value in reader.state_dict().items():
        if 'lora_' in name: continue
        digest.update(f'{name}:{tuple(value.shape)}:{value.dtype}'.encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def answer_loss(reader,ids,attention,target):
    if ids.shape!=attention.shape or ids.shape!=target.shape: raise ValueError('caption masks must match ids')
    valid=target[:,1:].bool() & attention[:,1:].bool()
    if not valid.any(): raise ValueError('text QA requires assistant answer targets')
    context=torch.empty(ids.shape[0],0,reader.hidden_size,device=ids.device,dtype=reader.compute_dtype)
    logits=reader.caption_logits(context,ids,attention)
    return F.cross_entropy(logits[:,:-1][valid].float(),ids[:,1:][valid])


def text_batches(taskset,events,seed):
    result=temporal_batches(taskset,seed)
    for batch in result:
        for task in batch:
            if task['cohort']!='fit' or any(events[u]['cohort']!='fit' for u in task['source_ids']):
                raise ValueError('text decoder supervision must be fit-only')
            task['query']=text_oracle_query(task,events)
    return result


def adapter_state(reader):
    from peft import get_peft_model_state_dict
    return {k:v.detach().cpu().clone() for k,v in get_peft_model_state_dict(reader.model).items()}


def save_checkpoint(path,reader,optimizer,step,source,scaler=None):
    torch.save(dict(step=step,adapter_state=adapter_state(reader),optimizer_state=optimizer.state_dict(),
        optimizer_names=[n for n,p in reader.named_parameters() if p.requires_grad],
        reader_settings=dict(rank=reader.model.peft_config['default'].r,layers=reader.layer_count,dim=reader.hidden_size),
        scaler_state=scaler.state_dict() if scaler is not None else None,source=source,
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),path)


def verify_checkpoint(path,reader,source):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    states=saved['optimizer_state']['state']; names=saved['optimizer_names']
    expected_names=[n for n,p in reader.named_parameters() if 'lora_' in n]
    expected_settings=dict(rank=reader.model.peft_config['default'].r,layers=reader.layer_count,dim=reader.hidden_size)
    if (saved['source']!=source or not states or len(states)!=len(names) or names!=expected_names
        or saved['reader_settings']!=expected_settings): raise RuntimeError('checkpoint source/parameter mismatch')
    actual={int(v['step']) for v in states.values()}
    if actual!={saved['step']}: raise RuntimeError('incorrect real optimizer steps')
    expected=adapter_state(reader)
    if set(expected)!=set(saved['adapter_state']) or not all('lora_' in k for k in saved['adapter_state']):
        raise RuntimeError('adapter-only export key mismatch')
    if not all(torch.isfinite(v).all() for v in saved['adapter_state'].values()): raise RuntimeError('nonfinite checkpoint')
    if not all(torch.isfinite(v['exp_avg']).all() and torch.isfinite(v['exp_avg_sq']).all() for v in states.values()):
        raise RuntimeError('nonfinite optimizer moments')
    return dict(step=saved['step'],actual_steps=saved['step'],parameters=len(states),sha256=sha(path))


@torch.no_grad()
def evaluate_text(reader,tokenizer,tasks,events,device):
    reader.eval(); rows=[]
    for index,task in enumerate(tasks):
        query=text_oracle_query(task,events)
        context=torch.empty(1,0,reader.hidden_size,device=device,dtype=reader.compute_dtype)
        answer=generate_from_memory(reader,tokenizer,context,query,max_new_tokens=96 if task['kind']=='sequence' else 24)
        kind='summary' if task['kind']=='sequence' else 'presence' if task['kind']=='presence' else 'first'
        rows.append(dict(uid=task['uid'],cohort=task['cohort'],kind=task['kind'],size=task['size'],label=task['label'],
            position=task.get('position'),pair_id=task.get('pair_id'),source_ids=task['source_ids'],
            query=temporal_query(task),oracle_query=query,reference=task['answer'],generated=answer,
            context_tokens=0,mode='text',correct=grade_answer(answer,task['answer'],kind)))
        if (index+1)%100==0: print(json.dumps(dict(phase='text-evaluation',done=index+1,total=len(tasks))),flush=True)
    return dict(mode='text',rows=rows,metrics=score_boundary(rows))


def run(settings,config_path):
    if (settings['steps'],settings['seed'],settings['rank'])!=(500,42,8): raise ValueError('locked 500-step seed42 rank8 run')
    output=Path(settings['output_dir'])
    if output.exists(): raise FileExistsError('refuse duplicate text decoder output')
    output.mkdir(parents=True); started=time.time()
    package=Path(__file__).parent
    modules=('parrad_qa_text_decoder.py','parrad_temporal_qa.py','models/event_former.py',
        'parrad_event_boundary_qa.py','parrad_event_boundary_tasks.py','parrad_stage4_qa.py',
        'parrad_short_context_tasks.py','parrad_short_context_qa.py','parrad_stage4_runner.py','stages/parrad_stage4.py')
    status=dict(status='running',phase='load',settings=settings,config_sha256=sha(config_path),
        code_sha256={p:sha(package/p) for p in modules},actual_updates=0,formal_go=False,started_at=started)
    atomic_json(output/'status.json',status)
    try:
        torch.set_num_threads(1); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        runtime,tokenizer,initial,payload,protected,provenance,digest=load_boundary_sources(settings,device)
        taskset=build_boundary_tasks(payload['events'],42); events={e['uid']:e for e in payload['events']}
        atomic_json(output/'tasks.json',{k:v for k,v in taskset.items() if k!='by_uid'})
        batches=text_batches(taskset,events,42)
        atomic_json(output/'batches.json',[[t['uid'] for t in b] for b in batches])
        model_dir=Path(settings['base_model'])
        base_files={str(p.relative_to(model_dir)):sha(p) for p in sorted(model_dir.rglob('*')) if p.is_file()}
        if not base_files: raise ValueError('missing local base model')
        from transformers import AutoModelForCausalLM
        torch.manual_seed(42)
        causal_lm=AutoModelForCausalLM.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float16 if device.type=='cuda' else torch.float32)
        if causal_lm.config.num_hidden_layers!=24 or causal_lm.config.hidden_size!=896 or causal_lm.config.model_type!='qwen2':
            raise ValueError('expected original local Qwen2.5 0.5B architecture')
        reader=create_reader(causal_lm,settings['rank']).to(device)
        if {p.data_ptr() for p in reader.parameters()} & {p.data_ptr() for p in runtime.controller.event_former.adapter.parameters()}:
            raise RuntimeError('new QA reader shares weight storage with original teacher')
        base_digest=frozen_base_digest(reader)
        source=dict(provenance=provenance,protected_sha256={str(p):h for p,h in protected.items()},
            original_upstream_digest=digest,new_frozen_base_digest=base_digest,base_model=str(model_dir),base_files=base_files,
            task_sha256=sha(output/'tasks.json'),batch_sha256=sha(output/'batches.json'),original_qa_updates=1000)
        atomic_json(output/'source.json',source)
        fit=[t for t in taskset['tasks'] if t['cohort']=='fit' and t['size']>1]
        dev=[t for t in taskset['tasks'] if t['cohort']=='dev' and t['size']>1]
        status.update(phase='fit-before'); atomic_json(output/'status.json',status)
        before=evaluate_text(reader,tokenizer,fit,events,device); atomic_json(output/'fit-before.json',before)
        trainable=[p for p in reader.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=settings['learning_rate'],weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',init_scale=256,enabled=device.type=='cuda')
        status.update(phase='training',trainable_parameters=sum(p.numel() for p in trainable)); atomic_json(output/'status.json',status)
        for step,batch in enumerate(batches,1):
            reader.train(); optimizer.zero_grad(set_to_none=True); losses=[]
            for task in batch:
                ids,attention,target=chat_supervision(tokenizer,task['query'],task['answer'],device)
                with torch.autocast(device.type,dtype=torch.float16,enabled=device.type=='cuda'):
                    loss=answer_loss(reader,ids,attention,target)
                if not torch.isfinite(loss): raise RuntimeError('nonfinite text decoder loss')
                scaler.scale(loss/len(batch)).backward(); losses.append(float(loss.detach()))
            scale=scaler.get_scale(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite text decoder gradient')
            scaler.step(optimizer); scaler.update()
            if scaler.get_scale()<scale: raise RuntimeError('text decoder update skipped')
            row=dict(step=step,size=batch[0]['size'],yes=1,no=1,role=batch[2]['kind'],position=batch[2].get('position'),
                task_uids=[t['uid'] for t in batch],loss=sum(losses)/3,task_losses=losses,gradient_norm=float(norm))
            with (output/'training.jsonl').open('a') as stream: stream.write(json.dumps(row)+'\n')
            status['actual_updates']=step
            if step%25==0:
                atomic_json(output/'status.json',status); print(json.dumps(dict(phase='training',**row)),flush=True)
            if step in (100,250,500): save_checkpoint(output/f'qa-lora-step{step}.pt',reader,optimizer,step,source,scaler)
        reader.requires_grad_(False); reader.eval()
        status.update(phase='fit-after'); atomic_json(output/'status.json',status)
        after=evaluate_text(reader,tokenizer,fit,events,device); atomic_json(output/'fit-after.json',after)
        gates=oracle_gate(after['metrics']); atomic_json(output/'fit-gates.json',dict(gates=gates,passed=all(gates.values())))
        dev_result=None
        if all(gates.values()):
            status.update(phase='dev-report'); atomic_json(output/'status.json',status)
            dev_result=evaluate_text(reader,tokenizer,dev,events,device); atomic_json(output/'dev-after.json',dev_result)
        status.update(phase='readback'); atomic_json(output/'status.json',status)
        checks=[verify_checkpoint(output/f'qa-lora-step{step}.pt',reader,source) for step in (100,250,500)]
        final=torch.load(output/'qa-lora-step500.pt',map_location='cpu',weights_only=False)
        actual=adapter_state(reader)
        if not all(torch.equal(v,final['adapter_state'][k]) for k,v in actual.items()): raise RuntimeError('final checkpoint does not match reader')
        exported={k:final[k] for k in ('step','adapter_state','reader_settings','source')}
        torch.save(exported,output/'qa-lora-inference.pt')
        from peft import set_peft_model_state_dict
        saved=torch.load(output/'qa-lora-inference.pt',map_location='cpu',weights_only=False)
        # Clear current adapters first so readback cannot succeed by a no-op load.
        with torch.no_grad():
            for name,param in reader.named_parameters():
                if 'lora_' in name: param.zero_()
        set_peft_model_state_dict(reader.model,saved['adapter_state'])
        if not all(torch.equal(v,adapter_state(reader)[k]) for k,v in actual.items()): raise RuntimeError('LoRA export reload failed')
        probe=evaluate_text(reader,tokenizer,fit[:12],events,device)
        if probe['rows']!=after['rows'][:12]: raise RuntimeError('export generation mismatch')
        if frozen_base_digest(reader)!=base_digest or upstream_digest(runtime)!=digest: raise RuntimeError('frozen model weights changed')
        if any(sha(p)!=h for p,h in protected.items()) or any(sha(model_dir/p)!=h for p,h in base_files.items()):
            raise RuntimeError('protected files changed')
        report=dict(updates=500,trainable_parameters=sum(p.numel() for p in trainable),before=before['metrics'],after=after['metrics'],
            gates=gates,text_fit_pass=all(gates.values()),dev=None if dev_result is None else dev_result['metrics'],
            checks=checks,source=source,new_base_unchanged=True,original_upstream_unchanged=True,
            export_reload_exact=True,inference_sha256=sha(output/'qa-lora-inference.pt'),
            original_single_event_path_untouched=True,radar_qa_evaluated=False,formal_go=False,
            size_updates=dict(Counter(b[0]['size'] for b in batches)))
        atomic_json(output/'result.json',report)
        status.update(status='complete',phase='complete',text_fit_pass=all(gates.values()))
    except Exception as error:
        status.update(status='failed',reason=repr(error)); raise
    finally:
        status['duration_s']=time.time()-started; atomic_json(output/'status.json',status)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',required=True); args=parser.parse_args()
    run(yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))['stage4'],args.config)


if __name__=='__main__': main()
