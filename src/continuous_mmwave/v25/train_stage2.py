"""Native fixed-budget gate migration with exact data/teacher trajectory."""
from __future__ import annotations

from collections import defaultdict
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cache import load_original
from .checkpoint import file_sha, state_digest, module_digest, code_digest
from .data import mean_evidence_target
from .models import install_age_free_gate
from .prepare import write_json
from .stage2 import Stage2Migration
from .training import accumulated_updates, fixed_query_retrieval,save_failed_optimizer


def frozen_digest(frontend):
    return state_digest({name:value for name,value in frontend.state_dict().items()
                         if not name.startswith('event_former.state_gate.')})


def prefix_evidence_supervision(record,*,stream_range,commit_range):
    """Visible ranges are updates; supervised ranges are original frame indices."""
    values,masks=record['frame_target'],record['frame_target_mask']
    if values.ndim!=2 or masks.shape!=values.shape:
        raise ValueError('matching frame attribute values/masks required')
    result={}
    for path,bounds in (('stream',stream_range),('commit',commit_range)):
        if len(bounds)!=2 or any(not isinstance(v,int) for v in bounds):
            raise ValueError('integer update coverage required')
        start,end=bounds
        if not 0<=start<end or 2*end>len(values):
            raise ValueError('evidence coverage outside visible frames')
        target,mask=mean_evidence_target(values[2*start:2*end].numpy(),masks[2*start:2*end].numpy())
        result.update({path+'_target':torch.from_numpy(target),path+'_target_mask':torch.from_numpy(mask),
                       path+'_visible_range':torch.tensor([start,end]),
                       path+'_target_range':torch.tensor([2*start,2*end])})
    return result


class GateBatches:
    def __init__(self,frontend,tokenizer,index,teachers,schedule,*,continuation=None):
        self.frontend,self.tokenizer,self.schedule=frontend,tokenizer,schedule
        self.device=next(frontend.parameters()).device
        self.runs={r['uid']:r for r in index['runs']}
        self.records={}; self.teachers={}; self.states={}; self.watermarks={}
        for record in index['records']:
            if record['role']!='fit':
                continue
            if file_sha(record['path'])!=record['sha256']:
                raise ValueError('encoder cache source SHA mismatch')
            self.records[record['uid']]=torch.load(record['path'],map_location='cpu',weights_only=False)
        for record in teachers['records']:
            if record['uid'] not in self.records:
                continue
            if file_sha(record['path'])!=record['sha256']:
                raise ValueError('teacher source SHA mismatch')
            self.teachers[record['uid']]=torch.load(record['path'],map_location='cpu',weights_only=False)
        self.draw_counts=defaultdict(int)
        if continuation is not None:
            self.states={uid:value.to(self.device) for uid,value in continuation['states'].items()}
            self.watermarks=dict(continuation['watermarks'])
            self.draw_counts.update(continuation['draw_counts'])

    def __iter__(self):
        from ..train import _tokenize_captions
        from ..stages.parrad_stage2 import default_parrad_event_templates
        templates=default_parrad_event_templates()
        for group in self.schedule:
            inputs=[]; commit_inputs=[]; masks=[]; commit_masks=[]; initial=[]
            supervision=[]; stream_keys=[]; commit_keys=[]; labels=[]; captions=[]
            for chunk in group['chunks']:
                run=self.runs[chunk['uid']]
                uid=chunk['uid']; record=self.records[run['record_uid']]
                start,end=run['start']+chunk['start'],run['start']+chunk['end']
                if chunk['reset']:
                    initial.append(self.frontend.event_former.initial_state(1)[0].detach())
                else:
                    if self.watermarks.get(uid)!=chunk['start']:
                        raise ValueError('continuous gate state watermark does not match next chunk')
                    initial.append(self.states[uid])
                inputs.append(record['radar'][start:end]); masks.append(record['radar_mask'][start:end])
                commit_inputs.append(record['radar'][run['start']:end].flatten(0,1))
                commit_masks.append(record['radar_mask'][run['start']:end].flatten())
                supervision.append(prefix_evidence_supervision(record,stream_range=(run['start'],end),
                                                               commit_range=(run['start'],end)))
                teacher=self.teachers[run['record_uid']][f'{uid}:{chunk["end"]}']
                if (teacher['visible_start'],teacher['visible_end'])!=(run['start'],end):
                    raise ValueError('teacher visible input range mismatch')
                stream_keys.append(teacher['stream_key']); commit_keys.append(teacher['commit_key'])
                labels.append(run['label'])
                captions.append(templates[run['label']][self.draw_counts[uid]%3]); self.draw_counts[uid]+=1
            maximum=max(len(r) for r in commit_inputs)
            radar=inputs[0].new_zeros((len(inputs),maximum,inputs[0].shape[-1]))
            radar_mask=torch.zeros((len(inputs),maximum),dtype=torch.bool)
            for i,(value,mask) in enumerate(zip(commit_inputs,commit_masks)):
                radar[i,:len(value)]=value; radar_mask[i,:len(value)]=mask
            ids,caption_mask=_tokenize_captions(self.tokenizer,tuple(captions))
            batch=dict(radar=torch.stack(inputs),radar_mask=torch.stack(masks),
                initial_state=torch.stack(initial),commit_radar=radar,commit_radar_mask=radar_mask,
                teacher_stream_key=torch.stack(stream_keys),teacher_commit_key=torch.stack(commit_keys),
                labels=torch.tensor(labels),caption_ids=ids,caption_mask=caption_mask)
            batch.update({key:torch.stack([s[key] for s in supervision]) for key in supervision[0]})
            def on_state(state, chunks=copy.deepcopy(group['chunks'])):
                for i,chunk in enumerate(chunks):
                    self.states[chunk['uid']]=state[i].detach().clone()
                    self.watermarks[chunk['uid']]=chunk['end']
            batch={key:value.to(self.device) for key,value in batch.items()}
            batch['on_state']=on_state
            yield batch

    def continuation_state(self):
        return dict(states={k:v.cpu() for k,v in self.states.items()},watermarks=dict(self.watermarks),
                    draw_counts=dict(self.draw_counts))


def preflight_schedule(index,micro_batch):
    runs=[r for r in index['runs'] if r['role']=='fit']
    if not runs:
        raise ValueError('native preflight fit runs missing')
    runs=sorted(runs,key=lambda r:-(r['end']-r['start']))
    groups=[]
    for number in range(2):
        selected=[]
        for run in runs:
            if not selected or run['label']!=selected[0]['label'] or len(selected)>=2:
                selected.append(run)
                if len(selected)==micro_batch:
                    break
        if len(selected)<micro_batch or len({r['label'] for r in selected})<2:
            raise ValueError('native preflight cannot supply distinct action batch')
        length=min(64,min(r['end']-r['start'] for r in selected))
        groups.append(dict(step=0,micro=number+1,chunks=[dict(uid=r['uid'],start=0,end=length,reset=True) for r in selected]))
    return groups


def native_gate_preflight(frontend,tokenizer,heads,index,teachers,output,*,seed):
    """At most2 batch4 +2 batch2 forward/backward trials; no optimizer exists."""
    output=Path(output)
    original_gate=frontend.event_former.state_gate
    original_evidence={k:v.detach().clone() for k,v in heads.evidence.state_dict().items()}
    frozen=frozen_digest(frontend)
    attempts=[]; selected=None
    devices=[next(frontend.parameters()).device.index or 0] if next(frontend.parameters()).is_cuda else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            for size in (4,2):
                install_age_free_gate(frontend.event_former)
                frontend.event_former.state_gate.float()
                wrapper=Stage2Migration(frontend,heads.evidence,text_weight=.1,align_weight=1.,consistency_weight=.1,
                                        evidence_weight=.5,retain_weight=1.,tbptt_steps=32)
                batches=GateBatches(frontend,tokenizer,index,teachers,preflight_schedule(index,size))
                if devices:
                    torch.cuda.reset_peak_memory_stats()
                try:
                    for number,batch in enumerate(batches,1):
                        wrapper.zero_grad(set_to_none=True)
                        result=wrapper.token_step(batch)
                        if not torch.isfinite(result['loss']):
                            raise RuntimeError('native preflight nonfinite loss')
                        result['loss'].backward()
                        gate_grads=[p.grad for p in frontend.event_former.state_gate.parameters() if p.grad is not None]
                        head_grads=[p.grad for p in heads.evidence.parameters() if p.grad is not None]
                        if not gate_grads or not head_grads or not all(torch.isfinite(g).all() for g in gate_grads+head_grads):
                            raise RuntimeError('native preflight missing/nonfinite gate or Evidence gradient')
                        attempts.append(dict(micro_batch=size,trial=number,loss=float(result['loss'].detach()),
                            updates=batch['radar'].shape[1],actual_optimizer_steps=0,
                            peak_bytes=torch.cuda.max_memory_allocated() if devices else None,passed=True))
                    selected=size
                    break
                except torch.cuda.OutOfMemoryError:
                    attempts.append(dict(micro_batch=size,passed=False,reason='cuda-oom',actual_optimizer_steps=0))
                    wrapper.zero_grad(set_to_none=True)
                    del batches
                    result=None; batch=None; wrapper=None
                    torch.cuda.empty_cache()
                    if size==2:
                        raise RuntimeError('native micro-batch2 OOM; approved plan stops without batch1')
            if frozen_digest(frontend)!=frozen:
                raise RuntimeError('native preflight changed frozen source')
    finally:
        frontend.event_former.state_gate=original_gate
        heads.evidence.load_state_dict(original_evidence)
        frontend.zero_grad(set_to_none=True); heads.zero_grad(set_to_none=True)
    report=dict(micro_batch=selected,accumulation=16//selected if selected else None,attempts=attempts,
                actual_optimizer_steps=0,frozen_unchanged=True)
    write_json(output/'native-gate-preflight.json',report)
    return report


@torch.no_grad()
def query_keys(frontend,tokenizer):
    from ..stages.parrad_stage2 import default_parrad_event_templates
    from ..train import _tokenize_captions
    ids,mask=_tokenize_captions(tokenizer,tuple(default_parrad_event_templates()[i][0] for i in range(14)))
    device=next(frontend.parameters()).device
    ids,mask=ids.to(device),mask.to(device)
    return frontend.semantic_projection(frontend.event_former.adapter.encode_text(ids,mask),mask).float().cpu()


@torch.no_grad()
def retrieval_gate(frontend,tokenizer,index,teachers,output):
    device=next(frontend.parameters()).device
    prompt=frontend.event_former.build_prompt_cache(1)
    queries=query_keys(frontend,tokenizer)
    runs=defaultdict(list)
    for run in index['runs']:
        if run['role']=='dev':
            runs[run['record_uid']].append(run)
    predictions=[]
    teacher_records={r['uid']:r for r in teachers['records']}
    for row in index['records']:
        if row['role']!='dev':
            continue
        record=torch.load(row['path'],map_location='cpu',weights_only=False)
        keys=torch.load(teacher_records[row['uid']]['path'],map_location='cpu',weights_only=False)
        for run in runs[row['uid']]:
            radar=record['radar'][run['start']:run['end']].to(device)
            mask=record['radar_mask'][run['start']:run['end']].to(device)
            state=frontend.event_former.initial_state(1)
            for i in range(len(radar)):
                state=frontend.event_former.stream_update(state,radar[i:i+1],1.,mask[i:i+1],prompt).event_tokens
            commit=frontend.event_former.commit_refine(radar.flatten(0,1)[None],mask.flatten()[None],prompt).event_tokens
            teacher=keys[f"{run['uid']}:{run['end']-run['start']}"]
            predictions.append(dict(uid=run['uid'],label=run['label'],metadata=row['metadata'],
                stream_key=frontend.semantic_projection(state)[0].float().cpu(),
                commit_key=frontend.semantic_projection(commit)[0].float().cpu(),
                teacher_stream_key=teacher['stream_key'],teacher_commit_key=teacher['commit_key']))
        print(json.dumps(dict(phase='retrieval-gate',recording=row['uid'],events=len(runs[row['uid']]))),flush=True)
    metrics={}; passed=True
    groups={'dev_regression':[r for r in predictions if r['metadata']['role']=='dev_regression'],
            'dev':predictions}
    if not groups['dev_regression']:
        del groups['dev_regression']
    for subject,sensor in sorted({(r['metadata']['subject_id'],r['metadata']['sensor_id']) for r in predictions}):
        groups[f'{subject}/{sensor}']=[r for r in predictions if (r['metadata']['subject_id'],r['metadata']['sensor_id'])==(subject,sensor)]
    for name,rows in groups.items():
        metrics[name]={}
        for path in ('stream','commit'):
            labels=torch.tensor([r['label'] for r in rows])
            new=fixed_query_retrieval(torch.stack([r[f'{path}_key'] for r in rows]),labels,queries)
            old=fixed_query_retrieval(torch.stack([r[f'teacher_{path}_key'] for r in rows]),labels,queries)
            deltas={key:None if new[key] is None or old[key] is None else new[key]-old[key]
                    for key in ('macro_map','macro_r_at_1')}
            tolerance=.01 if name in ('dev','dev_regression') else .03
            eligible=all(value is not None and value>=-tolerance-1e-10 for value in deltas.values())
            passed &= eligible
            metrics[name][path]=dict(current=new,original=old,delta=deltas,passed=eligible)
    torch.save(dict(predictions=predictions,queries=queries),Path(output)/'retrieval-predictions.pt')
    report=dict(passed=bool(passed),metrics=metrics,frontend_digest=module_digest(frontend),
                predictions_sha256=file_sha(Path(output)/'retrieval-predictions.pt'))
    write_json(Path(output)/'retrieval-gate.json',report)
    return report


def train_gate(frontend,tokenizer,heads,index,teachers,schedule,output,*,micro_batch,seed,source):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(seed)
    install_age_free_gate(frontend.event_former)
    frontend.event_former.state_gate.float()
    migration=Stage2Migration(frontend,heads.evidence,text_weight=.1,align_weight=1.,consistency_weight=.1,
                              evidence_weight=.5,retain_weight=1.,tbptt_steps=32)
    frozen_before=frozen_digest(frontend)
    gate_initial=module_digest(frontend.event_former.state_gate)
    evidence_initial=module_digest(heads.evidence)
    optimizer=torch.optim.AdamW([
        dict(params=list(frontend.event_former.state_gate.parameters()),lr=1e-4),
        dict(params=list(heads.evidence.parameters()),lr=1e-3)],weight_decay=.01)
    scaler=torch.amp.GradScaler('cuda',init_scale=256.,enabled=next(frontend.parameters()).is_cuda)
    # Native Qwen decoder is already FP16; trainable gate/head stay FP32.
    # No global autocast may downcast the new heads or frozen projection.
    accumulation=16//micro_batch
    if micro_batch not in (2,4) or len(schedule)!=1000*accumulation:
        raise ValueError('locked micro-batch and effective batch16 schedule required')
    batches=GateBatches(frontend,tokenizer,index,teachers,schedule)
    log=(output/'training.jsonl').open('x',encoding='utf-8')
    code_sha=code_digest()
    checkpoints=[]
    def checkpoint(step):
        path=output/f'gate-step{step}.pt'
        torch.save(dict(schema_version=25,architecture='v25-age-free-tanh',stage='A',step=step,
            gate=frontend.event_former.state_gate.state_dict(),evidence=heads.evidence.state_dict(),
            optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),continuation=batches.continuation_state(),
            rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state_all(),seed=seed,
            source=source,code_digest=code_sha,frozen_digest=frozen_before,schedule=schedule,
            actual_optimizer_steps=step,micro_batch=micro_batch,accumulation=accumulation),path)
        saved=torch.load(path,map_location='cpu',weights_only=False)
        counts=[int(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v]
        if not counts or any(v!=step for v in counts) or saved['step']!=step:
            raise RuntimeError('saved AdamW readback step mismatch')
        checkpoints.append(dict(step=step,path=str(path),sha256=file_sha(path),adamw_counts=sorted(set(counts))))
    def on_step(row):
        log.write(json.dumps(dict(phase='A',**row))+'\n'); log.flush()
        print(json.dumps(dict(phase='A',**row)),flush=True)
        if row['step']==20:
            # ETA is a measured prefix of the approved1000, not extra training.
            print(json.dumps(dict(phase='A-estimate',remaining_steps=980,
                                 remaining_training_seconds=980*row['seconds'])),flush=True)
        if row['step'] in (20,100,500,1000):
            checkpoint(row['step'])
    try:
        result=accumulated_updates(migration,optimizer,batches,steps=1000,accumulation=accumulation,
                                   scaler=scaler,on_step=on_step)
    except BaseException as exc:
        save_failed_optimizer(optimizer,output,reason=repr(exc),scaler=scaler)
        raise
    finally:
        log.close()
    frozen_after=frozen_digest(frontend)
    if frozen_after!=frozen_before:
        raise RuntimeError('frozen frontend parameters changed')
    if module_digest(frontend.event_former.state_gate)==gate_initial or module_digest(heads.evidence)==evidence_initial:
        raise RuntimeError('expected gate/Evidence parameters did not update')
    report=dict(stage='A',actual_optimizer_steps=result['actual_optimizer_steps'],amp_skipped_steps=result['amp_skipped_steps'],
        checkpoints=checkpoints,frozen_before=frozen_before,frozen_after=frozen_after,source=source,code_digest=code_sha,
        final_gate_digest=module_digest(frontend.event_former.state_gate),final_evidence_digest=module_digest(heads.evidence))
    write_json(output/'stage2-readback.json',report)
    frontend.eval().requires_grad_(False)
    report['retrieval_gate']=retrieval_gate(frontend,tokenizer,index,teachers,output)
    return report
