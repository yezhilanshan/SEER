"""Explicit task losses, successful-update budgets, and immutable checkpoints."""
from __future__ import annotations

import math
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .contract import make_joint_batch
from .model import trainable_manifest
from ..v26.source import file_sha, state_digest, write_json


def _record_mean(values,ids,valid):
    return torch.stack([values[valid & ids.eq(key)].mean() for key in ids[valid].unique()]).mean()


def joint_losses(outputs: dict,teacher: dict,batch: dict,fit_stats: dict,arm: str) -> dict:
    if arm not in ('M','A','B','AB'):
        raise ValueError('unknown arm')
    base=next(iter(outputs.values()))
    zero=base.sum()*0
    motion=semantic=preserve=zero
    ids=batch['recording_ids']
    motion_count=semantic_count=0
    if arm in ('M','A','AB'):
        R=fit_stats['R'].to(base.device).float()
        if R.shape != (3,) or not torch.isfinite(R).all() or (R<=0).any():
            raise ValueError('positive finite R required')
        error=(outputs['mean'].float()-batch['target'].detach().float()).square().mean(-1)
        per=(error/R*error.new_tensor([.1,.3,.6])).sum(-1)
        motion=_record_mean(per,ids,torch.ones_like(ids,dtype=torch.bool));motion_count=len(ids)
    valid=batch['semantic'].ge(0)
    if arm in ('B','AB') and valid.any():
        safe_target=batch['semantic'].clone();safe_target[~valid]=0
        per=F.cross_entropy(outputs['logits'].float(),safe_target,reduction='none')/math.log(14)
        semantic=_record_mean(per,ids,valid)
        teacher_emb=teacher['embedding'].detach().float()
        teacher_logprob=teacher['logits'].detach().float().log_softmax(-1)
        per_pres=1-F.cosine_similarity(outputs['old_embedding'].float(),teacher_emb,dim=-1)
        per_pres+=F.kl_div(outputs['old_logits'].float().log_softmax(-1),teacher_logprob,
                          reduction='none',log_target=True).sum(-1)
        preserve=_record_mean(per_pres,ids,valid);semantic_count=int(valid.sum())
    total=motion+semantic+.1*preserve
    if not torch.isfinite(total):
        raise ValueError('nonfinite joint loss')
    return dict(total=total,motion=motion,semantic=semantic,preserve=preserve,
                motion_count=motion_count,semantic_count=semantic_count,has_signal=bool(motion_count or semantic_count))


def gradient_diagnostic(model: nn.Module,losses: dict,scale: float = 1.) -> dict:
    parameters=tuple(model.shared_adapter.parameters())
    vectors=[]
    for key in ('motion','semantic'):
        loss=losses[key]
        grads=torch.autograd.grad(loss*scale,parameters,retain_graph=True,allow_unused=True) if loss.requires_grad else (None,)*len(parameters)
        vectors.append(torch.cat([(torch.zeros_like(p) if g is None else g/scale).detach().float().flatten() for p,g in zip(parameters,grads)]))
    a,b=vectors;na,nb=a.norm(),b.norm()
    return dict(motion_norm=float(na),semantic_norm=float(nb),
                cosine=float((a@b)/(na*nb)) if na>0 and nb>0 else None)


def validate_optimizer_checkpoint(payload: dict,model: nn.Module,step: int):
    expected=[name for name,p in model.named_parameters() if p.requires_grad]
    if payload.get('arm')!=model.arm or payload.get('seed')!=model.seed or payload.get('actual_optimizer_steps')!=step:
        raise ValueError('checkpoint arm/seed/update identity mismatch')
    learned=payload.get('learned_state',{})
    if set(learned)!=set(expected):raise ValueError('checkpoint trainable parameter whitelist mismatch')
    for name,p in model.named_parameters():
        if p.requires_grad and (p.shape!=learned[name].shape or not torch.isfinite(learned[name]).all()):
            raise ValueError('checkpoint learned tensor mismatch')
    reported=payload.get('parameter_steps',{})
    if reported!=({name:step for name in expected} if step else {}):
        raise ValueError('checkpoint optimizer parameter names/counters incomplete')
    optimizer=payload.get('optimizer_state',{})
    groups=optimizer.get('param_groups',[])
    identifiers=[identifier for group in groups for identifier in group['params']]
    state=optimizer.get('state',{})
    if len(identifiers)!=len(expected) or len(set(identifiers))!=len(expected):
        raise ValueError('optimizer parameter group incomplete')
    if set(state)!=(set(identifiers) if step else set()):
        raise ValueError('optimizer state incomplete')
    if step:
        for identifier in identifiers:
            if int(state[identifier].get('step',-1))!=step:
                raise ValueError('optimizer true step mismatch')
            if any(not torch.isfinite(v).all() for v in state[identifier].values() if torch.is_tensor(v)):
                raise ValueError('optimizer state nonfinite')


def optimizer_attempt(loss,optimizer,parameters,scaler=None,clip=1.):
    if not torch.isfinite(loss):
        raise ValueError('nonfinite loss; no update')
    enabled=scaler is not None and scaler.is_enabled()
    if enabled:
        scaler.scale(loss).backward();scaler.unscale_(optimizer)
    else:loss.backward()
    if not any(p.grad is not None for p in parameters):
        raise RuntimeError('disconnected optimizer parameters')
    norm=torch.nn.utils.clip_grad_norm_(parameters,clip)
    if enabled:
        previous=scaler.get_scale();scaler.step(optimizer);scaler.update()
        return bool(torch.isfinite(norm) and scaler.get_scale()>=previous),float(norm)
    if not torch.isfinite(norm):
        raise RuntimeError('nonfinite gradients; no optimizer step')
    optimizer.step()
    return True,float(norm)


def validate_training_options(config: dict,smoke: bool = False):
    expected=dict(lr=1e-4,weight_decay=.01,clip_norm=1.)
    if any(config.get(k)!=v for k,v in expected.items()):
        raise ValueError('locked optimizer budget differs')
    if smoke:
        if not 1 <= config['steps'] <=5 or config['max_attempts'] < config['steps']:
            raise ValueError('smoke budget exceeds five updates')
    elif config['steps']!=1000 or config['max_attempts']!=1100:
        raise ValueError('formal budget requires 1000 successful steps/1100 attempts')


def frozen_digest(model):
    return {key:state_digest(part) for key,part in
            (('former',model.former),('old_projection',model.old_projection),('old_classifier',model.old_classifier))}


def train_joint_job(model,records,stats,schedule,config,path: Path) -> dict:
    smoke=config.get('smoke',False);validate_training_options(config,smoke)
    if len(schedule)!=config['steps'] or any(len(row)!=2 for row in schedule):
        raise ValueError('locked schedule requires two examples per successful step')
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    device=str(model.fit_mean.device);amp=device.startswith('cuda')
    model.set_stats(stats);model.train()
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=config['lr'],weight_decay=config['weight_decay'])
    scaler=torch.amp.GradScaler('cuda',enabled=amp,init_scale=256.)
    initial_frozen=frozen_digest(model)
    write_json(path/'parameters.json',trainable_manifest(model))
    write_json(path/'schedule.json',schedule)
    if amp:torch.cuda.reset_peak_memory_stats()
    start=time.perf_counter();step=attempt=0;logs=[]
    initial=model.learned_state()

    def save(current):
        counts={name:int(optimizer.state[p].get('step',0)) for name,p in model.named_parameters() if p.requires_grad and p in optimizer.state}
        if current and any(count!=current for count in counts.values()):
            raise RuntimeError('optimizer parameter counters disagree')
        torch.save(dict(learned_state=model.learned_state(),optimizer_state=optimizer.state_dict(),
                        scaler_state=scaler.state_dict(),arm=model.arm,seed=model.seed,
                        actual_optimizer_steps=current,attempts=attempt,parameter_steps=counts,
                        stats=stats,training_options=config,schedule_sha256=file_sha(path/'schedule.json'),
                        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if amp else None),
                   path/f'step{current:04d}.pt')
    save(0)
    while step < config['steps'] and attempt < config['max_attempts']:
        batch=make_joint_batch(records,stats,schedule[step],device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            teacher=model.teacher_prefix(batch['F'],batch['mask']) if model.arm in ('B','AB') else {}
            outputs=model.forward_prefix(batch['F'],batch['mask'],batch['m_ref'])
            losses=joint_losses(outputs,teacher,batch,stats,model.arm)
        if not losses['has_signal']:
            raise RuntimeError('locked batch contains no valid task signal')
        diagnostic=gradient_diagnostic(model,losses,scaler.get_scale()) if step in (0,99,299,999) else None
        success,norm=optimizer_attempt(losses['total'],optimizer,parameters,scaler,config['clip_norm'])
        attempt+=1
        if success:step+=1
        row=dict(step=step,attempt=attempt,success=success,gradient_norm=norm if math.isfinite(norm) else None,
                 **{key:float(losses[key].detach()) for key in ('total','motion','semantic','preserve')},
                 motion_samples=losses['motion_count'],semantic_samples=losses['semantic_count'],
                 gradient=diagnostic,scaler_scale=scaler.get_scale())
        logs.append(row)
        if success and (step in (100,300,1000) or step==config['steps']):save(step)
        if success and (step%50==0 or step==config['steps']):
            write_json(path/'progress.json',dict(step=step,attempts=attempt,latest=row,seconds=time.perf_counter()-start))
            print(f'{model.arm} seed={model.seed} step={step} loss={row["total"]:.6f}',flush=True)
        if amp and torch.cuda.max_memory_allocated()>9*1024**3:
            raise RuntimeError('allocated memory exceeds locked 9GiB resource gate')
    if step!=config['steps']:
        write_json(path/'failure.json',dict(reason='successful-step budget not reached',step=step,attempts=attempt))
        raise RuntimeError('attempt budget exhausted before successful-step budget')
    if frozen_digest(model)!=initial_frozen:
        raise RuntimeError('frozen teacher/former changed')
    changed=[name for name,value in model.learned_state().items() if not torch.equal(value,initial[name])]
    if not changed:
        raise RuntimeError('optimizer did not change any learned parameters')
    result=dict(arm=model.arm,seed=model.seed,actual_optimizer_steps=step,attempts=attempt,
                seconds=time.perf_counter()-start,peak_allocated=torch.cuda.max_memory_allocated() if amp else None,
                changed=changed,frozen_unchanged=True,logs=logs,
                checkpoint_sha256={p.name:file_sha(p) for p in sorted(path.glob('step*.pt'))})
    write_json(path/'completed.json',result)
    return result
