"""Small explicit-budget training runs, protected sources and exact optimizer resumes."""
import hashlib
import random
from pathlib import Path
import numpy as np
import torch
from continuous_mmwave.harrad.preprocess import write_json


def frozen_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if p.requires_grad: continue
        h.update(f'{name}:{tuple(p.shape)}:{p.dtype}'.encode())
        h.update(p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def load_checkpoint(path, model, source, optimizer=None, training_settings=None):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['schema']!=1 or payload['source']!=source: raise ValueError('checkpoint source mismatch')
    if payload['model_settings']!=model.settings or payload['frozen_digest']!=frozen_digest(model):
        raise ValueError('checkpoint architecture/frozen model mismatch')
    if training_settings is not None and payload['training_settings']!=training_settings:
        raise ValueError('resume training settings mismatch')
    model.load_learned(payload['learned'])
    if optimizer is not None:
        names=[n for n,p in model.named_parameters() if p.requires_grad]
        if payload['optimizer_names']!=names: raise ValueError('optimizer parameter names mismatch')
        optimizer.load_state_dict(payload['optimizer'])
        torch.set_rng_state(payload['torch_rng'])
        if torch.cuda.is_available() and payload['cuda_rng']: torch.cuda.set_rng_state_all(payload['cuda_rng'])
    return payload


def train_steps(model, examples, output_dir, *, steps, source, seed=42, learning_rate=1e-4,
                weight_decay=.01, grad_clip=1., checkpoint_every=10, resume=None):
    if type(steps) is not int or steps<1 or not len(examples): raise ValueError('explicit positive steps and nonempty examples required')
    output=Path(output_dir); output.mkdir(parents=True,exist_ok=False)
    settings=dict(seed=seed,learning_rate=learning_rate,weight_decay=weight_decay,grad_clip=grad_clip,rounds=model.rounds)
    start=0; successful=0
    try:
        write_json(output/'status.json',dict(status='running',phase='setup',successful_steps=0))
        write_json(output/'source.json',source)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        trainable=[p for p in model.parameters() if p.requires_grad]
        optimizer=torch.optim.AdamW(trainable,lr=learning_rate,weight_decay=weight_decay)
        digest=frozen_digest(model)
        scaler=torch.amp.GradScaler('cuda',enabled=model.backend.device.type=='cuda',init_scale=64)
        if resume:
            saved=load_checkpoint(resume,model,source,optimizer,settings); start=saved['step']; successful=start
            scaler.load_state_dict(saved['scaler'])
            if start>=steps: raise ValueError('resume target must exceed saved successful step')
        for step in range(start+1,steps+1):
            index=int(np.random.default_rng(seed+step).integers(len(examples)))
            sample=examples[index]; model.train(); optimizer.zero_grad(set_to_none=True)
            result=model(sample['memory'],sample['prompt_ids'])
            loss=model.backend.answer_loss(result['latents'],sample['prompt_ids'],sample['answer_ids'])
            if not torch.isfinite(loss): raise RuntimeError('nonfinite caption loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(trainable,grad_clip)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite gradient; no optimizer update performed')
            scale=scaler.get_scale(); scaler.step(optimizer); scaler.update()
            if scaler.get_scale()<scale: raise RuntimeError('optimizer update skipped')
            successful=step
            row=dict(step=step,uid=sample['uid'],loss=float(loss.detach()),gradient_norm=float(norm),rounds=model.rounds)
            import json
            with (output/'training.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
            write_json(output/'status.json',dict(status='running',**row))
            if step%checkpoint_every==0 or step==steps:
                payload=dict(schema=1,step=step,learned=model.learned_state(),optimizer=optimizer.state_dict(),
                             optimizer_names=[n for n,p in model.named_parameters() if p.requires_grad],
                             scaler=scaler.state_dict(),source=source,model_settings=model.settings,training_settings=settings,
                             frozen_digest=digest,torch_rng=torch.get_rng_state(),
                             cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
                destination=output/f'step-{step:06d}.pt'; temporary=destination.with_suffix('.pt.tmp')
                torch.save(payload,temporary); temporary.replace(destination)
            print(f'latent_memory step={step} loss={float(loss.detach()):.6f}',flush=True)
        if frozen_digest(model)!=digest: raise RuntimeError('frozen parameters changed')
        write_json(output/'status.json',dict(status='complete',successful_steps=steps,initial_step=start,frozen_unchanged=True))
    except BaseException as error:
        write_json(output/'status.json',dict(status='failed',error=repr(error),successful_steps=successful,last_completed_step=successful))
        raise
    return output/f'step-{steps:06d}.pt'
