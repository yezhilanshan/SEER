"""Stage-specific adapters with full new-state export and exact bounded resumes."""
import json
from pathlib import Path
import numpy as np
import torch
from continuous_mmwave.harrad.preprocess import write_json
from ..training import frozen_digest
from .core import set_phase,new_state,load_new_state,read_prefix

MODES=(('E1',0),('E4',1),('E3',2),('E4',2),('E3',3),('E4',3))


def scheduled_mode(step,phase,lengths):
    if phase=='alignment': return lengths[(step-1)%len(lengths)],'E0',0
    if phase!='latent': raise ValueError('unknown phase')
    arm,rounds=MODES[(step-1)%len(MODES)]
    return lengths[((step-1)//len(MODES))%len(lengths)],arm,rounds


def load_stage(path, model, source, optimizer=None):
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if saved['schema']!=1 or saved['source']!=source: raise ValueError('stage checkpoint source mismatch')
    set_phase(model,saved['phase']); load_new_state(model,saved['new_state'])
    if frozen_digest(model)!=saved['frozen_digest']: raise ValueError('stage frozen parameter mismatch')
    if optimizer is not None:
        if saved['optimizer_names']!=[n for n,p in model.named_parameters() if p.requires_grad]:
            raise ValueError('optimizer parameter mismatch')
        optimizer.load_state_dict(saved['optimizer']); torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available() and saved['cuda_rng']: torch.cuda.set_rng_state_all(saved['cuda_rng'])
    return saved


def train_stage(model, examples, prompt, grammar, output_dir, *, steps, phase, source,
                seed=42, learning_rate=1e-4, resume=None):
    if not examples or type(steps) is not int or steps<=0: raise ValueError('explicit positive training budget required')
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=False); successful=0
    try:
        set_phase(model,phase); torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
        parameters=[p for p in model.parameters() if p.requires_grad]
        opt=torch.optim.AdamW(parameters,lr=learning_rate,weight_decay=.01)
        scaler=torch.amp.GradScaler('cuda',enabled=model.backend.device.type=='cuda',init_scale=64)
        signature=dict(source=source,phase=phase,seed=seed,learning_rate=learning_rate,
                       example_uids=[e['uid'] for e in examples],model_settings=model.settings)
        if resume:
            saved=load_stage(resume,model,signature,opt); successful=saved['step']; scaler.load_state_dict(saved['scaler'])
            if successful>=steps: raise ValueError('resume total budget must exceed saved step')
        digest=frozen_digest(model); lengths=sorted({e['seconds'] for e in examples})
        write_json(out/'source.json',signature)
        def save(step):
            payload=dict(schema=1,step=step,phase=phase,source=signature,new_state=new_state(model),
                optimizer=opt.state_dict(),optimizer_names=[n for n,p in model.named_parameters() if p.requires_grad],
                scaler=scaler.state_dict(),frozen_digest=digest,torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
            path=out/f'step-{step:06d}.pt'; temp=path.with_suffix('.pt.tmp'); torch.save(payload,temp); temp.replace(path)
            return path
        if not resume: save(0)
        for step in range(successful+1,steps+1):
            length,arm,rounds=scheduled_mode(step,phase,lengths); pool=[e for e in examples if e['seconds']==length]
            sample=pool[int(np.random.default_rng(seed+step).integers(len(pool)))]
            model.train(); opt.zero_grad(set_to_none=True)
            prefix=read_prefix(model,sample['memory'],prompt,arm,rounds)['prefix']
            answer=torch.tensor([grammar.answer_ids(sample['reference'])],device=prompt.device)
            loss=model.backend.answer_loss(prefix,prompt,answer)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite stage loss')
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
            if not torch.isfinite(norm): raise RuntimeError('nonfinite stage gradient')
            scale=scaler.get_scale(); scaler.step(opt); scaler.update()
            if scaler.get_scale()<scale: raise RuntimeError('stage optimizer update skipped')
            successful=step
            record=dict(step=step,uid=sample['uid'],phase=phase,arm=arm,rounds=rounds,
                        loss=float(loss.detach()),gradient_norm=float(norm))
            with (out/'training.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
            write_json(out/'status.json',dict(status='running',successful_steps=step,**record))
            if step in (100,300,1000) or step==steps: checkpoint=save(step)
            if step%10==0 or step==1: print(json.dumps(record),flush=True)
        if frozen_digest(model)!=digest: raise RuntimeError('frozen stage parameters changed')
        write_json(out/'status.json',dict(status='complete',successful_steps=successful,frozen_unchanged=True))
        return checkpoint
    except BaseException as error:
        write_json(out/'status.json',dict(status='failed',successful_steps=successful,error=repr(error))); raise
