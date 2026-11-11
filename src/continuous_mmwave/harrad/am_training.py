"""Matched-budget semantic 2x2 training with immutable reference and full curves."""
from contextlib import nullcontext
from pathlib import Path
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from .am_modules import AnchoredContrastive
from .am_audit import audit_semantic
from .joint_training import _record_mean,_device_batch
from .joint_data import collate_origins
from .training import classification_metrics
from .event_training import frozen_digest
from .preprocess import write_json,sha256


def alignment_losses(model,out,batch,bank,config):
    valid=batch['labels']>=0
    ce=F.cross_entropy(out['logits'].float(),batch['labels'],ignore_index=-100,reduction='none')/math.log(12)
    semantic=_record_mean(ce,batch['record_ids'],valid)
    values=AnchoredContrastive(config['temperature'])(out['query'],out['reference'],batch['labels'],bank['vectors'],bank['labels'])
    contrast=_record_mean(values,batch['record_ids'],valid)
    return {'semantic':semantic,'semantic_ce_raw':semantic*math.log(12),'contrastive':contrast,
            'total':semantic+config['alignment_weight']*contrast if model.use_alignment else semantic}


def _scores(arrays,bank):
    scores=arrays['logits']; labels=arrays['labels']; ids=arrays['panel'][:,0]
    def summarize(chosen):
        if not chosen.any(): return None
        by_record={str(rid):classification_metrics(scores[chosen&(ids==rid)],labels[chosen&(ids==rid)]) for rid in sorted(set(ids[chosen].tolist()))}
        return {'n':int(chosen.sum()),'records':by_record,'pooled':classification_metrics(scores[chosen],labels[chosen]),
                'record_macro_f1':float(np.mean([r['macro_f1'] for r in by_record.values()])),
                'record_accuracy':float(np.mean([r['accuracy'] for r in by_record.values()]))}
    result=summarize(np.ones(len(labels),dtype=bool)); lengths=arrays['lengths']
    result['prefix_groups']={name:summarize((lengths>=lo)&(lengths<=hi)) for name,lo,hi in (('u1_4',1,4),('u5_16',5,16),('u17_32',17,32))}
    result['positive_cosine']=float((arrays['query']*arrays['reference']).sum(1).mean())
    keys=bank['vectors'].detach().float().cpu().numpy(); bank_labels=bank['labels'].cpu().numpy()
    similarities=arrays['query']@keys.T; allowed=labels[:,None]!=bank_labels[None]
    if not allowed.any(1).all(): raise ValueError('evaluation query has no different-class reference')
    result['negative_cosine']=float(np.mean([s[m].mean() for s,m in zip(similarities,allowed)]))
    result['query_std_rms']=float(np.sqrt(arrays['query'].astype('float64').var(0).mean()))
    return result


@torch.no_grad()
def evaluate_alignment(model,records,stats,panel,batch_size,device,bank):
    model.eval(); pieces={}
    for start in range(0,len(panel),batch_size):
        selected=panel[start:start+batch_size]; batch=_device_batch(collate_origins(records,selected[:,:2],stats),device)
        if (batch['labels']<0).any(): raise ValueError('semantic evaluation panel has unknown target')
        out=model(batch['cells'],batch['mask'],batch['current'])
        arrays={k:out[k].detach().float().cpu().numpy() for k in ('logits','query','reference')}
        arrays.update(labels=batch['labels'].cpu().numpy(),lengths=batch['mask'].sum(1).cpu().numpy())
        for key,value in arrays.items(): pieces.setdefault(key,[]).append(value)
    arrays={key:np.concatenate(values) for key,values in pieces.items()}; arrays['panel']=panel.copy()
    metrics=_scores(arrays,bank); audit_semantic(arrays,metrics,records,bank)
    return arrays,metrics


def fit_alignment(model,records,stats,schedule,panels,bank,config,output,device='cuda'):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    if len(schedule)!=config['steps']: raise ValueError('alignment schedule budget mismatch')
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=config['learning_rate'],weight_decay=config['weight_decay'])
    amp=config['amp'] and str(device)=='cuda'; scaler=torch.amp.GradScaler('cuda',init_scale=256,enabled=amp)
    before=frozen_digest(model); bank_before=bank['vectors'].detach().cpu().clone()
    write_json(output/'parameters.json',{n:{'shape':list(p.shape),'n':p.numel(),'trainable':p.requires_grad} for n,p in model.named_parameters()})
    curves={}; audits={}; last_arrays={}; history=[]; attempts=0; started=time.time()
    if str(device)=='cuda': torch.cuda.reset_peak_memory_stats()
    def snapshot(step):
        torch.save({'learned':model.learned_state(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                    'step':step,'attempts':attempts,'config':config},output/f'step-{step:04d}.pt')
        curves[str(step)]={}; audits[str(step)]={}
        for role,panel in panels.items():
            arrays,metrics=evaluate_alignment(model,records,stats,panel,config['evaluation_batch_size'],device,bank)
            np.savez_compressed(output/f'predictions-step-{step:04d}-{role}.npz',**arrays)
            curves[str(step)][role]=metrics; audits[str(step)][role]=audit_semantic(arrays,metrics,records,bank)
            if step==len(schedule): last_arrays[role]=arrays
        write_json(output/'curves.json',curves); write_json(output/'independent-audit.json',audits)
    write_json(output/'status.json',{'phase':'evaluation','successful_steps':0,'attempts':0})
    snapshot(0)
    for step,origins in enumerate(schedule,1):
        if any(records[int(rid)]['role']!='train' for rid,_ in origins): raise ValueError('nontrain optimizer sample')
        batch=_device_batch(collate_origins(records,origins,stats),device)
        for retry in range(config['max_batch_attempts']):
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False) if amp else nullcontext():
                out=model(batch['cells'],batch['mask'],batch['current']); losses=alignment_losses(model,out,batch,bank,config)
            if not torch.isfinite(losses['total']): raise RuntimeError('nonfinite alignment loss')
            scaler.scale(losses['total']).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.); old=scaler.get_scale(); scaler.step(optimizer); scaler.update(); attempts+=1
            if scaler.get_scale()>=old:
                if not torch.isfinite(norm): raise RuntimeError('nonfinite alignment gradient')
                break
        else: raise RuntimeError('alignment AMP retry budget exceeded')
        if step==1 or step%10==0:
            row={'step':step,'attempts':attempts,'seconds':time.time()-started,**{k:float(v.detach()) for k,v in losses.items()}}
            history.append(row); write_json(output/'status.json',{'phase':'training','total_steps':len(schedule),**row})
        if step in config['checkpoints']:
            write_json(output/'status.json',{'phase':'evaluation','successful_steps':step,'attempts':attempts}); snapshot(step)
        if step%100==0: print(f'{model.arm} step {step}/{len(schedule)}',flush=True)
    write_json(output/'history.json',history)
    torch.save({'learned':model.learned_state(),'step':len(schedule),'attempts':attempts,'config':config},output/'final.pt')
    model.load_learned(torch.load(output/'final.pt',map_location=device,weights_only=False)['learned'])
    again,_=evaluate_alignment(model,records,stats,panels['validation'],config['evaluation_batch_size'],device,bank)
    gap=max(float(np.max(np.abs(again[k]-last_arrays['validation'][k]))) for k in ('logits','query','reference'))
    if gap!=0: raise RuntimeError(f'alignment final replay mismatch {gap}')
    unchanged=before==frozen_digest(model)
    if not unchanged or not torch.equal(bank_before,bank['vectors'].detach().cpu()): raise RuntimeError('frozen reference/Qwen/bank changed')
    off=None
    if model.use_modulation:
        model.modulation_enabled=False
        arrays,off=evaluate_alignment(model,records,stats,panels['validation'],config['evaluation_batch_size'],device,bank)
        np.savez_compressed(output/'predictions-final-modulation-off-validation.npz',**arrays)
        model.modulation_enabled=True
    result={'successful_steps':len(schedule),'attempts':attempts,'curves':curves,'replay_max_abs':gap,
            'modulation_off_validation':off,'frozen_unchanged':unchanged,'independent_verified':True,
            'trainable_parameters':sum(p.numel() for p in trainable),'final_sha256':sha256(output/'final.pt'),
            'seconds':time.time()-started,'peak_cuda_allocated':torch.cuda.max_memory_allocated() if str(device)=='cuda' else 0,
            'source_qwen_digest':model.source_qwen_digest,'test_signal_values_read':False}
    write_json(output/'complete.json',result); write_json(output/'status.json',{'phase':'complete','successful_steps':len(schedule)})
    return result
