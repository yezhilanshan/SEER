"""Matched successful-update budgets and fixed-panel HARrad task evaluation."""
from pathlib import Path
from contextlib import nullcontext
import math
import time
import numpy as np
import torch
from torch.nn import functional as F
from .joint_data import collate_origins, HORIZONS
from .training import classification_metrics
from .event_training import frozen_digest
from .preprocess import write_json, sha256
from .joint_audit import audit_job


def _record_mean(values, ids, valid=None):
    if valid is None: valid=torch.ones(len(values),dtype=torch.bool,device=values.device)
    groups=[values[(ids==rid)&valid].mean() for rid in ids[valid].unique()]
    return torch.stack(groups).mean() if groups else values.sum()*0


def joint_losses(output,batch,stats):
    result={}
    if 'mean' in output:
        errors=(output['mean']-batch['future']).square().mean(-1)
        scale=torch.as_tensor(stats['persistence_mse'],device=errors.device,dtype=errors.dtype)
        weights=errors.new_tensor([.1,.3,.6])
        result['motion']=_record_mean(((errors/scale)*weights).sum(1),batch['record_ids'])
    if 'logits' in output:
        ce=F.cross_entropy(output['logits'].float(),batch['labels'],ignore_index=-100,reduction='none')/math.log(12)
        result['semantic']=_record_mean(ce,batch['record_ids'],batch['labels']>=0)
    result['total']=sum(result.values())
    return result


def _device_batch(batch,device):
    return {key:value.to(device) if isinstance(value,torch.Tensor) else value for key,value in batch.items()}


@torch.no_grad()
def evaluate(model,records,stats,panel,batch_size,device):
    model.eval(); collected={}
    for start in range(0,len(panel),batch_size):
        selected=panel[start:start+batch_size]
        b=_device_batch(collate_origins(records,selected[:,:2],stats),device)
        output=model(b['cells'],b['mask'],b['current'])
        arrays={key:value.detach().float().cpu().numpy() for key,value in output.items() if key in ('mean','logits')}
        arrays.update({key:b[key].cpu().numpy() for key in ('future','current','previous','labels','crossing','future_unknown','future_mixed')})
        for key,value in arrays.items(): collected.setdefault(key,[]).append(value)
    result={key:np.concatenate(values) for key,values in collected.items()}
    result['panel']=panel.copy()
    return result, evaluate_arrays(result,records,stats)


def _forecast_metrics(mean,arrays,stats,selector):
    truth=arrays['future'].astype('float64'); current=arrays['current'].astype('float64')
    error=np.square(mean-truth).mean(2); base=np.square(current[:,None]-truth).mean(2)
    logvar=stats['logvar'].astype('float64')
    nll=(.5*(np.square(mean-truth)*np.exp(-logvar)+logvar+math.log(2*math.pi))).mean(2)
    rids=arrays['panel'][:,0]; rows={}
    for rid in sorted(set(rids[selector].tolist())):
        chosen=selector&(rids==rid)
        mse=error[chosen].mean(0); persistence=base[chosen].mean(0)
        rows[str(rid)]={'n':int(chosen.sum()),'mse':mse.tolist(),'nll':nll[chosen].mean(0).tolist(),
                        'persistence_mse':persistence.tolist(),
                        'skill':np.divide(persistence-mse,persistence,out=np.zeros(3),where=persistence>0).tolist()}
    if not rows: return None
    return {'n':int(selector.sum()),'records':rows,
            **{key:np.mean([r[key] for r in rows.values()],axis=0).tolist() for key in ('mse','nll','persistence_mse','skill')}}


def evaluate_arrays(arrays,records,stats):
    motion=arrays['panel'][:,2].astype(bool); semantic=arrays['panel'][:,3].astype(bool)
    rids=arrays['panel'][:,0]
    current=arrays['current'].astype('float64')
    persistence=np.repeat(current[:,None],3,axis=1)
    trend=current[:,None]+np.array(HORIZONS)[None,:,None]*(current-arrays['previous'])[:,None]
    result={'baselines':{'persistence':_forecast_metrics(persistence,arrays,stats,motion),
                         'linear_trend':_forecast_metrics(trend,arrays,stats,motion)}}
    if 'mean' in arrays:
        result['motion']=_forecast_metrics(arrays['mean'].astype('float64'),arrays,stats,motion)
        result['motion_groups']={}
        for index,horizon in enumerate(HORIZONS):
            groups={'cross_instance':arrays['crossing'][:,index],
                    'known_interior':(~arrays['crossing'][:,index])&(~arrays['future_unknown'][:,index])&(~arrays['future_mixed'][:,index])&(arrays['labels']>=0),
                    'unknown_future':arrays['future_unknown'][:,index],
                    'mixed_future':arrays['future_mixed'][:,index]}
            result['motion_groups'][str(horizon)]={}
            for name,mask in groups.items():
                scores=_forecast_metrics(arrays['mean'].astype('float64'),arrays,stats,motion&mask)
                result['motion_groups'][str(horizon)][name]=None if scores is None else {
                    'n':scores['n'],'mse':scores['mse'][index],'skill':scores['skill'][index]}
    if 'logits' in arrays:
        if np.any(arrays['labels'][semantic]<0): raise ValueError('invalid semantic panel')
        by_record={str(rid):classification_metrics(arrays['logits'][semantic&(rids==rid)],arrays['labels'][semantic&(rids==rid)])
                   for rid in sorted(set(rids[semantic].tolist()))}
        result['semantic']={'n':int(semantic.sum()),'records':by_record,
                            'record_macro_f1':float(np.mean([r['macro_f1'] for r in by_record.values()])),
                            'record_accuracy':float(np.mean([r['accuracy'] for r in by_record.values()])),
                            'pooled':classification_metrics(arrays['logits'][semantic],arrays['labels'][semantic])}
    return result


def fit_job(model,records,stats,schedule,panel,config,output,device='cuda'):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    if len(schedule)!=config['steps']: raise ValueError('schedule budget mismatch')
    trainable=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(trainable,lr=config['learning_rate'],weight_decay=config['weight_decay'])
    amp=config['amp'] and str(device)=='cuda'
    scaler=torch.amp.GradScaler('cuda',init_scale=256,enabled=amp)
    before=frozen_digest(model); attempts=0; started=time.time(); history=[]
    parameters={name:{'shape':list(p.shape),'n':p.numel(),'trainable':p.requires_grad} for name,p in model.named_parameters()}
    write_json(output/'parameters.json',parameters)
    torch.save({'learned':model.learned_state(),'step':0,'config':config},output/'step-0000.pt')
    if str(device)=='cuda': torch.cuda.reset_peak_memory_stats()
    for step,origins in enumerate(schedule,1):
        if any(records[int(i)]['role']!='train' for i,_ in origins): raise ValueError('nontrain schedule')
        batch=_device_batch(collate_origins(records,origins,stats),device)
        for retry in range(config['max_batch_attempts']):
            model.train(); optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16,cache_enabled=False) if amp else nullcontext():
                output_values=model(batch['cells'],batch['mask'],batch['current'])
                losses=joint_losses(output_values,batch,stats); loss=losses['total']
            if not torch.isfinite(loss): raise RuntimeError('nonfinite joint loss')
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(trainable,1.)
            old_scale=scaler.get_scale(); scaler.step(optimizer); scaler.update(); attempts+=1
            if scaler.get_scale()>=old_scale:
                if not torch.isfinite(norm): raise RuntimeError('nonfinite joint gradient')
                break
        else: raise RuntimeError('joint AMP retry budget exceeded')
        row={'step':step,'attempts':attempts,**{name:float(value.detach()) for name,value in losses.items()},
             'seconds':time.time()-started}
        if step==1 or step%10==0:
            history.append(row); write_json(output/'status.json',{'phase':'training',**row,'total_steps':len(schedule)})
        if step in config['checkpoints']:
            torch.save({'learned':model.learned_state(),'optimizer':optimizer.state_dict(),'scaler':scaler.state_dict(),
                        'step':step,'attempts':attempts,'config':config},output/f'step-{step:04d}.pt')
        if step%100==0: print(f"{model.arm} step {step}/{len(schedule)} loss={row['total']:.4f}",flush=True)
    write_json(output/'history.json',history)
    write_json(output/'status.json',{'phase':'validation','successful_steps':len(schedule),'attempts':attempts})
    arrays,metrics=evaluate(model,records,stats,panel,config['evaluation_batch_size'],device)
    np.savez_compressed(output/'validation-final.npz',**arrays)
    audit_job(output/'validation-final.npz',metrics,records,stats)
    torch.save({'learned':model.learned_state(),'step':len(schedule),'attempts':attempts,'config':config},output/'final.pt')
    saved=torch.load(output/'final.pt',map_location=device,weights_only=False); model.load_learned(saved['learned'])
    again,repeat=evaluate(model,records,stats,panel,config['evaluation_batch_size'],device)
    gap=max(float(np.max(np.abs(arrays[key]-again[key]))) for key in ('mean','logits') if key in arrays)
    if gap!=0: raise RuntimeError(f'joint final replay mismatch {gap}')
    unchanged=before==frozen_digest(model)
    if not unchanged: raise RuntimeError('frozen Qwen parameters changed')
    result={'successful_steps':len(schedule),'attempts':attempts,'metrics':metrics,'replay_max_abs':gap,
            'frozen_unchanged':unchanged,'trainable_parameters':sum(p.numel() for p in trainable),
            'final_sha256':sha256(output/'final.pt'),'seconds':time.time()-started,
            'peak_cuda_allocated':torch.cuda.max_memory_allocated() if str(device)=='cuda' else 0,
            'validation_unique_origins':len(panel),'test_signal_values_read':False}
    result['independent_verified']=True
    write_json(output/'complete.json',result); write_json(output/'status.json',{'phase':'complete','successful_steps':len(schedule)})
    return result
