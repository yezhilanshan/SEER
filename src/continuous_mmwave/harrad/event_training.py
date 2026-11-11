"""Frozen-backbone semantic warmup, complete epochs and independent prediction audits."""
from pathlib import Path
from contextlib import nullcontext
import hashlib
import json
import time
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from .event_data import EventDataset,event_collate
from .event_semantic import semantic_loss
from .training import classification_metrics
from .preprocess import write_json,sha256
from .verification import verify_predictions


def frozen_digest(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            h.update(name.encode()); h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def evaluate_semantic(model,dataset,batch_size,device,zero=False):
    model.eval(); groups={}
    with torch.no_grad():
        for b in DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=event_collate,num_workers=0):
            cells=b['cells'].to(device)
            if zero: cells=torch.zeros_like(cells)
            result=model(cells,b['mask'].to(device))
            slots={'end':{'logits':result['final_logits'],'valid':torch.ones(len(cells),dtype=torch.bool,device=device)}}
            slots.update({f'u{k}':v for k,v in result['prefix'].items()})
            for name,part in slots.items():
                valid=part['valid'].cpu().numpy(); idx=np.flatnonzero(valid)
                if not len(idx): continue
                group=groups.setdefault(name,{'logits':[],'labels':[],'subjects':[],'uids':[]})
                group['logits'].append(part['logits'].detach().float().cpu().numpy()[valid])
                group['labels'].extend(b['labels'].numpy()[valid].tolist())
                group['subjects'].extend(b['subjects'][i] for i in idx)
                group['uids'].extend(b['uids'][i] for i in idx)
    arrays={}; metrics={}
    for name,group in groups.items():
        arr={k:np.concatenate(v) if k=='logits' else np.array(v) for k,v in group.items()}
        arrays[name]=arr; metrics[name]=classification_metrics(arr['logits'],arr['labels'],arr['subjects'])
    return metrics,arrays


def _update(model,batch,opt,scaler,weights,device,amp,max_attempts):
    cells=batch['cells'].to(device); mask=batch['mask'].to(device); labels=batch['labels'].to(device)
    for attempt in range(1,max_attempts+1):
        model.train(); opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.float16) if amp else nullcontext():
            result=model(cells,mask); loss=semantic_loss(result,labels,weights)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite semantic loss')
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        old=scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale()>=old:
            if not torch.isfinite(norm): raise RuntimeError('nonfinite unscaled semantic gradient')
            return float(loss.detach()),attempt,float(norm)
    raise RuntimeError('semantic optimizer exceeded AMP retry budget')


def overfit_semantic(model,rows,cfg,output,device):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    train=[r for r in rows if r['role']=='train']; selected=[next(r for r in train if r['label']==k) for k in range(12)]
    ds=EventDataset(selected); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-3)
    amp=cfg['amp'] and device=='cuda'; scaler=torch.amp.GradScaler('cuda',init_scale=256,enabled=amp)
    weights=torch.ones(12,device=device); before=frozen_digest(model); history=[]; started=time.time(); attempts=0
    for step in range(1,301):
        ids=np.random.default_rng(1701+(step-1)//3).permutation(12)[((step-1)%3)*4:((step-1)%3+1)*4]
        _,tries,_=_update(model,event_collate([ds[int(i)] for i in ids]),opt,scaler,weights,device,amp,3); attempts+=tries
        if step%10==0:
            metrics,_=evaluate_semantic(model,ds,4,device)
            row={'step':step,'ce':metrics['end']['ce'],'accuracy':metrics['end']['accuracy']}; history.append(row)
            write_json(output/'status.json',row)
            print(f"semantic overfit {step}: accuracy={row['accuracy']:.3f} CE={row['ce']:.4f}",flush=True)
            if row['accuracy']>=.99 and row['ce']<=.3: break
    passed=row['accuracy']>=.99 and row['ce']<=.3
    unchanged=before==frozen_digest(model)
    result={'passed':passed and unchanged,'steps':step,'attempts':attempts,'history':history,'seconds':time.time()-started,
            'frozen_base_unchanged':unchanged,'selected_uids':[r['uid'] for r in selected],'formal_weights_reused':False}
    write_json(output/'result.json',result)
    if not result['passed']: raise RuntimeError('semantic 12-instance overfit gate failed')
    return result


def fit_semantic(model,rows,cfg,output,device='cuda'):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    train_rows=[r for r in rows if r['role']=='train']; val_rows=[r for r in rows if r['role']=='validation']
    if any(r['role']=='test' for r in rows) or {r['subject'] for r in train_rows}&{r['subject'] for r in val_rows}: raise ValueError('role/subject leakage')
    train=EventDataset(train_rows); val=EventDataset(val_rows)
    count=np.bincount([r['label'] for r in train_rows],minlength=12)
    if np.any(count==0): raise ValueError('all classes required in train')
    weights=1/count; weights/=weights.mean(); weights=torch.tensor(weights,dtype=torch.float32,device=device)
    trainable=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    amp=cfg['amp'] and device=='cuda'; scaler=torch.amp.GradScaler('cuda',init_scale=256,enabled=amp)
    before=frozen_digest(model); started=time.time(); steps=attempts=exposures=0; history=[]; best=-1
    torch.save({'learned':model.learned_state(),'epoch':0,'config':cfg},output/'initial.pt')
    write_json(output/'protocol.json',{'config':cfg,'train_instances':len(train),'validation_instances':len(val),'frozen_digest':before,
               'trainable_parameters':sum(p.numel() for p in trainable),'class_weights':weights.cpu().tolist()})
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,cfg['epochs']+1):
        order=np.random.default_rng(cfg['seed']*10000+epoch).permutation(len(train)).tolist()
        loader=DataLoader(Subset(train,order),batch_size=cfg['batch_size'],shuffle=False,collate_fn=event_collate,num_workers=0)
        total=0.; seen=0
        for batch_index,b in enumerate(loader):
            loss,tries,norm=_update(model,b,opt,scaler,weights,device,amp,cfg['max_batch_attempts'])
            steps+=1; attempts+=tries; n=len(b['labels']); exposures+=n; seen+=n; total+=loss*n
            if batch_index%10==0:
                write_json(output/'status.json',{'phase':'training','epoch':epoch,'total_epochs':cfg['epochs'],'batch':batch_index+1,
                           'batches':len(loader),'successful_steps':steps,'attempts':attempts,'exposures':exposures,
                           'loss':loss,'gradient_norm':norm,'updated_unix':time.time()})
        if seen!=len(train): raise RuntimeError('incomplete semantic epoch')
        metrics,arrays=evaluate_semantic(model,val,cfg['evaluation_batch_size'],device)
        row={'epoch':epoch,'mean_batch_loss':total/seen,'validation':metrics,'successful_steps':steps,'attempts':attempts,
             'exposures':exposures,'elapsed_seconds':time.time()-started}
        history.append(row)
        for slot,arr in arrays.items(): np.savez_compressed(output/f'validation-epoch-{epoch:03d}-{slot}.npz',**arr)
        saved={'learned':model.learned_state(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),
               'epoch':epoch,'config':cfg,'frozen_digest':before}
        torch.save(saved,output/'last.pt')
        if metrics['end']['macro_f1']>best:
            best=metrics['end']['macro_f1']; best_epoch=epoch; torch.save(saved,output/'best.pt')
            for slot,arr in arrays.items(): np.savez_compressed(output/f'validation-best-{slot}.npz',**arr)
        write_json(output/'history.json',history)
        write_json(output/'status.json',{'phase':'epoch_complete','epoch':epoch,'total_epochs':cfg['epochs'],
                   'best_epoch':best_epoch,'best_macro_f1':best,'validation':metrics,'updated_unix':time.time()})
        print(f"semantic epoch{epoch}/{cfg['epochs']} end F1={metrics['end']['macro_f1']:.4f}",flush=True)
    saved=torch.load(output/'best.pt',map_location=device,weights_only=False); model.load_learned(saved['learned'])
    replay,replayed=evaluate_semantic(model,val,cfg['evaluation_batch_size'],device)
    gap=0.; independent={}
    for slot,arr in replayed.items():
        path=output/f'validation-best-{slot}.npz'
        with np.load(path,allow_pickle=False) as old:
            gap=max(gap,float(np.abs(old['logits']-arr['logits']).max()))
            if not np.array_equal(old['uids'],arr['uids']) or not np.array_equal(old['logits'].argmax(1),arr['logits'].argmax(1)): raise RuntimeError('semantic replay mismatch')
        independent[slot]=verify_predictions(path,history[best_epoch-1]['validation'][slot])
    if gap>1e-4: raise RuntimeError(f'semantic replay gap {gap}')
    write_json(output/'independent-metrics.json',independent)
    zero_metrics,zero_arrays=evaluate_semantic(model,val,cfg['evaluation_batch_size'],device,zero=True)
    for slot,arr in zero_arrays.items(): np.savez_compressed(output/f'validation-zero-radar-{slot}.npz',**arr)
    unchanged=before==frozen_digest(model)
    if not unchanged: raise RuntimeError('frozen Qwen changed during training')
    result={'epochs_completed':cfg['epochs'],'successful_steps':steps,'attempts':attempts,'exposures':exposures,
            'best_epoch':best_epoch,'validation':replay,'zero_radar_validation':zero_metrics,'replay_max_abs':gap,
            'frozen_base_unchanged':unchanged,'frozen_digest':before,'best_sha256':sha256(output/'best.pt'),
            'trainable_parameters':sum(p.numel() for p in trainable),'peak_cuda_allocated':torch.cuda.max_memory_allocated() if device=='cuda' else 0,
            'seconds':time.time()-started,'test_signal_values_read':False}
    write_json(output/'complete.json',result)
    return result


def evaluate_continuous(model,records,output,device='cuda'):
    requests=[]; model.eval()
    with torch.no_grad():
        for record in records:
            if record['role']!='validation': continue
            x=np.load(record['cells_path'],mmap_mode='r'); schedule={}
            for r in record['instances']:
                for slot,size in [('end',r['stop']-r['start']),('u3',6),('u6',12),('u12',24)]:
                    if r['stop']-r['start']<size: continue
                    desired=r['start']+size; boundary=desired//2*2
                    schedule.setdefault(boundary,[]).append((slot,r,desired))
            state=model.former.initial_state(1); observed=torch.zeros(1,dtype=torch.long,device=device)
            for s in range(0,len(x)-1,2):
                cells=torch.from_numpy(np.array(x[s:s+2]))[None].to(device)
                state,observed=model.advance(cells,torch.ones(1,2,dtype=torch.bool,device=device),state,observed)
                boundary=s+2
                if boundary not in schedule: continue
                scores=model.read(state)[0].float().cpu().numpy()
                for slot,r,desired in schedule[boundary]:
                    requests.append({'slot':slot,'logits':scores,'label':r['label'],'subject':record['subject'],
                                     'uid':f"{record['file']}:{r['start']}:{r['stop']}",'actual_observation_end':boundary,'requested_end':desired})
    metrics={}
    for slot in ('end','u3','u6','u12'):
        group=[r for r in requests if r['slot']==slot]
        arr={'logits':np.stack([r['logits'] for r in group]),'labels':np.array([r['label'] for r in group]),
             'subjects':np.array([r['subject'] for r in group]),'uids':np.array([r['uid'] for r in group]),
             'actual_observation_end':np.array([r['actual_observation_end'] for r in group]),'requested_end':np.array([r['requested_end'] for r in group])}
        if not np.all(arr['actual_observation_end']<=arr['requested_end']): raise RuntimeError('future leakage in continuous readout')
        metrics[slot]=classification_metrics(arr['logits'],arr['labels'],arr['subjects'])
        np.savez_compressed(Path(output)/f'continuous-no-reset-{slot}.npz',**arr)
    write_json(Path(output)/'continuous-no-reset-metrics.json',{'metrics':metrics,'end_alignment':'last completed 2-frame update at or before requested cutoff; at most one frame earlier','test_signal_values_read':False})
    return metrics
