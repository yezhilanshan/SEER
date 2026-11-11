"""Controlled complete-instance training, small-set gate and saved-weight replay."""
from __future__ import annotations

import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,Subset

from .data import ClipDataset,collate_clips,CLASSES
from .model import SegmentClassifier
from .preprocess import write_json,sha256
from .verification import verify_predictions


def seed_everything(seed: int, threads: int=4):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)


def classification_metrics(logits,labels,subjects=None):
    logits=np.asarray(logits,dtype=np.float64); labels=np.asarray(labels,dtype=np.int64)
    if logits.shape!=(len(labels),12) or not len(labels) or not np.isfinite(logits).all(): raise ValueError('invalid predictions')
    if np.any((labels<0)|(labels>=12)): raise ValueError('invalid class')
    predicted=logits.argmax(1)
    def rates(y,p,classes):
        cm=np.zeros((12,12),np.int64); np.add.at(cm,(y,p),1)
        tp=cm.diagonal(); precision=np.divide(tp,cm.sum(0),out=np.zeros(12),where=cm.sum(0)>0)
        recall=np.divide(tp,cm.sum(1),out=np.zeros(12),where=cm.sum(1)>0)
        f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(12),where=(precision+recall)>0)
        return cm,recall,f1[list(classes)].mean()
    cm,recall,f1=rates(labels,predicted,range(12))
    shift=logits-logits.max(1,keepdims=True)
    ce=np.log(np.exp(shift).sum(1))-shift[np.arange(len(labels)),labels]
    result={'n':len(labels),'accuracy':float(np.mean(predicted==labels)),'macro_f1':float(f1),'ce':float(ce.mean()),
            'per_class_recall':recall.tolist(),'confusion':cm.tolist()}
    for group,selector,classes in [('events',labels<6,range(6)),('gestures',labels>=6,range(6,12))]:
        result[group+'_macro_f1']=float(rates(labels[selector],predicted[selector],classes)[2]) if selector.any() else None
    if subjects is not None:
        subjects=np.asarray(subjects)
        result['subjects']={s:classification_metrics(logits[subjects==s],labels[subjects==s]) for s in sorted(set(subjects.tolist()))}
    return result


def evaluate(model,dataset,batch_size,device):
    model.eval(); predictions=[]; labels=[]; subjects=[]; uids=[]; ids=[]
    with torch.no_grad():
        for b in DataLoader(dataset,batch_size=batch_size,shuffle=False,collate_fn=collate_clips,num_workers=0):
            pred=model(b['x'].to(device),b['mask'].to(device))
            predictions.append(pred.cpu().numpy()); labels.extend(b['labels'].tolist())
            subjects.extend(b['subjects']); uids.extend(b['uids']); ids.extend(b['instance_ids'].tolist())
    arr={'logits':np.concatenate(predictions),'labels':np.array(labels,dtype=np.int64),'subjects':np.array(subjects),
         'uids':np.array(uids),'instance_ids':np.array(ids,dtype=np.int64)}
    return classification_metrics(arr['logits'],arr['labels'],arr['subjects']),arr


def _weights(rows,device):
    count=np.bincount([r['label'] for r in rows],minlength=12)
    if not np.all(count>0): raise ValueError('train must contain all 12 classes')
    w=1/count.astype(np.float64); w/=w.mean()
    return torch.tensor(w,dtype=torch.float32,device=device)


def _device(cfg):
    if cfg['device']=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    return torch.device(cfg['device'])


def overfit_gate(arm,rows,normalizer,cfg,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    train=[r for r in rows if r['role']=='train']; selected=[]
    for label in range(12): selected.append(next(r for r in train if r['label']==label))
    seed_everything(1701,cfg['threads']); device=_device(cfg)
    model=SegmentClassifier(arm,dropout=0).to(device)
    ds=ClipDataset(selected,normalizer,arm); batch=collate_clips([ds[i] for i in range(len(ds))])
    x=batch['x'].to(device); mask=batch['mask'].to(device); y=batch['labels'].to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=.005,weight_decay=0)
    history=[]; started=time.time()
    for step in range(1,301):
        model.train(); opt.zero_grad(set_to_none=True); logits=model(x,mask)
        loss=nn.functional.cross_entropy(logits,y)
        if not torch.isfinite(loss): raise RuntimeError('nonfinite overfit loss')
        loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm): raise RuntimeError('nonfinite overfit gradient')
        opt.step()
        if step%10==0:
            model.eval()
            with torch.no_grad():
                pred=model(x,mask); ce=float(nn.functional.cross_entropy(pred,y)); acc=float((pred.argmax(1)==y).float().mean())
            history.append({'step':step,'ce':ce,'accuracy':acc})
            write_json(output/'status.json',history[-1])
            if acc>=.99 and ce<=.2: break
    passed=acc>=.99 and ce<=.2
    result={'passed':passed,'arm':arm,'steps':step,'history':history,'selected_uids':[r['uid'] for r in selected],
            'seconds':time.time()-started,'accuracy':acc,'ce':ce,'formal_weights_reused':False}
    write_json(output/'result.json',result)
    if not passed: raise RuntimeError(f'{arm} 12-instance overfit gate failed: accuracy={acc} ce={ce}')
    return result


def train_run(arm,seed,rows,normalizer,cfg,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    seed_everything(seed,cfg.get('threads',4)); device=_device(cfg)
    train_rows=[r for r in rows if r['role']=='train']; val_rows=[r for r in rows if r['role']=='validation']
    if any(r['role']=='test' for r in rows): raise ValueError('test arrays forbidden')
    if {r['subject'] for r in train_rows}&{r['subject'] for r in val_rows}: raise ValueError('subject leakage')
    train=ClipDataset(train_rows,normalizer,arm); val=ClipDataset(val_rows,normalizer,arm)
    model=SegmentClassifier(arm,cfg['dropout']).to(device); weights=_weights(train_rows,device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    parameters=sum(p.numel() for p in model.parameters()); frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    torch.save({'model':model.state_dict(),'arm':arm,'seed':seed,'epoch':0,'config':cfg},output/'initial.pt')
    write_json(output/'protocol.json',{'arm':arm,'seed':seed,'config':cfg,'classes':CLASSES,'train_instances':len(train),'validation_instances':len(val),
               'trainable_parameters':parameters-frozen,'frozen_parameters':frozen,'class_weights':weights.cpu().tolist(),'test_signal_values_read':False})
    best=-1.; steps=0; exposures=0; clipped=0; started=time.time(); history=[]
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats()
    for epoch in range(1,cfg['epochs']+1):
        order=np.random.default_rng(seed*10000+epoch).permutation(len(train)).tolist()
        order_hash=__import__('hashlib').sha256(np.asarray(order,dtype=np.int64).tobytes()).hexdigest()
        loader=DataLoader(Subset(train,order),batch_size=cfg['batch_size'][arm],shuffle=False,collate_fn=collate_clips,num_workers=0)
        model.train(); loss_total=0.; epoch_n=0
        for batch_index,b in enumerate(loader):
            opt.zero_grad(set_to_none=True)
            logits=model(b['x'].to(device),b['mask'].to(device))
            loss=nn.functional.cross_entropy(logits,b['labels'].to(device),weight=weights)
            if not torch.isfinite(loss): raise RuntimeError('nonfinite train loss')
            loss.backward(); norm=nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'])
            if not torch.isfinite(norm): raise RuntimeError('nonfinite train gradient')
            clipped+=int(norm>cfg['grad_clip']); opt.step(); steps+=1
            n=len(b['labels']); exposures+=n; epoch_n+=n; loss_total+=float(loss.detach())*n
            if batch_index%20==0:
                write_json(output/'status.json',{'phase':'training','arm':arm,'seed':seed,'epoch':epoch,'epoch_batch':batch_index+1,
                           'epoch_batches':len(loader),'successful_steps':steps,'train_exposures':exposures,'updated_unix':time.time()})
        if epoch_n!=len(train): raise RuntimeError('incomplete epoch')
        validation,arr=evaluate(model,val,cfg['evaluation_batch_size'],device)
        train_metrics,_=evaluate(model,train,cfg['evaluation_batch_size'],device)
        row={'epoch':epoch,'train_weighted_batch_ce':loss_total/epoch_n,'train':train_metrics,'validation':validation,
             'successful_steps':steps,'train_exposures':exposures,'sample_order_sha256':order_hash,'elapsed_seconds':time.time()-started}
        history.append(row)
        np.savez_compressed(output/f'validation-epoch-{epoch:03d}.npz',**arr)
        checkpoint={'model':model.state_dict(),'optimizer':opt.state_dict(),'arm':arm,'seed':seed,'epoch':epoch,'config':cfg,
                    'normalizer':normalizer,'classes':CLASSES,'successful_steps':steps,'train_exposures':exposures}
        torch.save(checkpoint,output/'last.pt')
        if validation['macro_f1']>best:
            best=validation['macro_f1']; best_epoch=epoch
            torch.save(checkpoint,output/'best.pt'); np.savez_compressed(output/'validation-best.npz',**arr)
        write_json(output/'history.json',history)
        write_json(output/'status.json',{'phase':'epoch_complete','arm':arm,'seed':seed,'epoch':epoch,'total_epochs':cfg['epochs'],
                   'validation_macro_f1':validation['macro_f1'],'best_macro_f1':best,'best_epoch':best_epoch,
                   'successful_steps':steps,'train_exposures':exposures,'updated_unix':time.time()})
        print(f"{arm} seed{seed} epoch{epoch}/{cfg['epochs']} trainF1={train_metrics['macro_f1']:.4f} valF1={validation['macro_f1']:.4f}",flush=True)
    checkpoint=torch.load(output/'best.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model']); replay,replayed=evaluate(model,val,cfg['evaluation_batch_size'],device)
    with np.load(output/'validation-best.npz',allow_pickle=False) as saved:
        gap=float(np.max(np.abs(saved['logits']-replayed['logits'])))
        if gap>1e-5 or not np.array_equal(saved['logits'].argmax(1),replayed['logits'].argmax(1)):
            raise RuntimeError(f'best-weight replay failed: {gap}')
        if not np.array_equal(saved['uids'],replayed['uids']): raise RuntimeError('replay example order mismatch')
    independent=verify_predictions(output/'validation-best.npz',history[best_epoch-1]['validation'])
    write_json(output/'independent-metrics.json',independent)
    result={'arm':arm,'seed':seed,'epochs_completed':cfg['epochs'],'successful_steps':steps,'train_exposures':exposures,
            'best_epoch':best_epoch,'validation':replay,'replay_max_abs':gap,'trainable_parameters':parameters-frozen,
            'seconds':time.time()-started,'clipped_steps':clipped,'peak_cuda_allocated':torch.cuda.max_memory_allocated() if device.type=='cuda' else 0,
            'best_sha256':sha256(output/'best.pt'),'test_signal_values_read':False}
    result['independent_metrics_passed']=independent['passed']
    write_json(output/'complete.json',result)
    return result
