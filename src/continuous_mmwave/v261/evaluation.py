"""Common-population native heads, standardized readouts, and array verification."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .contract import HORIZONS, joint_schedule, make_joint_batch
from .training import joint_losses


def classification_metrics(target,logits):
    target=np.asarray(target);logits=np.asarray(logits)
    if logits.shape != (len(target),14) or not np.isfinite(logits).all():
        raise ValueError('finite fourteen-class logits required')
    valid=target>=0
    if not valid.any():return dict(samples=0,accuracy=None,macro_f1=None,confusion=np.zeros((14,14),int).tolist())
    if (target[valid]>=14).any():raise ValueError('class ID out of range')
    prediction=logits.argmax(-1)
    confusion=np.zeros((14,14),dtype=np.int64)
    np.add.at(confusion,(target[valid],prediction[valid]),1)
    diagonal=np.diag(confusion).astype(float)
    denominator=confusion.sum(0)+confusion.sum(1)
    f1=np.divide(2*diagonal,denominator,out=np.zeros(14),where=denominator>0)
    return dict(samples=int(valid.sum()),accuracy=float(diagonal.sum()/valid.sum()),macro_f1=float(f1.mean()),
                confusion=confusion.tolist(),per_class_f1=f1.tolist(),support=confusion.sum(1).tolist())


def _average(values):
    return float(np.mean(values)) if values else None


def aggregate_motion(rows):
    people={}
    for subject in sorted({r['subject'] for r in rows}):
        own=[r for r in rows if r['subject']==subject and r['samples']>0]
        if not own:continue
        values={key:_average([r[key] for r in own if r.get(key) is not None]) for key in ('mse','persistence_mse','trend_mse','nll')}
        values['skill']=1-values['mse']/values['persistence_mse'] if values['persistence_mse'] else None
        people[subject]=values
    return dict(by_subject=people,samples=sum(r['samples'] for r in rows),
                subject_macro={key:_average([r[key] for r in people.values() if r.get(key) is not None])
                               for key in ('mse','persistence_mse','trend_mse','nll','skill')})


def group_masks(record,horizon):
    selected=record.common_start.numpy()
    cross=np.asarray(record.groups.get(f'deadline_cross_h{horizon}',np.zeros(len(record.radar),bool)))[selected]
    mixed=np.asarray(record.groups.get(f'mixed_current_h{horizon}',np.zeros(len(record.radar),bool)))[selected]
    future=np.asarray(record.groups.get(f'mixed_future_h{horizon}',np.zeros(len(record.radar),bool)))[selected]
    return dict(all=np.ones(len(selected),bool),deadline_cross=cross,mixed_current=mixed,mixed_future=future,
                clean_interior=~(cross|mixed|future))


def reference_arrays(record,stats):
    t=record.common_start.numpy()
    motion=((record.motion_ref-stats['mean'])/stats['std']).numpy()
    future=t[:,None]+np.array(HORIZONS)[None,:]
    result=dict(selected=t,future=future,current=motion[t],target=motion[future],
                semantic=record.semantic_target[t].numpy())
    for h in HORIZONS:
        for name,mask in group_masks(record,h).items():result[f'group_{h}_{name}']=mask
    result['trend']=np.stack([motion[t]+h*(motion[t]-motion[np.maximum(t-1,0)]) for h in HORIZONS],1)
    return result


@torch.no_grad()
def predict_joint_record(model,record,stats,batch_size=64):
    device=str(model.fit_mean.device);amp=device.startswith('cuda')
    model.eval();result=reference_arrays(record,stats)
    values={}
    for start in range(0,len(record.common_start),batch_size):
        samples=[(0,int(t)) for t in record.common_start[start:start+batch_size]]
        batch=make_joint_batch([record],stats,samples,device)
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            output=model.forward_prefix(batch['F'],batch['mask'],batch['m_ref'])
        for key in ('mean','logits','old_logits','shared'):
            if key in output:values.setdefault(key,[]).append(output[key].float().cpu().numpy())
    result.update({key:np.concatenate(rows) for key,rows in values.items()})
    result['uid']=np.array(record.uid);result['subject']=np.array(record.subject)
    return result


def verify_prediction_array(actual,expected,arm):
    for key in ('selected','future','current','target','semantic'):
        if key not in actual or not np.array_equal(actual[key],expected[key]):
            raise ValueError(f'{key} mismatch')
    for key in expected:
        if key.startswith('group_') or key=='trend':
            if key not in actual or not np.array_equal(actual[key],expected[key]):raise ValueError(f'{key} mismatch')
    n=len(expected['selected'])
    shapes={}
    if arm in ('M','A','AB','probe'):shapes['mean']=(n,3,256)
    if arm in ('B','AB','probe'):shapes['logits']=(n,14)
    for key,shape in shapes.items():
        if key not in actual or actual[key].shape!=shape or not np.isfinite(actual[key]).all():
            raise ValueError(f'finite matching {key} required')
    for key in ('mean','logits','old_logits','shared'):
        if key in actual and not np.isfinite(actual[key]).all():raise ValueError(f'finite {key} required')


def summarize_joint(predictions,stats):
    motion_rows=[];semantic=[];legacy=[]
    logvar=stats['logvar'].detach().cpu().numpy().astype(np.float64)
    for row in predictions:
        subject=str(row['subject']);uid=str(row['uid'])
        if 'mean' in row:
            error=(row['mean'].astype(np.float64)-row['target'].astype(np.float64))**2
            per_mse=error.mean(-1)
            persistence=((row['current'].astype(np.float64)[:,None]-row['target'])**2).mean(-1)
            nll=.5*(math.log(2*math.pi)+logvar+error*np.exp(-logvar)).mean(-1)
            trend=((row['trend'].astype(np.float64)-row['target'])**2).mean(-1)
            for j,h in enumerate(HORIZONS):
                for group in ('all','deadline_cross','mixed_current','mixed_future','clean_interior'):
                    mask=row[f'group_{h}_{group}'];count=int(mask.sum())
                    motion_rows.append(dict(subject=subject,uid=uid,horizon=h,group=group,samples=count,
                          mse=float(per_mse[mask,j].mean()) if count else None,
                          persistence_mse=float(persistence[mask,j].mean()) if count else None,
                          trend_mse=float(trend[mask,j].mean()) if count else None,
                          nll=float(nll[mask,j].mean()) if count else None))
        for key,destination in (('logits',semantic),('old_logits',legacy)):
            if key in row:destination.append(dict(subject=subject,uid=uid,**classification_metrics(row['semantic'],row[key])))
    motion={f'h{h}':{group:aggregate_motion([r for r in motion_rows if r['horizon']==h and r['group']==group])
                      for group in ('all','deadline_cross','mixed_current','mixed_future','clean_interior')}
            for h in HORIZONS} if motion_rows else None
    def aggregate_class(rows):
        if not rows:return None
        people={}
        for person in sorted({r['subject'] for r in rows}):
            own=[r for r in rows if r['subject']==person and r['samples']]
            people[person]={key:_average([r[key] for r in own]) for key in ('accuracy','macro_f1')}
        return dict(records=rows,by_subject=people,
                    subject_macro={key:_average([r[key] for r in people.values() if r[key] is not None]) for key in ('accuracy','macro_f1')})
    return dict(motion=motion,motion_records=motion_rows,semantic=aggregate_class(semantic),legacy_reader=aggregate_class(legacy))


@torch.no_grad()
def extract_shared_features(adapter,records,stats,device='cpu'):
    features={}
    for record in records:
        values=[]
        for chunk in record.radar.split(256):
            Z=adapter(chunk.to(device))
            values.append(((Z.mean(1)-stats['mean'].to(device))/stats['std'].to(device)).cpu())
        features[record.uid]=torch.cat(values)
    return features


def fit_shared_probes(features,records,stats,seed,steps=200,device='cpu'):
    if steps!=200:raise ValueError('probe budget requires 200 updates')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed+1000)
        motion=nn.Linear(256,768);nn.init.zeros_(motion.weight);nn.init.zeros_(motion.bias)
        semantic=nn.Linear(256,14)
    motion.to(device);semantic.to(device)
    optimizers=[torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.01) for head in (motion,semantic)]
    schedule=joint_schedule(records,seed+5000,steps*32)
    original={uid:values.clone() for uid,values in features.items()}
    logs=[]
    for step in range(steps):
        samples=[pair for batch in schedule[step*32:(step+1)*32] for pair in batch]
        inputs=torch.stack([features[records[ri].uid][t] for ri,t in samples]).detach().to(device)
        # Probes use exactly the same physical targets and semantic mask, with
        # no recurrent prefixes or gradient connection to the source features.
        current=torch.stack([(records[ri].motion_ref[t]-stats['mean'])/stats['std'] for ri,t in samples]).to(device)
        target=torch.stack([torch.stack([(records[ri].motion_ref[t+h]-stats['mean'])/stats['std'] for h in HORIZONS]) for ri,t in samples]).to(device)
        batch=dict(target=target,semantic=torch.tensor([int(records[ri].semantic_target[t]) for ri,t in samples],device=device),
                   recording_ids=torch.tensor([ri for ri,t in samples],device=device))
        prediction=current[:,None]+motion(inputs).reshape(-1,3,256)
        output=dict(mean=prediction,logits=semantic(inputs))
        la=joint_losses(output,{},batch,stats,'A')['motion']
        valid=batch['semantic']>=0
        per=torch.nn.functional.cross_entropy(output['logits'],batch['semantic'].clamp_min(0),reduction='none')/math.log(14)
        lb=torch.stack([per[valid & batch['recording_ids'].eq(i)].mean() for i in batch['recording_ids'][valid].unique()]).mean()
        for opt,loss,head in zip(optimizers,(la,lb),(motion,semantic)):
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.);opt.step()
        logs.append(dict(motion=float(la.detach()),semantic=float(lb.detach())))
    if any(not torch.equal(features[uid],values) for uid,values in original.items()):raise RuntimeError('probe mutated source features')
    result=dict(motion_state={k:v.cpu() for k,v in motion.state_dict().items()},semantic_state={k:v.cpu() for k,v in semantic.state_dict().items()},
                actual_optimizer_steps=dict(motion=steps,semantic=steps),logs=logs,seed=seed,
                optimizer_states={key:opt.state_dict() for key,opt in zip(('motion','semantic'),optimizers)})
    validate_probe_checkpoint(result,seed)
    return result


def validate_probe_checkpoint(probe,seed):
    if probe.get('seed')!=seed or probe.get('actual_optimizer_steps')!=dict(motion=200,semantic=200):
        raise ValueError('probe seed/step identity mismatch')
    for key,width in (('motion',768),('semantic',14)):
        state=probe.get(key+'_state',{})
        if set(state)!= {'weight','bias'} or state['weight'].shape!=(width,256) or state['bias'].shape!=(width,):
            raise ValueError('probe parameter whitelist mismatch')
        if not all(torch.isfinite(x).all() for x in state.values()):raise ValueError('probe weights nonfinite')
        optimizer=probe.get('optimizer_states',{}).get(key,{})
        identifiers=[i for group in optimizer.get('param_groups',[]) for i in group['params']]
        slots=optimizer.get('state',{})
        if len(identifiers)!=2 or len(set(identifiers))!=2 or set(slots)!=set(identifiers):
            raise ValueError('probe optimizer state incomplete')
        if any(int(slots[i].get('step',-1))!=200 for i in identifiers):
            raise ValueError('probe optimizer step mismatch')


@torch.no_grad()
def predict_probe_record(probe,features,record,stats):
    result=reference_arrays(record,stats)
    x=features[record.uid][record.common_start].float()
    result['mean']=result['current'][:,None]+torch.nn.functional.linear(x,probe['motion_state']['weight'],probe['motion_state']['bias']).reshape(-1,3,256).numpy()
    result['logits']=torch.nn.functional.linear(x,probe['semantic_state']['weight'],probe['semantic_state']['bias']).numpy()
    result['subject']=np.array(record.subject);result['uid']=np.array(record.uid)
    return result


def verify_joint_outputs(root: Path) -> dict:
    """Inspect saved predictions; runner additionally reconstructs targets and weights."""
    files=sorted(Path(root).rglob('*.npz'))
    for path in files:
        with np.load(path,allow_pickle=False) as row:
            for key in row.files:
                if np.issubdtype(row[key].dtype,np.number) and not np.isfinite(row[key]).all():
                    raise ValueError(f'nonfinite saved array: {path}:{key}')
    return dict(files=len(files),finite=True)
