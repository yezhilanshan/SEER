"""Efficient windows with unchanged online availability; frozen-label reporting."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from .model import SplitBiGRU
from .streaming import correction_windows,_views,_update,_delays
from .loss import combo_loss
from .diagnostics import metric,masked_metric,metrics_from_confusion,reliability_diagnostics,correction_gains,collar_mask


@torch.no_grad()
def fast_replay(net,md,*,window=512,hop=171,window_batch=8,labels=None):
    if net.training or md.ndim!=3 or md.shape[0]!=1 or md.shape[1]==0 or window_batch<1:
        raise ValueError('eval and one nonempty recording required')
    schedule=correction_windows(md.shape[1],window,hop)
    if isinstance(net,SplitBiGRU):
        forward,cache,_=net.forward_chunk(md)
    else:
        forward=torch.full((1,md.shape[1],9),float('nan'),device=md.device);cache=None
    result=_views(forward)
    event_changes=[]
    y=None if labels is None else np.asarray(labels)
    for offset in range(0,len(schedule),window_batch):
        group=schedule[offset:offset+window_batch]
        inp=torch.cat([cache[:,a:b] if cache is not None else md[:,a:b] for a,b in group],0)
        predictions=net.correct(inp) if cache is not None else net(inp)
        for j,(a,b) in enumerate(group):
            if y is not None:
                old=result['final_logits'][0,a:b].cpu().numpy();new=predictions[j].cpu().numpy()
                gains=correction_gains(old,new,y[a:b])
                event_changes.append({'start_frame':a,'end_frame_exclusive':b,
                                      'emission_frame':b-1,**{k:v for k,v in gains.items() if k!='per_class'}})
            _update(result,predictions[j:j+1],a,b)
    result['event_changes']=event_changes
    return _delays(result,.09)


class EvaluationPool:
    def __init__(self,rows):
        self.rows=sorted(rows,key=lambda r:r['recording_id']);self.data=[]
        if not self.rows:raise ValueError('empty evaluation sources')
        for r in self.rows:
            p=Path(r['path'])
            if 'npz_sha256' in r and hashlib.sha256(p.read_bytes()).hexdigest()!=r['npz_sha256']:
                raise ValueError('evaluation recording digest mismatch')
            with np.load(p) as d:self.data.append((d['md'].astype(np.float32),d['labels'].astype(np.int64),d['timestamps'].copy()))


@torch.no_grad()
def evaluate(net,pool,*,output=None,diagnostics=False,window=512,hop=171):
    net.eval();device=next(net.parameters()).device
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=False)
    views={};reports=[];loss_sum,loss_frames=0.,0;branch_losses={};groups={}
    for r,(md,y,timestamps) in zip(pool.rows,pool.data):
        x=torch.from_numpy(md).unsqueeze(0).to(device)
        replay=fast_replay(net,x,window=window,hop=hop,labels=y if diagnostics else None)
        logits={v:replay[k][0].cpu().numpy() for v,k in (('forward','forward_logits'),('first','first_logits'),('final','final_logits'))}
        covered=replay['available_mask'];valid=y!=-100
        metrics={'full':{v:metric(a,y) for v,a in logits.items()},
                 'matched':{v:masked_metric(a,y,covered) for v,a in logits.items()}}
        coverage=float((valid&covered).sum()/valid.sum()) if valid.any() else None
        record={'recording_id':r['recording_id'],'subject_id':r['subject_id'],'sensor_id':r.get('sensor_id','unknown'),
                'coverage':coverage,'metrics':metrics}
        # Loss on valid frames with a defined model output; report branch sums explicitly.
        yy=torch.from_numpy(y).unsqueeze(0).to(device)
        if isinstance(net,SplitBiGRU):
            losses={}
            for name,key in (('forward','forward_logits'),('corrected','final_logits')):
                loss,_=combo_loss(replay[key],yy);losses[name]=float(loss)
            weight=int(valid.sum());value=sum(losses.values())
        else:
            target=yy.clone();target[:,~covered]=-100;weight=int((valid&covered).sum())
            losses={'baseline':float(combo_loss(replay['final_logits'],target)[0])} if weight else {'baseline':0.}
            value=losses['baseline']
        loss_sum+=value*weight;loss_frames+=weight
        for name,value0 in losses.items():branch_losses[name]=branch_losses.get(name,0.)+value0*weight
        for scope,items in metrics.items():
            for name,met in items.items():
                key=scope+'/'+name;conf=np.asarray(met['confusion_matrix'],dtype=np.int64)
                views[key]=views.get(key,np.zeros((9,10),dtype=np.int64))+conf
                for groupkey in ('subject_id','sensor_id'):
                    g=r.get(groupkey,'unknown');groups.setdefault(groupkey,{}).setdefault(g,{})
                    groups[groupkey][g][key]=groups[groupkey][g].get(key,np.zeros((9,10),dtype=np.int64))+conf
        if diagnostics:
            boundaries=r.get('alignment',{}).get('command_frame_indices',[])
            tail_start=boundaries[-1] if boundaries else len(y)
            record['label_reliability']={v:reliability_diagnostics(a,y,boundaries,tail_start=tail_start) for v,a in logits.items()}
            if isinstance(net,SplitBiGRU):
                record['correction_gains']={v:correction_gains(logits['forward'],logits[v],y) for v in ('first','final')}
                stable=collar_mask(len(y),boundaries,10)
                record['gains_stable']=correction_gains(logits['forward'],logits['final'],y,mask=stable)
                record['gains_boundary']=correction_gains(logits['forward'],logits['final'],y,mask=~stable)
            record['delay_summary']={v:{'corrected_frames':int(np.isfinite(replay[v+'_delay_s']).sum()),
                'mean_s':float(np.nanmean(replay[v+'_delay_s'])) if covered.any() else None,
                'p95_s':float(np.nanpercentile(replay[v+'_delay_s'],95)) if covered.any() else None} for v in ('first','final')}
        if output is not None:
            dest=output/(r['recording_id']+'.npz')
            np.savez_compressed(dest,**{k+'_logits':v for k,v in logits.items()},labels=y,timestamps=timestamps,
                available_mask=covered,correction_counts=replay['correction_counts'],
                first_end_frame=replay['first_end_frame'],final_end_frame=replay['final_end_frame'],
                first_delay_s=replay['first_delay_s'],final_delay_s=replay['final_delay_s'])
            record['prediction_file']=dest.name;record['prediction_sha256']=hashlib.sha256(dest.read_bytes()).hexdigest()
            if diagnostics:
                (output/(r['recording_id']+'-changes.json')).write_text(json.dumps(replay['event_changes'],indent=2))
        reports.append(record)
    aggregate={key:metrics_from_confusion(conf) for key,conf in views.items()}
    grouped={gkey:{g:{k:metrics_from_confusion(c) for k,c in fields.items()} for g,fields in vals.items()} for gkey,vals in groups.items()}
    result={'metrics':aggregate,'groups':grouped,'recordings':reports,'coverage':
        aggregate['matched/final']['valid_frames']/aggregate['full/final']['valid_frames'] if aggregate['full/final']['valid_frames'] else None,
        'loss':loss_sum/loss_frames if loss_frames else None,
        'branch_losses':{k:v/loss_frames for k,v in branch_losses.items()} if loss_frames else {},
        'loss_valid_frame_weight':loss_frames,'loss_aggregation':'valid_frame_weighted_recording_combo_means',
        'score':aggregate['matched/final']['weighted_f1'],'model_eval_mode':True,'window':window,'hop':hop,
        'diagnostics_not_for_checkpoint_selection':True}
    if output is not None:(output/'metrics.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
