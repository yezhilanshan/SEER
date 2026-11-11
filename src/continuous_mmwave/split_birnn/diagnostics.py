"""Predesignated approximate-label and correction diagnostics; never tune truth."""
from __future__ import annotations
import numpy as np
import torch
from .streaming import classification_metrics


def metric(logits, labels):
    return classification_metrics(torch.as_tensor(np.asarray(logits)),torch.as_tensor(np.asarray(labels)))


def shift_labels(labels, shift):
    y=np.asarray(labels); out=np.full_like(y,-100)
    if abs(shift)>=len(y): return out
    if shift>0: out[shift:]=y[:-shift]
    elif shift<0: out[:shift]=y[-shift:]
    else: out[:]=y
    return out


def collar_mask(frames, boundaries, collar):
    mask=np.ones(frames,dtype=bool)
    for b in boundaries:
        if 0<int(b)<frames: mask[max(0,int(b)-collar):min(frames,int(b)+collar+1)]=False
    return mask


def masked_metric(logits,labels,mask):
    y=np.asarray(labels).copy();y[~np.asarray(mask,dtype=bool)]=-100
    return metric(logits,y)


def reliability_diagnostics(logits,labels,boundaries,*,tail_start):
    y=np.asarray(labels); before=y.copy(); n=len(y)
    tail=np.arange(n)>=max(0,min(n,int(tail_start)))
    result={'primary':metric(logits,y),
            'shifts':{str(k):metric(logits,shift_labels(y,k)) for k in (-10,-6,-3,3,6,10)},
            'collars':{str(k):masked_metric(logits,y,collar_mask(n,boundaries,k)) for k in (3,6,10)},
            'tail':masked_metric(logits,y,tail),'non_tail':masked_metric(logits,y,~tail),
            'shift_semantics':'positive_moves_labels_later_vacated_frames_ignored',
            'boundary_semantics':'internal_CSV_command_boundaries_inclusive_collar_first_prompt_excluded',
            'diagnostic_only_not_used_for_checkpoint_selection':True}
    result['primary_labels_unchanged']=bool(np.array_equal(before,y))
    return result


def comparison_views(split_logits,baseline_logits,labels):
    a,b=np.asarray(split_logits),np.asarray(baseline_logits);y=np.asarray(labels)
    common=np.isfinite(a).all(-1)&np.isfinite(b).all(-1)
    valid=y!=-100;n=int(valid.sum())
    return {'common_valid_frames':int((common&valid).sum()),
            'split_coverage':float((np.isfinite(a).all(-1)&valid).sum()/n) if n else None,
            'baseline_coverage':float((np.isfinite(b).all(-1)&valid).sum()/n) if n else None,
            'matched':{'split':masked_metric(a,y,common),'baseline':masked_metric(b,y,common)},
            'full':{'split':metric(a,y),'baseline':metric(b,y)}}


def correction_gains(forward,corrected,labels,*,mask=None):
    f,c=np.asarray(forward),np.asarray(corrected);y=np.asarray(labels)
    valid=(y!=-100)&np.isfinite(f).all(-1)&np.isfinite(c).all(-1)
    if mask is not None: valid&=np.asarray(mask,dtype=bool)
    fp=np.nan_to_num(f).argmax(-1);cp=np.nan_to_num(c).argmax(-1)
    improved=valid&(fp!=y)&(cp==y);harmed=valid&(fp==y)&(cp!=y)
    def counts(selection):
        n=int(selection.sum());i=int((improved&selection).sum());h=int((harmed&selection).sum())
        return {'valid_frames':n,'improved':i,'harmed':h,'improve_rate':i/n if n else None,
                'harm_rate':h/n if n else None,'net_gain':(i-h)/n if n else None}
    return {**counts(valid),'per_class':[counts(valid&(y==k)) for k in range(9)]}


def metrics_from_confusion(confusion):
    c=np.asarray(confusion,dtype=np.int64); support=c.sum(1);tp=np.diag(c[:,:9]);pred=c[:,:9].sum(0)
    f1=np.divide(2*tp,support+pred,out=np.zeros(9),where=(support+pred)>0);n=int(support.sum())
    return {'valid_frames':n,'accuracy':float(tp.sum()/n) if n else None,
            'weighted_f1':float((f1*support).sum()/n) if n else None,
            'macro_f1_all_9':float(f1.mean()) if n else None,'class_f1':f1.tolist(),
            'class_support':support.tolist(),'confusion_matrix':c.tolist(),'unavailable_frames':int(c[:,9].sum())}
