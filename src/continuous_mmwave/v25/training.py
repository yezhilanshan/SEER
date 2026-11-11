"""Stage2/3 v2.5 supervision. No Verify phase, age target, or GT online input."""
from __future__ import annotations

import math
import hashlib
import json
import time
import torch
from torch.nn import functional as F


def save_failed_optimizer(optimizer,output,*,reason,scaler=None):
    from pathlib import Path
    from .prepare import write_json
    from .checkpoint import file_sha
    output=Path(output)
    path=output/'failure-optimizer.pt'
    with path.open('xb') as stream:
        torch.save(dict(optimizer=optimizer.state_dict(),scaler=None if scaler is None else scaler.state_dict(),reason=reason),stream)
    saved=torch.load(path,map_location='cpu',weights_only=False)
    counts=[int(value['step']) for value in saved['optimizer']['state'].values() if 'step' in value]
    report=dict(reason=reason,max_adamw_parameter_step=max(counts,default=0),
                all_parameter_steps=counts,optimizer_sha256=file_sha(path))
    write_json(output/'failure-readback.json',report)
    return report


def accumulated_updates(module,optimizer,rows,*,steps,accumulation,scaler=None,on_step=None,start_step=0):
    """Count independent AdamW updates, including strict AMP overflow failure.

    Exhausting rows or any invalid loss/gradient fails before another update.
    Callbacks run after the successful update and can save the exact trajectory.
    """
    if steps<=0 or accumulation<=0 or start_step<0:
        raise ValueError('positive optimizer budget and accumulation required')
    iterator=iter(rows)
    parameters=[p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    records=[]
    def counts():
        return [int(optimizer.state[p]['step']) for p in parameters if 'step' in optimizer.state[p]]
    for index in range(start_step,steps):
        begin=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss_sum=0.
        for micro in range(accumulation):
            row=next(iterator)
            result=module.token_step(row)
            loss=result['loss']
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite loss; optimizer not updated')
            scaled_loss=loss/accumulation
            if scaler is None:
                scaled_loss.backward()
            else:
                scaler.scale(scaled_loss).backward()
            loss_sum+=float(loss.detach())/accumulation
            if callable(row.get('on_state')):
                row['on_state'](result['next_state'])
        if scaler is not None:
            scaler.unscale_(optimizer)
        norm=torch.nn.utils.clip_grad_norm_(parameters,1.)
        if not torch.isfinite(norm) or not any(p.grad is not None for p in parameters):
            raise RuntimeError('nonfinite/missing gradient; optimizer not updated; AMP skip is not an AdamW step')
        before=max(counts(),default=0)
        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()
        after=counts()
        if not after or any(value!=before+1 for value in after):
            raise RuntimeError('AdamW actual counter mismatch or skipped AMP step')
        record=dict(step=index+1,loss=loss_sum,gradient_norm=float(norm),seconds=time.monotonic()-begin,
                    actual_optimizer_steps=index+1,amp_skipped_steps=0)
        records.append(record)
        if on_step is not None:
            on_step(record)
    return dict(actual_optimizer_steps=steps,records=records,amp_skipped_steps=0)


def fixed_query_retrieval(keys,labels,queries):
    keys,labels,queries=torch.as_tensor(keys).float(),torch.as_tensor(labels),torch.as_tensor(queries).float()
    if keys.ndim!=2 or queries.ndim!=2 or keys.shape[1]!=queries.shape[1] or labels.shape!=(len(keys),):
        raise ValueError('matching event keys, labels and fixed query matrix required')
    if not len(keys) or not torch.isfinite(keys).all() or not torch.isfinite(queries).all():
        raise ValueError('finite nonempty retrieval inputs required')
    scores=F.normalize(queries,dim=-1)@F.normalize(keys,dim=-1).T
    rows=[]; missing=[]
    for label in range(len(queries)):
        truth=labels==label
        if not truth.any():
            missing.append(label)
            rows.append(dict(label=label,ap=None,r_at_1=None,r_at_5=None,margin=None,positives=0))
            continue
        order=torch.argsort(scores[label],descending=True,stable=True)
        relevant=truth[order].double()
        precision=relevant.cumsum(0)/torch.arange(1,len(keys)+1,dtype=torch.float64)
        margin=float(scores[label,truth].max()-scores[label,~truth].max()) if (~truth).any() else None
        rows.append(dict(label=label,ap=float((precision*relevant).sum()/relevant.sum()),
                         r_at_1=float(relevant[0]),r_at_5=float(relevant[:5].bool().any()),margin=margin,
                         positives=int(truth.sum())))
    return dict(macro_map=None if missing else sum(r['ap'] for r in rows)/len(rows),
                macro_r_at_1=None if missing else sum(r['r_at_1'] for r in rows)/len(rows),
                macro_r_at_5=None if missing else sum(r['r_at_5'] for r in rows)/len(rows),
                r_at_5_definition='at-least-one-relevant-event-in-top5',
                margin_definition='best-positive-event-minus-best-negative-event',
                per_label=rows,missing_labels=missing,events=len(keys))


def semantic_loss(closed_logits, closed_targets, oracle_logits=None, oracle_targets=None, *, oracle_weight=.3, closed_weights=None, oracle_weights=None):
    def term(logits, targets, weights):
        if logits.shape != targets.shape or logits.numel() == 0 or not torch.isfinite(targets).all():
            raise ValueError('invalid semantic supervision')
        if ((targets < 0) | (targets > 1)).any():
            raise ValueError('semantic labels must be binary probabilities')
        losses = F.binary_cross_entropy_with_logits(logits,targets.to(logits),reduction='none')
        if weights is None:
            return losses.mean()
        if weights.shape != losses.shape or not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError('invalid semantic sample weights')
        return (losses*weights.to(losses)).sum()/weights.sum()
    if not math.isfinite(oracle_weight) or oracle_weight < 0:
        raise ValueError('invalid oracle weight')
    result = term(closed_logits,closed_targets,closed_weights)
    if (oracle_logits is None) != (oracle_targets is None):
        raise ValueError('oracle logits/targets must both be present')
    if oracle_logits is not None:
        result = result+oracle_weight*term(oracle_logits,oracle_targets,oracle_weights)
    return result


def evidence_loss(prediction, target, mask):
    if prediction.shape != target.shape or mask.shape != target.shape or not mask.bool().any():
        raise ValueError('evidence loss requires matching shapes and observed attributes')
    valid = mask.bool()
    if not torch.isfinite(target[valid]).all() or ((target[valid] < 0) | (target[valid] > 1)).any():
        raise ValueError('evidence targets require fixed normalization into [0,1]')
    safe_target = torch.where(valid,target,torch.zeros_like(target))
    return F.smooth_l1_loss(prediction[valid],safe_target[valid])


def age_matched_weights(ages, labels):
    """Equal mass per age stratum and per class, without giving age to a head."""
    if ages.shape != labels.shape or ages.ndim != 1 or not ages.numel():
        raise ValueError('matching nonempty age/class vectors required')
    if not torch.isfinite(ages).all() or not torch.isfinite(labels).all() or not ((labels == 0) | (labels == 1)).all():
        raise ValueError('invalid age/class data')
    strata = ages.unique()
    result = torch.zeros_like(labels,dtype=torch.float32)
    for age in strata:
        for label in (0,1):
            indices = (ages == age) & (labels == label)
            if not indices.any():
                raise ValueError('age matching requires both classes in each stratum')
            result[indices] = 1./(2*len(strata)*int(indices.sum()))
    return result


def age_bias_probe(probabilities, labels, ages, *, early_max_age, threshold=.5):
    p = torch.as_tensor(probabilities,dtype=torch.float64)
    y = torch.as_tensor(labels)
    a = torch.as_tensor(ages,dtype=torch.float64)
    if p.ndim != 1 or p.shape != y.shape or p.shape != a.shape or not p.numel():
        raise ValueError('probe requires matching nonempty vectors')
    if not torch.isfinite(p).all() or not torch.isfinite(a).all() or ((p < 0)|(p > 1)).any() or not ((y == 0)|(y == 1)).all():
        raise ValueError('invalid bias probe data')
    def auc(mask):
        positive, negative = p[mask & (y == 1)], p[mask & (y == 0)]
        if not positive.numel() or not negative.numel():
            return None
        delta = positive[:,None]-negative[None,:]
        return float(((delta > 0).double()+.5*(delta == 0)).mean())
    same = y == 0
    correlation = None
    if same.sum() > 1 and p[same].std() > 0 and a[same].std() > 0:
        correlation = float(torch.corrcoef(torch.stack((p[same],a[same])))[0,1])
    early = a <= early_max_age
    rows = []
    for age in a.unique(sorted=True):
        for label in (0,1):
            values = p[(a == age)&(y == label)]
            rows.append({'age':float(age),'label':label,'count':len(values),
                         'mean_probability':float(values.mean()) if len(values) else None,
                         'mean_ci95_normal':([float(values.mean()-1.96*values.std()/math.sqrt(len(values))),
                                              float(values.mean()+1.96*values.std()/math.sqrt(len(values)))]
                                             if len(values) > 1 else None)})
    return {'same_early_fpr':float((p[same & early] >= threshold).double().mean()) if (same & early).any() else None,
            'same_age_correlation':correlation,'early_auroc':auc(early),'mature_auroc':auc(~early),
            'probability_by_age_and_class':rows}


def seconds_bias_probe(probabilities,labels,age_seconds,episode_ids,*,early_range,mature_range,threshold=.5):
    p=torch.as_tensor(probabilities,dtype=torch.float64)
    y=torch.as_tensor(labels)
    a=torch.as_tensor(age_seconds,dtype=torch.float64)
    if p.ndim!=1 or p.shape!=y.shape or p.shape!=a.shape or len(episode_ids)!=len(p) or not len(p):
        raise ValueError('matching probabilities, labels, actual ages and Candidate IDs required')
    if not torch.isfinite(p).all() or not torch.isfinite(a).all() or ((p<0)|(p>1)).any() or not ((y==0)|(y==1)).all():
        raise ValueError('invalid seconds probe data')
    for interval in (early_range,mature_range):
        if len(interval)!=2 or not all(math.isfinite(v) for v in interval) or not 0<=interval[0]<=interval[1]:
            raise ValueError('explicit finite actual-second intervals required')
    if early_range[1]>=mature_range[0] or any(not isinstance(uid,str) or not uid for uid in episode_ids):
        raise ValueError('disjoint age ranges and nonempty Candidate IDs required')
    windows={name:(a>=interval[0]-1e-8)&(a<=interval[1]+1e-8)
             for name,interval in (('early',early_range),('mature',mature_range))}
    def auc(mask):
        positives=p[mask&(y==1)]; negatives=p[mask&(y==0)].sort().values
        if not len(positives) or not len(negatives):
            return None
        lower=torch.searchsorted(negatives,positives,right=False)
        upper=torch.searchsorted(negatives,positives,right=True)
        return float((lower.double()+.5*(upper-lower)).sum()/(len(positives)*len(negatives)))
    same=y==0
    correlation=None
    if same.sum()>1 and p[same].std()>0 and a[same].std()>0:
        correlation=float(torch.corrcoef(torch.stack((p[same],a[same])))[0,1])
    coverage={name:{f'{label_name}_candidates':len({uid for uid,valid in zip(episode_ids,(mask&(y==label)).tolist()) if valid})
                   for label,label_name in ((1,'positive'),(0,'negative'))} for name,mask in windows.items()}
    curves=[]
    for age in a.unique(sorted=True):
        for label in (0,1):
            selected=(a==age)&(y==label)
            clusters={}
            for uid,value,valid in zip(episode_ids,p.tolist(),selected.tolist()):
                if valid:
                    clusters.setdefault(uid,[]).append(value)
            values=torch.tensor([sum(v)/len(v) for v in clusters.values()],dtype=torch.float64)
            mean=float(values.mean()) if len(values) else None
            half=float(1.96*values.std()/math.sqrt(len(values))) if len(values)>1 else None
            curves.append(dict(age_seconds=float(age),label=label,independent_candidates=len(values),
                mean_probability=mean,mean_ci95_normal=None if half is None else [mean-half,mean+half]))
    return dict(early_auroc=auc(windows['early']),mature_auroc=auc(windows['mature']),
        same_early_fpr=float((p[same&windows['early']]>=threshold).double().mean()) if (same&windows['early']).any() else None,
        same_age_correlation=correlation,coverage=coverage,early_range=list(early_range),mature_range=list(mature_range),
        probability_by_age_and_class=curves,age_unit='seconds',candidate_count=len(set(episode_ids)))


def seconds_probe_pass(metrics,criteria,*,min_candidates=20):
    if not isinstance(min_candidates,int) or min_candidates<20:
        raise ValueError('at least20 independent Candidates per window/class required')
    coverage=metrics.get('coverage',{})
    return bool(_criteria_pass(metrics,criteria) and all(coverage.get(window,{}).get(key,0)>=min_candidates
        for window in ('early','mature') for key in ('positive_candidates','negative_candidates')))


def config_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _criteria_pass(metrics,criteria):
    keys = ('min_early_auroc','min_mature_auroc','max_same_early_fpr','max_abs_same_age_correlation')
    if set(criteria) != set(keys) or not all(isinstance(criteria[k],(int,float)) and math.isfinite(criteria[k])
                                            and 0 <= criteria[k] <= 1 for k in keys):
        raise ValueError('explicit finite bias probe acceptance criteria required')
    measured = ('early_auroc','mature_auroc','same_early_fpr','same_age_correlation')
    if any(metrics.get(k) is None or not math.isfinite(metrics[k]) for k in measured):
        return False
    return bool(metrics['early_auroc'] >= criteria['min_early_auroc']
        and metrics['mature_auroc'] >= criteria['min_mature_auroc']
        and metrics['same_early_fpr'] <= criteria['max_same_early_fpr']
        and abs(metrics['same_age_correlation']) <= criteria['max_abs_same_age_correlation'])


@torch.no_grad()
def probe_semantic_head(head,rows,*,frontend_digest,cache_sha256,early_max_age=None,split,criteria=None,
                        early_range=None,mature_range=None,min_candidates=20):
    """Measure the CURRENT head on separate development pairs after training."""
    if split != 'development' or not rows:
        raise ValueError('bias probe requires separate development rows')
    from .checkpoint import module_digest
    device = next(head.parameters()).device
    was_training = head.training
    head.eval()
    try:
        probabilities = torch.cat([head(row['active'].to(device),row['candidate'].to(device)).sigmoid().cpu() for row in rows])
    finally:
        head.train(was_training)
    if early_range is not None or mature_range is not None:
        if early_range is None or mature_range is None or early_max_age is not None:
            raise ValueError('explicit seconds probe cannot use legacy age bucket threshold')
        metrics=seconds_bias_probe(probabilities,torch.cat([r['target'] for r in rows]),
            torch.cat([r['age_seconds'] for r in rows]),[uid for r in rows for uid in r['episode_ids']],
            early_range=early_range,mature_range=mature_range)
        return dict(probe_schema=2,metrics=metrics,semantic_head_digest=module_digest(head),
            frontend_digest=frontend_digest,cache_sha256=cache_sha256,split=split,early_range=list(early_range),
            mature_range=list(mature_range),min_candidates=min_candidates,criteria=criteria,
            criteria_sha256=config_digest(criteria) if criteria is not None else None,
            passed=seconds_probe_pass(metrics,criteria,min_candidates=min_candidates) if criteria is not None else None)
    if early_max_age is None:
        raise ValueError('legacy probe needs explicit bucket threshold; use seconds for training pilot')
    metrics = age_bias_probe(probabilities,torch.cat([r['target'] for r in rows]),
                            torch.cat([r['age_bucket'] for r in rows]),early_max_age=early_max_age)
    return dict(metrics=metrics,semantic_head_digest=module_digest(head),frontend_digest=frontend_digest,
                cache_sha256=cache_sha256,split=split,early_max_age=early_max_age,criteria=criteria,
                criteria_sha256=config_digest(criteria) if criteria is not None else None,
                passed=_criteria_pass(metrics,criteria) if criteria is not None else None)


def validate_bias_probe(probe,*,frontend_digest,semantic_head_digest):
    if (not isinstance(probe,dict) or probe.get('passed') is not True or probe.get('split') != 'development'
        or probe.get('frontend_digest') != frontend_digest or probe.get('semantic_head_digest') != semantic_head_digest
        or not isinstance(probe.get('cache_sha256'),str) or len(probe['cache_sha256']) != 64
        or any(c not in '0123456789abcdef' for c in probe['cache_sha256'])
        or probe.get('criteria') is None):
        raise ValueError('passed current initialization-bias probe with development source required')
    eligible=(seconds_probe_pass(probe['metrics'],probe['criteria'],min_candidates=probe.get('min_candidates',0))
              if probe.get('probe_schema')==2 else _criteria_pass(probe['metrics'],probe['criteria']))
    if probe.get('criteria_sha256') != config_digest(probe['criteria']) or not eligible:
        raise ValueError('bias probe acceptance/readback mismatch')
    return True
