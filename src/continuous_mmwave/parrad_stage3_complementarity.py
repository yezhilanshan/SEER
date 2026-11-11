"""Train-only frozen semantic/physical probes, with separate session calibration."""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from continuous_mmwave.parrad_stage3_linear_probe import SUBJECTS, binary_metrics
from continuous_mmwave.parrad_stage3_radar_readout import (
    calibrate_threshold, fit_readout, fold_masks, operating_metrics,
)
from continuous_mmwave.parrad_stage3_sequence_data import select_endpoints, sha256
from continuous_mmwave.parrad_stage3_sequence_features import (
    SOURCE_SETS, _validate, features_from_pack, iter_history_blocks,
)

SETTINGS = {
    'version':1, 'projection_seeds':[3407,7211,12893], 'projection_dimension':128,
    'geometry':['joint scaled history cosine distance','joint scaled history residual L2'],
    'input_dimension':130, 'parameter_count':131, 'classifier':'logistic',
    'primary_task':'clean update same/different GT run relative to frozen Active',
    'forbidden_inputs':['age','maturity','lifetime','decision','verify_score','threshold'],
    'calibration_session':'set003', 'fit_sessions':['set000','set001','set002'],
    'held_sessions':['set000','set001','set002','set003'], 'target_fprs':[.05,.10],
    'ridge_lambda':1e-3, 'max_iter':400, 'tolerance_grad':1e-7, 'tolerance_change':1e-9,
    'decision_threshold':.5, 'cpu_threads':1,
    'fit_policy':'one clean common endpoint per Candidate; independently fit each endpoint/cohort group',
    'source_scale':'fit-only RMS residual norm per update, separately by source',
    'projection':'joint concatenated residual causal history; fixed random sign / sqrt(128)',
    'threshold_policy':'separate set003 negative quantile, higher order statistic, reject all cutoff ties',
    'aggregation':'subject macro per seed, then seed median/range; undefined subjects explicitly counted',
    'interpretation':'probe-held TRAIN subjects only; frozen upstream saw these subjects; no production promotion',
}


def experiment_grid():
    groups=[]
    for cohort,ages in (('natural',(4,5,6,7,8)),('forced',(4,5,6,7,8,13))):
        for age in ages:
            groups.append({'name':f'{cohort}-fixed_age-{age}','cohort':cohort,'endpoint_kind':'fixed_age',
                           'age':age,'histories':(1,3) if age<8 else (1,3,8),'task':'update'})
    for age,histories in ((3,(1,3)),(8,(1,3,8))):
        groups.append({'name':f'natural-terminal-min{age}','cohort':'natural','endpoint_kind':'terminal',
                       'age':age,'histories':histories,'task':'update'})
    return tuple(groups)


@lru_cache(maxsize=15)
def _projection(dimension,seed):
    # Same generator/ordering as the immutable sequential audit, cached for one K.
    generator=torch.Generator().manual_seed(seed)
    return (torch.randint(0,2,(dimension,128),generator=generator).float()*2-1)/math.sqrt(128)


def pack_block(current,reference,sources,seed):
    n,history=_validate(current,reference,sources)
    dimensions=[current[s].shape[2] for s in sources]
    matrix=_projection(history*sum(dimensions),seed).reshape(history,sum(dimensions),128)
    blocks={}; offset=0
    for s,d in zip(sources,dimensions):
        c,a=np.asarray(current[s],np.float64),np.asarray(reference[s],np.float64)
        residual=c-a
        projection=(torch.from_numpy(residual.astype(np.float32).reshape(n,history*d)) @
                    matrix[:,offset:offset+d].reshape(history*d,128)).numpy()
        blocks[s]={'projection':projection,'current_norm2':np.square(c).sum(axis=(1,2)),
                   'reference_norm2':np.square(a).sum(axis=(1,2)), 'dot':(c*a).sum(axis=(1,2)),
                   'residual_norm2':np.square(residual).sum(axis=(1,2))}
        offset+=d
    return {'sources':sources,'history':history,'blocks':blocks}


def prepare(source,endpoints,history):
    _projection.cache_clear()
    packs={}
    for indices,current,reference in iter_history_blocks(source,endpoints,('F','E','z'),history):
        for sources in SOURCE_SETS:
            for seed in SETTINGS['projection_seeds']:
                block=pack_block({s:current[s] for s in sources},{s:reference[s] for s in sources},sources,seed)
                key=(sources,seed)
                if key not in packs:
                    packs[key]={'sources':sources,'history':history,'blocks':{
                        s:{field:np.empty((len(endpoints),*v.shape[1:]),dtype=v.dtype)
                           for field,v in block['blocks'][s].items()} for s in sources}}
                for s in sources:
                    for field,v in block['blocks'][s].items():
                        packs[key]['blocks'][s][field][indices]=v
    return packs


def source_scales(pack,fit):
    if pack is None or not np.asarray(fit).any(): return {}
    return {s:max(math.sqrt(float(pack['blocks'][s]['residual_norm2'][fit].mean())/pack['history']),1e-6)
            for s in pack['sources']}


def select_group(checked,group):
    endpoints=[]; excluded=Counter()
    for key,rows in checked['rows_by_record'].items():
        selected,reasons=select_endpoints(rows,checked['candidates_by_record'][key],
            cohort=group['cohort'],endpoint_kind=group['endpoint_kind'],age=group['age'],
            history_lengths=group['histories'],task='update')
        endpoints.extend(selected); excluded.update(reasons)
    return endpoints,dict(excluded)


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def _metrics(y,scores):
    result=binary_metrics(np.asarray(y,bool).tolist(),np.asarray(scores).tolist())
    result.update(negatives=len(y)-int(np.asarray(y).sum()),prevalence=float(np.asarray(y).mean()) if len(y) else None)
    return result


def fit_record(output,identifier,descriptor,rows,y,x,scales):
    fit,cal,held=fold_masks(rows,descriptor['held_subject'])
    ids=np.asarray([f"{r['recording_key']}|{r['candidate_id']}|{r['cohort']}|{r['current_update']}" for r in rows],dtype=str)
    if len(set(ids))!=len(ids): raise ValueError('duplicate endpoint in common cohort')
    role_data={role:{'ids':ids[mask].tolist(),'labels':y[mask].tolist()}
               for role,mask in (('fit',fit),('calibration',cal),('held',held))}
    record={**descriptor,'fit_id':identifier,'status':'undefined','reason':None,'converged':None,
            'parameter_count':0,'source_scales':scales,'endpoint_fingerprint':_digest(role_data['held']),
            'role_fingerprint':_digest(role_data),'fit_subjects':[s for s in SUBJECTS if s!=descriptor['held_subject']],
            'role_counts':{role:{'count':int(mask.sum()),'positives':int(y[mask].sum()),
                         'negatives':int((~y[mask]).sum()),'prevalence':float(y[mask].mean()) if mask.any() else None}
                         for role,mask in (('fit',fit),('calibration',cal),('held',held))}}
    fitted=None; scores=np.empty(0)
    if not held.any(): record['reason']='no_held_endpoints'
    elif not scales: record['reason']='no_fitting_source_scale'
    elif len(np.unique(y[fit]))!=2: record['reason']='fitting_lacks_both_classes'
    else:
        solver_settings={k:SETTINGS[k] for k in ('cpu_threads','ridge_lambda','tolerance_grad','tolerance_change')}
        solver_settings.update(logistic_max_iter=SETTINGS['max_iter'],mlp_hidden=0)
        fitted=fit_readout(x[fit],y[fit],np.ones(int(fit.sum())),x,kind='logistic',
                           seed=descriptor['seed'],settings=solver_settings)
        if fitted['parameter_count']!=131: raise RuntimeError('unequal probe capacity')
        scores=fitted['scores']
        record.update(status='fitted',parameter_count=131,converged=fitted['converged'],
                      iterations=fitted['iterations'],gradient_inf=fitted['gradient_inf'],
                      optimizer_status=fitted['optimizer_status'],objective=float(fitted['losses'][-1]))
    record['metrics']=_metrics(y[held],scores[held]) if fitted else {
        'auroc':None,'average_precision':None,'balanced_accuracy':None,
        **record['role_counts']['held']}
    record['operating_points']={}; threshold_values=[]
    empty_op={k:None for k in ('threshold','tp','fp','tn','fn','false_positive_rate','recall','precision','balanced_accuracy')}
    for rate in SETTINGS['target_fprs']:
        reason=record['reason'] if not fitted else ('no_calibration_negatives' if not (~y[cal]).any() else None)
        threshold=calibrate_threshold(y[cal],scores[cal],rate) if reason is None else None
        threshold_values.append(threshold if threshold is not None else np.nan)
        record['operating_points'][str(rate)]={
            'status':'calibrated' if threshold is not None else 'undefined','reason':reason,
            'target_calibration_fpr':rate,'negative_calibration_count':int((~y[cal]).sum()),
            'calibration_fpr_resolution':1/int((~y[cal]).sum()) if (~y[cal]).any() else None,
            'calibration':operating_metrics(y[cal],scores[cal],threshold) if threshold is not None else empty_op.copy(),
            'held':operating_metrics(y[held],scores[held],threshold) if threshold is not None else empty_op.copy()}
    arrays={f'{role}_endpoint_ids':ids[mask] for role,mask in (('fit',fit),('calibration',cal),('held',held))}
    for role,mask in (('fit',fit),('calibration',cal),('held',held)):
        arrays[f'{role}_targets']=y[mask]
        arrays[f'{role}_scores']=scores[mask] if fitted else np.empty(0)
    arrays.update(source_names=np.asarray(list(scales)),source_scales=np.asarray(list(scales.values())),
                  projection_seed=np.asarray(descriptor['seed']),history=np.asarray(descriptor['history']),
                  target_fprs=np.asarray(SETTINGS['target_fprs']),thresholds=np.asarray(threshold_values))
    if fitted:
        arrays.update({k:fitted[k] for k in ('mean','std','class_weights','sample_weights','base_weights','losses')})
        arrays.update({f'state_{k}':v for k,v in fitted['state'].items()})
        record['fit_metrics']=_metrics(y[fit],scores[fit]); record['calibration_metrics']=_metrics(y[cal],scores[cal])
    file=Path(output)/'fits'/f'{identifier}.npz'
    np.savez_compressed(file,**arrays)
    record.update(artifact=str(file.relative_to(output)),artifact_sha256=sha256(file))
    return record


def flattened_metrics(record):
    values={k:record['metrics'].get(k) for k in ('auroc','average_precision','balanced_accuracy')}
    for label,point in record['operating_points'].items():
        for metric in ('recall','false_positive_rate','precision','balanced_accuracy'):
            values[f'{metric}@{label}']=point['held'].get(metric)
    return values


def paired_delta(left,right):
    if any(left[k]!=right[k] for k in ('endpoint_fingerprint','role_fingerprint')):
        raise ValueError('paired deltas require identical fitting/calibration/held Candidate IDs and labels')
    a,b=flattened_metrics(left),flattened_metrics(right)
    return {k:b[k]-a[k] if a[k] is not None and b[k] is not None else None for k in a}
