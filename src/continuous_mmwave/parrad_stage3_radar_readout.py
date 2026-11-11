"""Frozen radar-relational pilot utilities; no production controller integration."""
from __future__ import annotations

from collections import Counter
import math

import numpy as np
import torch
from torch.nn import functional as nnf

from continuous_mmwave.parrad_stage3_linear_probe import SUBJECTS, geometry, relational_features

SETTINGS = {
    'version': 1, 'seeds': [3407, 7211, 12893], 'calibration_session': 'set003',
    'minimum_age_updates': 3, 'cpu_threads': 1, 'ridge_lambda': 1e-3,
    'logistic_max_iter': 400, 'tolerance_grad': 1e-7, 'tolerance_change': 1e-9,
    'mlp_hidden': 32, 'mlp_epochs': 60, 'mlp_batch_size': 256,
    'mlp_learning_rate': 1e-3, 'mlp_weight_decay': 1e-2,
    'target_fprs': [0.05, 0.10], 'threshold_source': 'separate train-session calibration negatives',
    'sample_weight': '1 / eligible natural updates per Candidate, then fit-only weighted class balance',
    'standardization': 'fit-only Candidate-weighted mean/std',
    'stability': 'mean of two adjacent cosine similarities in past three natural updates',
    'training': 'all clean natural updates with complete three-update history',
    'evaluation': 'common natural fixed ages 4/6/7 and actual terminal, minimum age 3',
}
MODEL_SPECS = (
    ('age_logistic', 'logistic'), ('projected_logistic', 'logistic'),
    ('delta_logistic', 'logistic'), ('relation_logistic', 'logistic'),
    ('relation_mlp', 'mlp'), ('stability_mlp', 'mlp'),
)


def fold_masks(rows, held_subject):
    if held_subject not in SUBJECTS or any(
        r.get('source_split') != 'train' or r.get('subject_id') not in SUBJECTS
        or r.get('session_id') not in ('set000','set001','set002','set003')
        or r.get('sensor_id') not in ('TI_xWR14xx','TI_xWR68xx') for r in rows
    ):
        raise ValueError('only the approved train subject/session/sensor matrix is permitted')
    held = np.array([r['subject_id'] == held_subject for r in rows], bool)
    calibration = np.array([not h and r['session_id'] == SETTINGS['calibration_session']
                            for r,h in zip(rows,held)], bool)
    return ~(held | calibration), calibration, held


def endpoint_ids(rows):
    return [f"{r['recording_key']}|{r['candidate_id']}|natural|{r['current_update']}" for r in rows]


def candidate_weights(rows):
    if len(set(endpoint_ids(rows))) != len(rows):
        raise ValueError('duplicate online endpoint')
    counts = Counter((r['recording_key'],r['candidate_id']) for r in rows)
    return np.array([1. / counts[r['recording_key'],r['candidate_id']] for r in rows], np.float64)


def features(current, reference, age_seconds, model, seed, scale):
    c,a,t = np.asarray(current,np.float64),np.asarray(reference,np.float64),np.asarray(age_seconds,np.float64)
    if (c.ndim != 3 or c.shape[1] != 3 or c.shape != a.shape or t.shape != (len(c),)
        or not np.isfinite(c).all() or not np.isfinite(a).all() or not np.isfinite(t).all()
        or not math.isfinite(scale) or scale <= 0
        or not np.array_equal(a,np.repeat(a[:,:1],3,axis=1))):
        raise ValueError('features require finite causal three-update histories and immutable reference')
    c,a = c/scale,a/scale
    now,ref = c[:,-1],a[:,-1]
    if model == 'age_logistic': return t[:,None]
    if model == 'projected_logistic': return relational_features(now,ref,seed)
    if model == 'delta_logistic': return np.column_stack((now-ref,t))
    if model == 'relation_logistic': return np.column_stack((now,ref,t))
    if model in ('relation_mlp','stability_mlp'):
        result = np.column_stack((now,ref,now-ref,t))
        if model == 'stability_mlp':
            # The reference is frozen; only adjacent current Radar vectors enter stability.
            cosine = [1-geometry(c[:,i],c[:,i-1])[:,0] for i in (1,2)]
            result = np.column_stack((result,np.mean(cosine,axis=0)))
        return result
    raise ValueError('unknown radar readout model')


def calibrate_threshold(targets, scores, target_fpr):
    y,p = np.asarray(targets,bool),np.asarray(scores,np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or not 0 < target_fpr < 1 or not (~y).any():
        raise ValueError('calibration requires finite scores and negative examples')
    # Ties at cutoff are all rejected; no optimistic arbitrary tie-breaking.
    cutoff = np.quantile(p[~y],1-target_fpr,method='higher')
    return float(np.nextafter(cutoff,np.inf))


def operating_metrics(targets,scores,threshold):
    y,p = np.asarray(targets,bool),np.asarray(scores,np.float64)
    if y.ndim != 1 or p.shape != y.shape or not np.isfinite(p).all() or not math.isfinite(threshold):
        raise ValueError('invalid operating-point inputs')
    pred = p >= threshold
    tp,fp,tn,fn = [int(v.sum()) for v in (pred&y,pred&~y,~pred&~y,~pred&y)]
    recall = tp/(tp+fn) if tp+fn else None
    fpr = fp/(fp+tn) if fp+tn else None
    return {'threshold':threshold,'tp':tp,'fp':fp,'tn':tn,'fn':fn,
            'false_positive_rate':fpr,'recall':recall,'precision':tp/(tp+fp) if tp+fp else None,
            'balanced_accuracy':(recall+1-fpr)/2 if recall is not None and fpr is not None else None}


def _model(dimension,kind,hidden):
    if kind == 'logistic': return torch.nn.Linear(dimension,1,dtype=torch.float64)
    if kind == 'mlp': return torch.nn.Sequential(torch.nn.Linear(dimension,hidden),torch.nn.GELU(),torch.nn.Linear(hidden,1))
    raise ValueError('unknown readout architecture')


def predict_readout(fitted,values):
    x = np.asarray(values,np.float64)
    if x.ndim != 2 or x.shape[1] != len(fitted['mean']) or not np.isfinite(x).all():
        raise ValueError('invalid prediction features')
    with torch.random.fork_rng(devices=[]):
        model = _model(x.shape[1],fitted['kind'],fitted['hidden'])
    model.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in fitted['state'].items()})
    model.eval()
    dtype = next(model.parameters()).dtype
    scores = []
    with torch.no_grad():
        for start in range(0,len(x),1024):
            batch = torch.tensor((x[start:start+1024]-fitted['mean'])/fitted['std'],dtype=dtype)
            scores.append(torch.sigmoid(model(batch)).flatten().numpy())
    return np.concatenate(scores) if scores else np.empty(0)


def fit_readout(train_x,train_y,base_weights,prediction_x,*,kind,seed,settings=None):
    """The optimizer and transformations receive only fitting labels/features."""
    cfg = SETTINGS if settings is None else settings
    x,y,w = np.asarray(train_x,np.float64),np.asarray(train_y,np.float64),np.asarray(base_weights,np.float64)
    if (x.ndim != 2 or y.shape != (len(x),) or w.shape != y.shape or not len(y)
        or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0)
        or set(np.unique(y)) != {0.,1.}):
        raise ValueError('fitting requires finite weighted features and both training classes')
    torch.set_num_threads(cfg['cpu_threads'])
    mean = np.average(x,axis=0,weights=w)
    std = np.maximum(np.sqrt(np.average((x-mean)**2,axis=0,weights=w)),1e-6)
    masses = np.array([w[y==k].sum() for k in (0,1)])
    class_weights = w.sum()/(2*masses)
    sample_weights = w*class_weights[y.astype(int)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = _model(x.shape[1],kind,cfg['mlp_hidden'])
        if kind == 'logistic':
            with torch.no_grad():
                model.weight.zero_(); model.bias.zero_()
        dtype = next(model.parameters()).dtype
        xx,yy,ww = [torch.tensor(v,dtype=dtype) for v in ((x-mean)/std,y,sample_weights)]
        def objective():
            data = (nnf.binary_cross_entropy_with_logits(model(xx).flatten(),yy,reduction='none')*ww).sum()/ww.sum()
            if kind == 'logistic': data = data + .5*cfg['ridge_lambda']*model.weight.square().sum()
            return data
        losses=[]; convergence=None; iterations=None; gradient=None
        if kind == 'logistic':
            optimizer=torch.optim.LBFGS(model.parameters(),max_iter=cfg['logistic_max_iter'],
                tolerance_grad=cfg['tolerance_grad'],tolerance_change=cfg['tolerance_change'],line_search_fn='strong_wolfe')
            def closure():
                optimizer.zero_grad(); loss=objective(); loss.backward(); return loss
            optimizer.step(closure)
            loss=closure(); losses.append(float(loss.detach()))
            gradient=max(float(p.grad.abs().max()) for p in model.parameters())
            state=optimizer.state[next(model.parameters())]
            iterations=int(state.get('n_iter',0))
            change=abs(float(loss.detach())-float(state.get('prev_loss',float('inf'))))
            convergence=gradient<=cfg['tolerance_grad'] or (iterations<cfg['logistic_max_iter'] and change<=cfg['tolerance_change'])
        else:
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['mlp_learning_rate'],weight_decay=cfg['mlp_weight_decay'])
            batches=math.ceil(len(y)/cfg['mlp_batch_size'])
            for _ in range(cfg['mlp_epochs']):
                model.train()
                order=torch.randperm(len(y))
                for start in range(0,len(y),cfg['mlp_batch_size']):
                    indices=order[start:start+cfg['mlp_batch_size']]
                    optimizer.zero_grad()
                    loss=(nnf.binary_cross_entropy_with_logits(model(xx[indices]).flatten(),yy[indices],reduction='none')*ww[indices]).sum()/(ww.sum()/batches)
                    loss.backward(); optimizer.step()
                with torch.no_grad(): losses.append(float(objective()))
            iterations=cfg['mlp_epochs']
        fitted={'kind':kind,'hidden':cfg['mlp_hidden'],'mean':mean,'std':std,
                'class_weights':class_weights,'base_weights':w,'sample_weights':sample_weights,
                'state':{k:v.detach().numpy().copy() for k,v in model.state_dict().items()},
                'losses':np.array(losses),'converged':convergence,'iterations':iterations,'gradient_inf':gradient,
                'optimizer_status':'converged' if convergence else 'not_converged' if kind=='logistic' else 'fixed_budget_complete',
                'parameter_count':sum(p.numel() for p in model.parameters())}
    if not np.isfinite(fitted['losses']).all() or any(not np.isfinite(v).all() for v in fitted['state'].values()):
        raise RuntimeError('nonfinite readout fit')
    fitted['scores']=predict_readout(fitted,prediction_x)
    fitted['train_scores']=predict_readout(fitted,x)
    return fitted


def evaluate(source,output):
    from continuous_mmwave.parrad_stage3_radar_readout_runner import evaluate as run
    return run(source,output)
