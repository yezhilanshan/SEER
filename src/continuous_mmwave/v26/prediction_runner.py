"""Fixed-budget, immutable prediction pilot and independent result readback."""
from __future__ import annotations

import argparse
import gc
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml

from .features import export_prediction_cache
from .losses import gaussian_nll, reduce_prediction_losses, fit_persistence_variance
from .predictors import make_predictor, select_architecture
from .source import (audit_sources, file_sha, load_frozen_frontend, read_manifest, require_sha,
                     source_digest, write_json, require_frontend_assets)
from .targets import fit_normalization

ARMS = ('event_free','gru','tcn','event')


def sample_schedule(records,steps,batch_size,seed):
    if not records or any(len(r['selected']) == 0 for r in records):
        raise ValueError('nonempty prediction samples required')
    rng = random.Random(seed)
    return [[(ri,rng.randrange(len(records[ri]['selected'])))
             for ri in [rng.randrange(len(records)) for _ in range(batch_size)]] for _ in range(steps)]


def make_batch(records,stats,samples,device):
    if not samples: raise ValueError('empty samples')
    prefixes,motions,targets,events = [],[],[],[]
    for ri,si in samples:
        record = records[ri]
        t,future,start = (int(record[key][si]) for key in ('selected','future','prefix_start'))
        if not 0 <= start <= t < future < len(record['motion']):
            raise ValueError('invalid causal sample indices')
        prefixes.append((record['radar'][start:t+1]-stats['mean'])/stats['std'])
        motions.append((record['motion'][t]-stats['mean'])/stats['std'])
        targets.append((record['motion'][future]-stats['mean'])/stats['std'])
        events.append(record['event'][si])
    length = max(len(p) for p in prefixes)
    prefix = torch.zeros(len(samples),length,*prefixes[0].shape[1:])
    mask = torch.zeros(len(samples),length,dtype=torch.bool)
    for i,value in enumerate(prefixes): prefix[i,:len(value)] = value; mask[i,:len(value)] = True
    batch = dict(motion=torch.stack(motions),target=torch.stack(targets),radar_prefix=prefix,
                 prefix_mask=mask,event_tokens=torch.stack(events),recording_ids=torch.tensor([ri for ri,_ in samples]))
    return {k:v.detach().to(device) for k,v in batch.items()}


def forward(model,batch):
    return model(batch['motion'],batch['radar_prefix'],batch['event_tokens'],batch['prefix_mask'])


def train_one(model,records,stats,config,output_dir,device):
    output = Path(output_dir); output.mkdir(parents=True,exist_ok=False)
    schedule = sample_schedule(records,config['steps'],config['batch_size'],config['seed'])
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    model.to(device).train()
    logs = []
    for step,samples in enumerate(schedule,1):
        batch = make_batch(records,stats,samples,device)
        optimizer.zero_grad(set_to_none=True)
        mean,logvar = forward(model,batch)
        losses = gaussian_nll(mean,logvar,batch['target'])
        loss = reduce_prediction_losses(losses,batch['recording_ids'],torch.ones_like(losses,dtype=torch.bool))
        if not torch.isfinite(loss): raise RuntimeError('nonfinite loss; no optimizer update')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(),config['clip_norm'])
        if not torch.isfinite(norm) or not any(p.grad is not None for p in model.parameters()):
            raise RuntimeError('invalid gradients; no optimizer update')
        optimizer.step()
        logs.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))
        if step % 50 == 0: print(f'{model.kind} step={step} loss={logs[-1]["loss"]:.5f}',flush=True)
    counts = [int(v['step']) for v in optimizer.state.values()]
    if not counts or any(c != config['steps'] for c in counts): raise RuntimeError('optimizer step mismatch')
    torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),architecture=model.config,
                    kind=model.kind,training_config=config,schedule=schedule,torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    actual_optimizer_steps=config['steps'],code_sha256=source_digest()),output/'final.pt')
    write_json(output/'training.json',dict(actual_optimizer_steps=config['steps'],logs=logs,schedule=schedule))
    return dict(actual_optimizer_steps=config['steps'],final_loss=logs[-1]['loss'],checkpoint_sha256=file_sha(output/'final.pt'))


def evaluate_predictions(prediction_rows,source):
    if not prediction_rows: raise ValueError('no evaluation samples')
    subjects = {}
    for row in prediction_rows: subjects.setdefault(row['subject_id'],[]).append(row)
    by_subject = {key:{metric:float(np.mean([r[metric] for r in rows])) for metric in ('nll','mse','variance')}
                  for key,rows in subjects.items()}
    return dict(recordings=prediction_rows,by_subject=by_subject,
                subject_macro={metric:float(np.mean([r[metric] for r in by_subject.values()]))
                               for metric in ('nll','mse','variance')},source=source,formal_go=False)


def load_records(index_path):
    index = json.loads(Path(index_path).read_text())
    records = []
    for row in index['records']:
        require_sha(row['path'],row['sha256'])
        record = torch.load(row['path'],map_location='cpu',weights_only=False)
        records.append(record)
    return records


def select_pilot_rows(manifest):
    rows = read_manifest(manifest)
    selected = []
    for subject in sorted({r['subject_id'] for r in rows}):
        group = None
        for session in sorted({r['session_id'] for r in rows if r['subject_id']==subject}):
            candidate=sorted([r for r in rows if r['subject_id']==subject and r['session_id']==session],key=lambda r:r['sensor_id'])
            if len(candidate)==2 and {r['sensor_id'] for r in candidate} == {'TI_xWR14xx','TI_xWR68xx'}:
                group=candidate
                break
        if group is None: raise ValueError('no complete dual-sensor pilot session')
        selected.extend(group)
    return selected


@torch.no_grad()
def predict_record(model,record,stats,device,persistence_logvar=None):
    means,variances,targets = [],[],[]
    for start in range(0,len(record['selected']),128):
        samples=[(0,si) for si in range(start,min(start+128,len(record['selected'])))]
        batch = make_batch([record],stats,samples,device)
        if model is None: mean,logvar=batch['motion'],persistence_logvar.to(device).expand_as(batch['motion'])
        else: mean,logvar=forward(model,batch)
        means.append(mean.cpu()); variances.append(logvar.cpu()); targets.append(batch['target'].cpu())
    return torch.cat(means),torch.cat(variances),torch.cat(targets)


def check_lock(root,config_path):
    lock = json.loads((root/'protocol.json').read_text())
    if lock['config_sha256'] != file_sha(config_path) or lock['code_sha256'] != source_digest():
        raise ValueError('code/config SHA differs from immutable protocol')
    return lock


def check_cache_lock(root):
    for relative, expected in json.loads((Path(root)/'cache-lock.json').read_text()).items():
        require_sha(Path(root)/relative,expected)


def require_final_checkpoint(root,kind):
    completed=json.loads((Path(root)/'train/completed.json').read_text())
    if set(completed) != set(ARMS): raise ValueError('incomplete trained arms')
    path=Path(root)/f'train/{kind}/final.pt'
    require_sha(path,completed[kind]['checkpoint_sha256'])
    return path


def load_trained_arm(root,kind,lock,config,device):
    path=require_final_checkpoint(root,kind)
    payload=torch.load(path,map_location=device,weights_only=False)
    if (payload['kind'] != kind or payload['architecture'] != lock['architectures'][kind]
        or payload['training_config'] != config['training'] or payload['actual_optimizer_steps'] != config['training']['steps']
        or payload['code_sha256'] != lock['code_sha256']):
        raise ValueError('final checkpoint provenance mismatch')
    counts=[int(v['step']) for v in payload['optimizer_state']['state'].values()]
    if not counts or any(c != config['training']['steps'] for c in counts): raise ValueError('saved optimizer steps mismatch')
    model=make_predictor(kind,payload['architecture']).to(device).eval()
    model.load_state_dict(payload['model_state'],strict=True)
    return model


def validate_evaluation_report(reports,records):
    if set(reports) != set(('persistence',)+ARMS): raise ValueError('incomplete evaluation arms')
    for report in reports.values():
        rows=report['recordings']
        if len(rows) != len(records): raise ValueError('recording identity count mismatch')
        for row,record in zip(rows,records,strict=True):
            if (row['uid'] != record['uid'] or row['subject_id'] != record['metadata']['subject_id']
                or row['samples'] != len(record['selected'])): raise ValueError('evaluation recording identity mismatch')


def validate_saved_indices(saved,record):
    if (not np.array_equal(saved['selected'],record['selected'].numpy())
        or not np.array_equal(saved['future'],record['future'].numpy())):
        raise ValueError('saved future indices differ from cache')


def run_prediction_pilot(config_path,mode):
    config_path=Path(config_path); config=yaml.safe_load(config_path.read_text())
    root=Path(config['output_dir']); device=config['device']
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    if mode=='audit':
        report=audit_sources(config)
        architectures={kind:select_architecture(kind,config['predictor']) for kind in ARMS}
        if not all(c['capacity_matched'] for c in architectures.values()): raise ValueError('capacity unmatched')
        root.mkdir(parents=True,exist_ok=False)
        report.update(config_sha256=file_sha(config_path),architectures=architectures,
                      selected={role:select_pilot_rows(config['manifests'][role]) for role in ('train','validation')},
                      experiment='command_boundary_proxy_prediction_diagnostic')
        write_json(root/'protocol.json',report)
        return report
    lock=check_lock(root,config_path)
    require_frontend_assets(lock['frontend_assets'])
    for name in ('stage1','stage2'):
        require_sha(config[name],lock['checkpoints'][name])
    for role in config['manifests']:
        require_sha(config['manifests'][role],lock['splits'][role]['sha256'])
    if mode != 'export':
        check_cache_lock(root)
    if mode=='export':
        frontend,_,source=load_frozen_frontend(config['frontend_config'],config['stage1'],config['stage2'],lock['checkpoints'],device)
        write_json(root/'frontend.json',source)
        for role in ('train','validation'):
            export_prediction_cache(frontend,config['manifests'][role],lock['selected'][role],root/f'cache-{role}',
                                    dict(role=role,max_prefix=config['max_prefix'],sensor_names=config['sensor_names']))
        del frontend; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        train=load_records(root/'cache-train/index.json'); stats=fit_normalization(train)
        torch.save(stats,root/'normalization.pt')
        current=torch.cat([(r['motion'][r['selected']]-stats['mean'])/stats['std'] for r in train])
        future=torch.cat([(r['motion'][r['future']]-stats['mean'])/stats['std'] for r in train])
        recording_ids=torch.cat([torch.full((len(r['selected']),),i,dtype=torch.long) for i,r in enumerate(train)])
        torch.save(fit_persistence_variance(current,future,recording_ids),root/'persistence-logvar.pt')
        write_json(root/'cache-lock.json', {p.relative_to(root).as_posix():file_sha(p) for p in
                   (root/'cache-train/index.json',root/'cache-validation/index.json',root/'frontend.json',
                    root/'normalization.pt',root/'persistence-logvar.pt')})
        return dict(exported=True,formal_go=False)
    if mode in ('smoke','train'):
        records=load_records(root/'cache-train/index.json'); stats=torch.load(root/'normalization.pt',weights_only=False)
        if mode=='train' and not (root/'smoke/completed.json').exists(): raise ValueError('smoke prerequisite missing')
        destination=root/mode; destination.mkdir(exist_ok=False)
        result={}
        for kind in ARMS:
            torch.manual_seed(config['training']['seed'])
            model=make_predictor(kind,lock['architectures'][kind])
            options=dict(config['training'],steps=1 if mode=='smoke' else config['training']['steps'])
            result[kind]=train_one(model,records,stats,options,destination/kind,device)
        write_json(destination/'completed.json',result)
        return result
    if mode=='evaluate':
        if not (root/'train/completed.json').exists(): raise ValueError('all trained arms required')
        trained={kind:load_trained_arm(root,kind,lock,config,device) for kind in ARMS}
        output=root/'evaluation'; output.mkdir(exist_ok=False)
        records=load_records(root/'cache-validation/index.json'); stats=torch.load(root/'normalization.pt',weights_only=False)
        persistence=torch.load(root/'persistence-logvar.pt',weights_only=False)
        reports={}
        for kind in ('persistence',)+ARMS:
            model=None
            if kind!='persistence':
                model=trained[kind]
            arm=output/kind; arm.mkdir(); rows=[]
            for record in records:
                mean,logvar,target=predict_record(model,record,stats,device,persistence)
                np.savez_compressed(arm/(record['uid']+'.npz'),mean=mean.numpy(),logvar=logvar.numpy(),target=target.numpy(),
                                    selected=record['selected'].numpy(),future=record['future'].numpy())
                rows.append(dict(uid=record['uid'],subject_id=record['metadata']['subject_id'],session_id=record['metadata']['session_id'],
                                 samples=len(mean),nll=float(gaussian_nll(mean,logvar,target).mean()),
                                 mse=float((mean-target).square().mean()),variance=float(logvar.exp().mean())))
            reports[kind]=evaluate_predictions(rows,dict(config_sha256=lock['config_sha256'],code_sha256=lock['code_sha256']))
        write_json(output/'results.json',reports)
        return reports
    if mode=='verify':
        reports=json.loads((root/'evaluation/results.json').read_text()); checks=[]
        records=load_records(root/'cache-validation/index.json'); stats=torch.load(root/'normalization.pt',weights_only=False)
        validate_evaluation_report(reports,records)
        persistence=torch.load(root/'persistence-logvar.pt',weights_only=False)
        for kind,report in reports.items():
            model=None
            if kind!='persistence':
                model=load_trained_arm(root,kind,lock,config,device)
            recomputed=[]
            for record,row in zip(records,report['recordings'],strict=True):
                path=root/f'evaluation/{kind}/{record["uid"]}.npz'
                with np.load(path) as saved:
                    validate_saved_indices(saved,record)
                    mean,logvar,target=(saved[k].astype(np.float64) for k in ('mean','logvar','target'))
                    nll=float(.5*(math.log(2*math.pi)+logvar+(mean-target)**2*np.exp(-logvar)).mean())
                    mse=float(((mean-target)**2).mean()); variance=float(np.exp(logvar).mean())
                    if not all(np.isclose(a,b,rtol=1e-5,atol=1e-6) for a,b in ((nll,row['nll']),(mse,row['mse']),(variance,row['variance']))):
                        raise RuntimeError('independent metric recomputation mismatch')
                    replay=predict_record(model,record,stats,device,persistence)
                    for key,value in zip(('mean','logvar','target'),replay,strict=True):
                        if not np.array_equal(saved[key],value.numpy()): raise RuntimeError('weight replay mismatch')
                recomputed.append(dict(row,nll=nll,mse=mse,variance=variance))
                checks.append(dict(arm=kind,uid=record['uid'],sha256=file_sha(path),replayed=True))
            again=evaluate_predictions(recomputed,{})
            for metric in ('nll','mse','variance'):
                if not np.isclose(again['subject_macro'][metric],report['subject_macro'][metric],rtol=1e-5,atol=1e-6):
                    raise RuntimeError('subject aggregation mismatch')
        result=dict(status='verified',checks=checks,formal_go=False,test_arrays_read=False)
        write_json(root/'verification.json',result)
        return result
    raise ValueError('unknown mode')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--mode',required=True,choices=('audit','export','smoke','train','evaluate','verify'))
    args=parser.parse_args()
    result=run_prediction_pilot(args.config,args.mode)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)
