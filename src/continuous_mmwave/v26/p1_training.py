"""Fixed-budget residual experiment; no horizon/reset/variance-policy changes."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch
import yaml
from .p1 import _load_original, _fixed_probe, _probe, require_same_schedule
from .prediction_runner import ARMS, load_records, make_batch, forward, sample_schedule, predict_record
from .predictors import make_residual_predictor
from .losses import gaussian_nll, reduce_prediction_losses
from .source import source_digest, file_sha, require_sha, write_json


def train_residual_arm(model,records,stats,options,schedule,output,device,checkpoints=(0,50,100,200,300)):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    if len(schedule)!=options['steps'] or checkpoints[0]!=0 or checkpoints[-1]!=options['steps']:
        raise ValueError('fixed budget/checkpoints mismatch')
    model.to(device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=options['lr'],weight_decay=options['weight_decay'])
    probes=_fixed_probe(records,schedule);logs=[];saved={};metrics={}
    def save(step):
        path=output/f'step{step:03d}.pt'
        counts=[int(row['step']) for row in optimizer.state.values()]
        if step and (not counts or any(n!=step for n in counts)):raise RuntimeError('Adam step count mismatch')
        torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),kind=model.kind,
               architecture=model.config,parameterization='residual_zero',actual_optimizer_steps=step,
               training_config=options,code_sha256=source_digest(),torch_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),path)
        saved[str(step)]=file_sha(path)
        metrics[str(step)]=_probe(model.eval(),probes,stats,device,require_identity=(step==0));model.train()
    save(0)
    for step,samples in enumerate(schedule,1):
        optimizer.zero_grad(set_to_none=True);batch=make_batch(records,stats,samples,device)
        mu,lv=forward(model,batch);loss=reduce_prediction_losses(gaussian_nll(mu,lv,batch['target']),batch['recording_ids'],
                                          torch.ones(len(mu),dtype=torch.bool,device=device))
        if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),options['clip_norm'])
        if not torch.isfinite(norm):raise RuntimeError('nonfinite training gradient')
        optimizer.step();logs.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))
        if step in checkpoints:
            save(step);print(f'residual {model.kind}: {step} successful steps; fixed fit MSE={metrics[str(step)]["subject_macro"]["mse"]:.7f}',flush=True)
    result=dict(actual_optimizer_steps=options['steps'],checkpoints=saved,fit_probes=metrics)
    write_json(output/'training.json',dict(**result,logs=logs,schedule=schedule))
    result['training_sha256']=file_sha(output/'training.json')
    return result


def _mean(values):
    valid=[float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(valid)) if valid else None


def prediction_decomposition(x,y,mu,lv):
    x,y,mu,lv=[np.asarray(v,dtype=np.float64) for v in (x,y,mu,lv)]
    d=y-x;a=mu-x;err=mu-y
    p=(d*d).mean(-1);energy=(a*a).mean(-1);alignment=(a*d).mean(-1);mse=(err*err).mean(-1)
    assert np.allclose(mse,p+energy-2*alignment,rtol=1e-11,atol=1e-12)
    denominator=np.linalg.norm(a,axis=-1)*np.linalg.norm(d,axis=-1)
    cosine=np.divide((a*d).sum(-1),denominator,out=np.full_like(denominator,np.nan),where=denominator>1e-12)
    return dict(mse=mse,persistence_mse=p,correction_energy=energy,alignment=alignment,
                residual_cosine=cosine,win=(mse<p).astype(float),variance=np.exp(lv).mean(-1),
                nll=(.5*(math.log(2*math.pi)+lv+err*err*np.exp(-lv))).mean(-1))


def aggregate(rows):
    keys=rows[0]['metrics'].keys();by_subject={}
    for subject in sorted({row['subject_id'] for row in rows}):
        group=[row for row in rows if row['subject_id']==subject]
        values={key:_mean([r['metrics'][key] for r in group]) for key in keys}
        values['skill']=1-values['mse']/values['persistence_mse'] if values['persistence_mse']>0 else None
        by_subject[subject]=values
    macro={key:_mean([row[key] for row in by_subject.values()]) for key in next(iter(by_subject.values()))}
    return dict(recordings=rows,by_subject=by_subject,subject_macro=macro,formal_go=False)


def _context(config_path):
    config_path=Path(config_path);config=yaml.safe_load(config_path.read_text());original,oldlock=_load_original(config)
    root=Path(config['output_dir']);lock=json.loads((root/'protocol.json').read_text())
    if lock['code_sha256']!=source_digest() or lock['config_sha256']!=file_sha(config_path):raise ValueError('P1 source/config changed')
    for path,sha in lock['protected'].items():require_sha(path,sha)
    for row in json.loads((root/'index-lock.json').read_text()):require_sha(row['path'],row['sha256'])
    for path,sha in json.loads((root/'prepare-lock.json').read_text()).items():require_sha(root/path,sha)
    if config['training'] != dict(seed=42,steps=300,batch_size=8,lr=.001,weight_decay=.01,clip_norm=1.):
        raise ValueError('unregistered training budget')
    return config,root,original,oldlock


def _load_final(root,arm,config,oldlock,device):
    completed=json.loads((root/'train/completed.json').read_text())
    if set(completed)!=set(ARMS):raise ValueError('incomplete trained arms')
    entry=completed[arm]
    for step in ('0','50','100','200','300'):require_sha(root/f'train/{arm}/step{int(step):03d}.pt',entry['checkpoints'][step])
    require_sha(root/f'train/{arm}/training.json',entry['training_sha256'])
    payload=torch.load(root/f'train/{arm}/step300.pt',map_location=device,weights_only=False)
    if (payload['parameterization']!='residual_zero' or payload['kind']!=arm or payload['actual_optimizer_steps']!=300
        or payload['training_config']!=config['training'] or payload['architecture']!=oldlock['architectures'][arm]
        or payload['code_sha256']!=source_digest()):raise ValueError('checkpoint provenance differs')
    counts=[int(row['step']) for row in payload['optimizer_state']['state'].values()]
    if not counts or any(step!=300 for step in counts):raise ValueError('Adam count differs')
    model=make_residual_predictor(arm,payload['architecture']).to(device).eval();model.load_state_dict(payload['model_state'],strict=True)
    return model


def run(config_path,mode):
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    config,root,original,oldlock=_context(config_path);device=config['device']
    stats=torch.load(original/'normalization.pt',weights_only=False)
    if mode=='train':
        records=load_records(original/'cache-train/index.json');options=config['training']
        schedule=sample_schedule(records,options['steps'],options['batch_size'],options['seed'])
        for arm in ARMS:require_same_schedule(schedule,json.loads((original/f'train/{arm}/training.json').read_text())['schedule'])
        destination=root/'train';destination.mkdir(exist_ok=False);results={}
        for arm in ARMS:
            torch.manual_seed(options['seed']);model=make_residual_predictor(arm,oldlock['architectures'][arm])
            initial=torch.load(root/f'prepare/{arm}/step000.pt',map_location='cpu',weights_only=False)
            if initial['parameterization']!='residual_zero' or initial['actual_optimizer_steps']!=0:raise ValueError('wrong prepared initial state')
            model.load_state_dict(initial['model_state'],strict=True)
            results[arm]=train_residual_arm(model,records,stats,options,schedule,destination/arm,device)
        write_json(destination/'completed.json',results);return results
    records=load_records(original/'cache-validation/index.json')
    persistence=torch.load(original/'persistence-logvar.pt',weights_only=False)
    if mode=='evaluate':
        trained={arm:_load_final(root,arm,config,oldlock,device) for arm in ARMS}
        destination=root/'evaluation';destination.mkdir(exist_ok=False);reports={}
        for arm in ('persistence',)+ARMS:
            folder=destination/arm;folder.mkdir();rows=[]
            for record in records:
                model=None if arm=='persistence' else trained[arm]
                mu,lv,target=predict_record(model,record,stats,device,persistence)
                x=(record['motion'][record['selected']]-stats['mean'])/stats['std']
                arrays=prediction_decomposition(x.numpy(),target.numpy(),mu.numpy(),lv.numpy())
                np.savez_compressed(folder/f'{record["uid"]}.npz',mean=mu.numpy(),logvar=lv.numpy(),target=target.numpy(),
                     current=x.numpy(),selected=record['selected'].numpy(),future=record['future'].numpy(),**arrays)
                rows.append(dict(uid=record['uid'],subject_id=record['metadata']['subject_id'],samples=len(mu),
                                 metrics={key:_mean(value) for key,value in arrays.items()}))
            reports[arm]=aggregate(rows)
        write_json(destination/'results.json',reports);return reports
    if mode=='verify':
        reports=json.loads((root/'evaluation/results.json').read_text());checks=[]
        if set(reports)!=set(('persistence',)+ARMS):raise ValueError('incomplete evaluation')
        for arm,report in reports.items():
            model=None if arm=='persistence' else _load_final(root,arm,config,oldlock,device)
            if len(report['recordings'])!=len(records):raise ValueError('recording count differs')
            rows=[]
            for record,row in zip(records,report['recordings'],strict=True):
                if row['uid']!=record['uid'] or row['subject_id']!=record['metadata']['subject_id'] or row['samples']!=len(record['selected']):
                    raise ValueError('recording identity differs')
                path=root/f'evaluation/{arm}/{record["uid"]}.npz'
                with np.load(path) as saved:
                    assert np.array_equal(saved['selected'],record['selected'].numpy()) and np.array_equal(saved['future'],record['future'].numpy())
                    x=((record['motion'][record['selected']]-stats['mean'])/stats['std']).numpy()
                    assert np.array_equal(saved['current'],x)
                    values=prediction_decomposition(x,saved['target'],saved['mean'],saved['logvar'])
                    for key,value in values.items():
                        assert np.array_equal(saved[key],value,equal_nan=True)
                        metric=_mean(value)
                        if metric is None:assert row['metrics'][key] is None
                        else:assert np.isclose(metric,row['metrics'][key],rtol=1e-10,atol=1e-12)
                    replay=predict_record(model,record,stats,device,persistence)
                    for key,value in zip(('mean','logvar','target'),replay,strict=True):assert np.array_equal(saved[key],value.numpy())
                    rows.append(dict(row,metrics={key:_mean(value) for key,value in values.items()}))
                checks.append(dict(arm=arm,uid=record['uid'],sha256=file_sha(path),weight_replay_exact=True))
            again=aggregate(rows)
            assert again['by_subject']==report['by_subject'] and again['subject_macro']==report['subject_macro']
        result=dict(status='verified',checks=checks,formal_go=False,test_arrays_read=False)
        write_json(root/'verification.json',result);return result
    raise ValueError(mode)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--mode',required=True,choices=('train','evaluate','verify'))
    args=parser.parse_args();print(json.dumps(run(args.config,args.mode),allow_nan=False),flush=True)


if __name__=='__main__':main()
