"""Fixed-budget multi-horizon diagnostics, full-history and Event input controls."""
from __future__ import annotations
import argparse
import copy
import gc
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from .long_prediction import MatchedResidualPredictor,matched_architecture,make_long_batch
from .losses import gaussian_nll,reduce_prediction_losses
from .p1 import _load_original
from .p1_training import prediction_decomposition
from .prediction_runner import sample_schedule,forward
from .source import file_sha,require_sha,write_json,source_digest,state_digest

POPULATIONS=dict(oracle_full=('event_free','gru','tcn','event'),
    oracle_ablation=('event_free','gru','tcn','event','event_shuffle','event_zero'),
    no_reset_full=('event_free','gru','tcn','event','event_zero'))


def registered_jobs():
    return [dict(population=pop,horizon=h,arm=arm) for pop,arms in POPULATIONS.items() for h in (1,3,5) for arm in arms]


def base_kind(arm):return 'event' if arm.startswith('event_') and arm!='event_free' else arm


@torch.no_grad()
def predict_long_record(model,record,stats,arm,device):
    selected=record['selected'];motion=(record['motion']-stats['mean'])/stats['std']
    if base_kind(arm) not in ('gru','tcn'):
        mu=[];lv=[]
        for begin in range(0,len(selected),128):
            t=selected[begin:begin+128];source=record['shuffle_donor'][t] if arm=='event_shuffle' else t
            if (source<0).any():raise ValueError('unavailable donor')
            event=record['events'][source].to(device)
            if arm=='event_zero':event=torch.zeros_like(event)
            a,b=model(motion[t].to(device),None,event,None);mu.append(a.cpu());lv.append(b.cpu())
        return torch.cat(mu),torch.cat(lv)
    means=torch.empty_like(motion);variances=torch.empty_like(motion)
    for start in np.unique(record['segment_start'][selected]):
        positions=selected[record['segment_start'][selected]==start];stop=int(positions.max())+1;start=int(start)
        radar=((record['radar'][start:stop]-stats['mean'])/stats['std'])[None].to(device)
        mask=torch.ones(1,stop-start,dtype=torch.bool,device=device)
        a,b=model.forward_sequence(motion[start:stop][None].to(device),radar,None,mask)
        offset=torch.as_tensor(positions-start,device=device)
        means[positions]=a[0,offset].cpu();variances[positions]=b[0,offset].cpu()
    return means[selected],variances[selected]


def load_population(config,original,population,role):
    replay=Path(config['replay_root']);view='no_reset' if population.startswith('no_reset') else 'oracle'
    manifest=json.loads((replay/'state-manifest.json').read_text())
    by_uid={r['uid']:r for r in manifest['records'] if r['role']==role and r['view']==view}
    index=json.loads((original/f'cache-{role}/index.json').read_text());records=[]
    for entry in index['records']:
        raw=torch.load(entry['path'],map_location='cpu',weights_only=False);uid=raw['uid'];state_row=by_uid[uid]
        state=torch.load(state_row['path'],map_location='cpu',weights_only=False)
        with np.load(replay/f'compact/{role}-{uid}.npz') as arrays:
            selected=arrays[f'{view}_ablation_common_start' if population=='oracle_ablation' else f'{view}_common_start'].copy()
            assert np.array_equal(state['indices'].numpy(),np.arange(len(raw['motion'])))
            record=dict(uid=uid,subject_id=raw['metadata']['subject_id'],role=role,motion=raw['motion'],radar=raw['radar'],
                events=state['events'],segment_start=state['plan']['segment_start'],selected=selected,
                shuffle_donor=arrays[f'{view}_shuffle_donor'].copy(),age=arrays[f'{view}_age'].copy(),
                state_rms=arrays[f'{view}_state_rms'].copy(),intervals=arrays['intervals'].copy())
            for h in (1,3,5):
                for key in ('deadline_cross','mixed_current','mixed_future'):
                    record[f'{key}_h{h}']=arrays[f'{view}_{key}_h{h}'].copy()
        records.append(record);del raw,state
    return records


def _context(config_path,create=False):
    config_path=Path(config_path);config=yaml.safe_load(config_path.read_text());original,oldlock=_load_original(config)
    replay=Path(config['replay_root'])
    for name,sha in config['replay_locks'].items():require_sha(replay/name,sha)
    for row in json.loads((replay/'state-manifest.json').read_text())['records']:require_sha(row['path'],row['sha256'])
    for name,row in json.loads((replay/'delivery-manifest.json').read_text()).items():require_sha(replay/name,row['sha256'])
    root=Path(config['output_dir'])
    if create:root.mkdir(parents=True,exist_ok=False)
    else:
        lock=json.loads((root/'protocol.json').read_text())
        if lock['code_sha256']!=source_digest() or lock['config_sha256']!=file_sha(config_path):raise ValueError('immutable source/config changed')
        for name,sha in lock['protected'].items():require_sha(name,sha)
    return config,root,original,oldlock


def group_masks(record,horizon):
    t=record['selected'];cross=record[f'deadline_cross_h{horizon}'][t]
    current=record[f'mixed_current_h{horizon}'][t];future=record[f'mixed_future_h{horizon}'][t]
    support=cross|current|future;age=record['age'][t];high=record['state_rms'][t]>3
    return dict(all=np.ones(len(t),bool),deadline_cross=cross,mixed_current=current,mixed_future=future,
                boundary_support=support,clean_interior=~support,age1_6=age<=6,age7_32=(age>=7)&(age<=32),
                age33plus=age>32,rms_le3=~high,rms_above3=high)


def _avg(values):
    values=[v for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(values)) if values else None


def aggregate_groups(rows):
    result={}
    for group in rows[0]['groups']:
        subjects={}
        for subject in sorted({r['subject_id'] for r in rows}):
            records=[r['groups'][group] for r in rows if r['subject_id']==subject and r['groups'][group]['samples']>0]
            metrics={key:_avg([r[key] for r in records]) for key in ('mse','nll','variance','persistence_mse','trend_mse')}
            metrics['skill']=1-metrics['mse']/metrics['persistence_mse'] if metrics['persistence_mse'] and metrics['persistence_mse']>0 else None
            metrics['skill_vs_trend']=1-metrics['mse']/metrics['trend_mse'] if metrics['trend_mse'] and metrics['trend_mse']>0 else None
            subjects[subject]=metrics
        result[group]=dict(samples=sum(r['groups'][group]['samples'] for r in rows),by_subject=subjects,
            subject_macro={key:_avg([v[key] for v in subjects.values()]) for key in next(iter(subjects.values()))})
    return dict(recordings=rows,groups=result)


def report_record(record,horizon,mean,lv,stats):
    t=record['selected'];motion=(record['motion']-stats['mean'])/stats['std'];x=motion[t].numpy();y=motion[t+horizon].numpy()
    trend=(motion[t]+horizon*(motion[t]-motion[t-1])).numpy()
    decomposition=prediction_decomposition(x,y,mean,np.zeros_like(mean) if lv is None else lv)
    if lv is None:
        decomposition['nll'][:]=np.nan;decomposition['variance'][:]=np.nan
    trend_mse=((trend.astype(np.float64)-y)**2).mean(-1)
    row=dict(uid=record['uid'],subject_id=record['subject_id'],groups={})
    for group,mask in group_masks(record,horizon).items():
        row['groups'][group]=dict(samples=int(mask.sum()),**{key:_avg(value[mask]) for key,value in decomposition.items()},
                                  trend_mse=_avg(trend_mse[mask]))
    return row,decomposition


def train_job(model,records,stats,options,schedule,horizon,arm,path,device):
    if options['steps']!=300 or len(schedule)!=300 or any(len(batch)!=8 for batch in schedule):
        raise ValueError('fixed budget requires 300 updates with batch8')
    path.mkdir(parents=True,exist_ok=False);optimizer=torch.optim.AdamW(model.parameters(),lr=options['lr'],weight_decay=options['weight_decay'])
    logs=[];checkpoints={};begin=time.perf_counter();torch.cuda.reset_peak_memory_stats() if device=='cuda' else None
    def save(step):
        counts=[int(s['step']) for s in optimizer.state.values()]
        if step and (not counts or any(n!=step for n in counts)):raise RuntimeError('optimizer count mismatch')
        destination=path/f'step{step:03d}.pt'
        torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),architecture=model.config,
            arm=arm,horizon=horizon,population=path.parents[1].name,actual_optimizer_steps=step,training_config=options,code_sha256=source_digest(),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if device=='cuda' else None),destination)
        checkpoints[str(step)]=file_sha(destination)
    save(0)
    for step,samples in enumerate(schedule,1):
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=make_long_batch(records,stats,samples,horizon,arm,device);mu,lv=forward(model,batch)
        loss=reduce_prediction_losses(gaussian_nll(mu,lv,batch['target']),batch['recording_ids'],torch.ones(len(samples),dtype=torch.bool,device=device))
        if not torch.isfinite(loss):raise RuntimeError('nonfinite loss; no update')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),options['clip_norm'])
        if not torch.isfinite(norm):raise RuntimeError('nonfinite gradient; no update')
        optimizer.step();logs.append(dict(step=step,loss=float(loss.detach()),gradient_norm=float(norm)))
        if step in (50,100,200,300):save(step);print(f'{path.parent.name}/{arm} h{horizon} {step}/300',flush=True)
    result=dict(actual_optimizer_steps=len(logs),checkpoints=checkpoints,elapsed_seconds=time.perf_counter()-begin,
        peak_cuda_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
        clipping_fraction=float(np.mean([r['gradient_norm']>options['clip_norm'] for r in logs])))
    write_json(path/'training.json',dict(**result,logs=logs,schedule=schedule))
    return dict(**result,training_sha256=file_sha(path/'training.json'))


def _setup():
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)


def begin_preparation(root,expected_frontend_sha256):
    if any((root/name).exists() for name in ('preparation-started.json','plans.json','preparation.json','prepare-lock.json')):
        raise FileExistsError('preparation is one-shot; preserve prior smoke evidence')
    path=root/'state-diagnosis/summary.json'
    if not path.is_file():raise ValueError('successful state diagnosis required before preparation')
    summary=json.loads(path.read_text());protocol=json.loads((root/'protocol.json').read_text())
    valid=(summary.get('frontend_unchanged') is True and summary.get('frontend_sha256')==expected_frontend_sha256
        and summary.get('optimizer_steps')==0 and len(summary.get('records',[]))==48
        and len({(r['uid'],r['view']) for r in summary['records']})==48
        and summary.get('code_sha256')==protocol['code_sha256'] and summary.get('config_sha256')==protocol['config_sha256']
        and summary.get('gate_probes') and all(r['state_exact'] for r in summary['gate_probes']))
    if not valid:raise ValueError('diagnosis identity, frozen state, completeness or zero-step proof failed')
    with (root/'preparation-started.json').open('x') as handle:
        json.dump(dict(diagnosis_sha256=file_sha(path),code_sha256=source_digest()),handle)


def verify_prediction_arrays(saved,mean,lv,selected,horizon,decomposition):
    for key,value in dict(mean=mean,selected=selected,future=selected+horizon,**decomposition).items():
        assert np.array_equal(saved[key],value,equal_nan=True), 'saved prediction differs: '+key
    if lv is None:assert 'logvar' not in saved, 'trend must not fabricate uncertainty'
    else:assert np.array_equal(saved['logvar'],lv), 'saved logvar differs'


def verify_target_arrays(saved,expected):
    for key,value in expected.items():assert np.array_equal(saved[key],value), 'saved target/group differs: '+key


def _job_name(job):return f"{job['population']}/h{job['horizon']}/{job['arm']}"


def _model_for(job,plans,device):
    kind=base_kind(job['arm']);torch.manual_seed(42)
    return MatchedResidualPredictor(kind,plans['architectures'][job['population']][kind]).to(device)


def _load_final(root,job,plans,config):
    name=_job_name(job);entry=json.loads((root/'training-completed.json').read_text())[name]
    path=root/'train'/name
    for step,sha in entry['checkpoints'].items():require_sha(path/f'step{int(step):03d}.pt',sha)
    require_sha(path/'training.json',entry['training_sha256'])
    payload=torch.load(path/'step300.pt',map_location=config['device'],weights_only=False)
    if any(payload[key]!=job[key] for key in ('population','horizon','arm')):raise ValueError('checkpoint identity differs')
    if payload['actual_optimizer_steps']!=300 or payload['training_config']!=config['training'] or payload['code_sha256']!=source_digest():
        raise ValueError('checkpoint budget/source differs')
    if payload['architecture']!=plans['architectures'][job['population']][base_kind(job['arm'])]:raise ValueError('architecture differs')
    steps=[int(r['step']) for r in payload['optimizer_state']['state'].values()]
    if not steps or any(s!=300 for s in steps):raise ValueError('saved Adam counts differ')
    model=_model_for(job,plans,config['device']);model.load_state_dict(payload['model_state'],strict=True)
    return model.eval()


def _persistence_logvar(records,stats,h):
    second=[]
    for record in records:
        motion=(record['motion']-stats['mean'])/stats['std'];t=record['selected']
        second.append((motion[t+h]-motion[t]).double().square().mean(0))
    return torch.stack(second).mean(0).clamp(min=math.exp(-6),max=math.exp(2)).log().float().tolist()


def run(config_path,mode):
    _setup();config,root,original,oldlock=_context(config_path,create=(mode=='diagnose'));device=config['device']
    options=config['training']
    if options!=dict(seed=42,steps=300,batch_size=8,lr=.001,weight_decay=.01,clip_norm=1.):raise ValueError('unregistered budget')
    if mode=='diagnose':
        protected=dict(json.loads((Path(config['replay_root'])/'protocol.json').read_text())['protected'])
        for row in json.loads((Path(config['replay_root'])/'state-manifest.json').read_text())['records']:protected[row['path']]=row['sha256']
        for name in config['replay_locks']:protected[str(Path(config['replay_root'])/name)]=file_sha(Path(config['replay_root'])/name)
        write_json(root/'protocol.json',dict(config_sha256=file_sha(config_path),code_sha256=source_digest(),protected=protected,
            jobs=registered_jobs(),formal_training_steps=13500,smoke_steps=45,reset_training=False,test_arrays_read=False,formal_go=False))
        from .state_diagnostic import diagnose
        return diagnose(config,root,oldlock)
    stats=torch.load(original/'normalization.pt',map_location='cpu',weights_only=False)
    if mode=='prepare':
        begin_preparation(root,config['frontend_sha256'])
        plans=dict(architectures={},schedules={},sample_tables={},persistence_logvar={});smokes={}
        for pop,arms in POPULATIONS.items():
            records=load_population(config,original,pop,'train');maximum=max(int(r['age'].max()) for r in records)
            plans['architectures'][pop]={kind:matched_architecture(kind,maximum) for kind in ('event_free','gru','tcn','event')}
            schedule=sample_schedule(records,300,8,42);plans['schedules'][pop]=schedule
            plans['sample_tables'][pop]=[[[records[ri]['uid'],int(records[ri]['selected'][si])] for ri,si in batch] for batch in schedule]
            plans['persistence_logvar'][pop]={str(h):_persistence_logvar(records,stats,h) for h in (1,3,5)}
            for job in [j for j in registered_jobs() if j['population']==pop]:
                model=_model_for(job,plans,device);batch=make_long_batch(records,stats,schedule[0],job['horizon'],job['arm'],device)
                mu,lv=forward(model,batch)
                if not torch.equal(mu,batch['motion']):raise RuntimeError('zero residual identity failed')
                if not torch.allclose(lv,torch.zeros_like(lv),atol=1e-6,rtol=0):raise RuntimeError('initial variance differs')
                initial=state_digest(model);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
                loss=reduce_prediction_losses(gaussian_nll(mu,lv,batch['target']),batch['recording_ids'],torch.ones(len(mu),dtype=torch.bool,device=device))
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm) or not torch.isfinite(loss):raise RuntimeError('smoke gradient/loss nonfinite')
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in model.parameters()):raise RuntimeError('smoke update nonfinite')
                # Nonzero learned mean: CUDA all-prefix implementation checked against independent prefixes.
                model.eval();record=dict(records[0]);record['selected']=record['selected'][np.unique(np.linspace(0,len(record['selected'])-1,3,dtype=int))]
                a,b=predict_long_record(model,record,stats,job['arm'],device)
                direct=make_long_batch([record],stats,[(0,i) for i in range(len(record['selected']))],job['horizon'],job['arm'],device)
                with torch.no_grad():c,d=forward(model,direct)
                torch.testing.assert_close(a,c.cpu(),atol=1e-4,rtol=0);torch.testing.assert_close(b,d.cpu(),atol=1e-4,rtol=0)
                smokes[_job_name(job)]=dict(initial_sha256=initial,zero_mean_exact=True,independent_smoke_steps=1,
                    gradient_norm=float(norm),prefix_equivalence_max_error=max(float((a-c.cpu()).abs().max()),float((b-d.cpu()).abs().max())),
                    max_training_batch_history=int(batch['prefix_mask'].sum(1).max()))
                print('prepared '+_job_name(job),flush=True);del model,optimizer,batch,mu,lv,loss,direct,c,d
            del records;gc.collect();torch.cuda.empty_cache() if device=='cuda' else None
        if plans['sample_tables']['oracle_full']!=plans['sample_tables']['no_reset_full']:raise RuntimeError('full population sample mismatch')
        write_json(root/'plans.json',plans);write_json(root/'preparation.json',smokes)
        write_json(root/'prepare-lock.json',{name:file_sha(root/name) for name in ('plans.json','preparation.json','preparation-started.json','state-diagnosis/summary.json')})
        return dict(status='prepared',jobs=len(smokes),smoke_steps=45)
    for name,sha in json.loads((root/'prepare-lock.json').read_text()).items():require_sha(root/name,sha)
    plans=json.loads((root/'plans.json').read_text());completed={};all_reports={};checks=[]
    for pop,arms in POPULATIONS.items():
        records=load_population(config,original,pop,'train' if mode=='train' else 'validation')
        for job in [j for j in registered_jobs() if j['population']==pop]:
            name=_job_name(job);h=job['horizon'];arm=job['arm']
            if mode=='train':
                model=_model_for(job,plans,device)
                expected=json.loads((root/'preparation.json').read_text())[name]['initial_sha256']
                if state_digest(model)!=expected:raise RuntimeError('formal initialization differs from zero probe')
                completed[name]=train_job(model,records,stats,options,plans['schedules'][pop],h,arm,root/'train'/name,device)
                write_json(root/'training-progress.json',completed)
            elif mode in ('evaluate','verify'):
                model=_load_final(root,job,plans,config);rows=[]
                for record in records:
                    mu,lv=predict_long_record(model,record,stats,arm,device)
                    row,decomp=report_record(record,h,mu.numpy(),lv.numpy(),stats);rows.append(row)
                    path=root/'evaluation'/name/f"{record['uid']}.npz"
                    if mode=='evaluate':
                        path.parent.mkdir(parents=True,exist_ok=True)
                        np.savez_compressed(path,mean=mu.numpy(),logvar=lv.numpy(),selected=record['selected'],future=record['selected']+h,**decomp)
                    else:
                        with np.load(path) as saved:verify_prediction_arrays(saved,mu.numpy(),lv.numpy(),record['selected'],h,decomp)
                        checks.append(dict(job=name,uid=record['uid'],sha256=file_sha(path),saved_weight_replay_exact=True))
                all_reports[name]=aggregate_groups(rows)
                if mode=='verify':assert all_reports[name]==json.loads((root/'evaluation/results.json').read_text())[name]
                print(mode+' '+name,flush=True)
            else:raise ValueError(mode)
            del model;gc.collect()
        if mode in ('evaluate','verify'):
            for h in (1,3,5):
                for arm in ('persistence','linear_trend'):
                    rows=[]
                    for record in records:
                        t=record['selected'];motion=(record['motion']-stats['mean'])/stats['std']
                        mu=motion[t] if arm=='persistence' else motion[t]+h*(motion[t]-motion[t-1])
                        lv=np.broadcast_to(np.array(plans['persistence_logvar'][pop][str(h)],np.float32),mu.shape).copy() if arm=='persistence' else None
                        row,decomp=report_record(record,h,mu.numpy(),lv,stats);rows.append(row)
                        target=root/'targets'/pop/f'h{h}'/f"{record['uid']}.npz"
                        expected=dict(current=motion[t].numpy(),target=motion[t+h].numpy(),selected=t,future=t+h,
                                      age=record['age'][t],state_rms=record['state_rms'][t],**group_masks(record,h))
                        path=root/'evaluation'/pop/f'h{h}'/arm/f"{record['uid']}.npz"
                        arrays=dict(mean=mu.numpy(),selected=t,future=t+h,**decomp)
                        if lv is not None:arrays['logvar']=lv
                        if mode=='evaluate':
                            target.parent.mkdir(parents=True,exist_ok=True);path.parent.mkdir(parents=True,exist_ok=True)
                            if not target.exists():np.savez_compressed(target,**expected)
                            np.savez_compressed(path,**arrays)
                        else:
                            with np.load(target) as saved:verify_target_arrays(saved,expected)
                            with np.load(path) as saved:verify_prediction_arrays(saved,mu.numpy(),lv,t,h,decomp)
                            checks.append(dict(job=f'{pop}/h{h}/{arm}',uid=record['uid'],sha256=file_sha(path),
                                control_recomputed_exact=True,target_sha256=file_sha(target)))
                    name=f'{pop}/h{h}/{arm}';all_reports[name]=aggregate_groups(rows)
                    if mode=='verify':assert all_reports[name]==json.loads((root/'evaluation/results.json').read_text())[name]
        del records;gc.collect();torch.cuda.empty_cache() if device=='cuda' else None
    if mode=='train':write_json(root/'training-completed.json',completed)
    if mode=='evaluate':write_json(root/'evaluation/results.json',all_reports)
    if mode=='verify':
        if len(checks)!=378:raise RuntimeError('incomplete weight replay and controls')
        for name,sha in json.loads((root/'protocol.json').read_text())['protected'].items():require_sha(name,sha)
        write_json(root/'verification.json',dict(status='verified',checks=checks,formal_optimizer_steps=13500,
            smoke_optimizer_steps=45,formal_go=False,reset_training=False,test_arrays_read=False,protected_unchanged=True))
    return dict(status=mode,jobs=len(completed) if mode=='train' else len(all_reports),checks=len(checks))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--mode',required=True,
        choices=('diagnose','prepare','train','evaluate','verify'))
    args=parser.parse_args();print(json.dumps(run(args.config,args.mode)),flush=True)
