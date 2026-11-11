"""One-shot, identity-bound experiment stages; old outputs are never modified."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')

import numpy as np
import torch
import yaml

from .contract import (build_joint_record,fit_motion_contract,joint_schedule,make_joint_batch,
                       validate_class_maps,verify_source_lock,with_previous_observation)
from .model import JointPilotModel,ARMS
from .training import train_joint_job,joint_losses,gradient_diagnostic,validate_training_options,validate_optimizer_checkpoint
from .evaluation import (predict_joint_record,reference_arrays,verify_prediction_array,summarize_joint,
                          extract_shared_features,fit_shared_probes,predict_probe_record,verify_joint_outputs,validate_probe_checkpoint)
from ..v26.source import file_sha,require_sha,write_json,source_digest,load_frozen_frontend,resolve_recording,require_frontend_assets

PREDECESSOR=dict(audit=None,prepare='audit',smoke='prepare',train='smoke',evaluate='train',verify='evaluate')


def validate_protocol(config):
    if config['arms']!=list(ARMS) or config['seeds']!=[42,43,44]:
        raise ValueError('locked protocol arms/seeds differ')
    validate_training_options(config['training'])


def begin_stage(root: Path,stage: str,identity: dict):
    root=Path(root)
    if stage not in PREDECESSOR:raise ValueError('unknown stage')
    root.mkdir(parents=True,exist_ok=True)
    started=root/f'{stage}-started.json'
    if started.exists() or (root/f'{stage}-completed.json').exists():
        raise ValueError('stage already exists; preserve partial evidence and use a new version')
    previous=PREDECESSOR[stage]
    if previous:
        path=root/f'{previous}-completed.json'
        if not path.is_file():raise ValueError(f'{stage} requires successful {previous}')
        value=json.loads(path.read_text())
        if not value['result'].get('passed'):raise ValueError('successful predecessor required')
        if value['identity']!=identity:raise ValueError('stage identity mismatch')
    # Exclusive create prevents two concurrent invocations from claiming a stage.
    with started.open('x',encoding='utf-8') as handle:
        json.dump(dict(identity=identity,stage=stage),handle,indent=2)


def finish_stage(root: Path,stage: str,identity: dict,result: dict):
    root=Path(root);started=json.loads((root/f'{stage}-started.json').read_text())
    if started['identity']!=identity or not result.get('passed'):
        raise ValueError('cannot complete failed or changed stage')
    path=root/f'{stage}-completed.json'
    with path.open('x',encoding='utf-8') as handle:
        json.dump(dict(identity=identity,result=result),handle,indent=2,allow_nan=False)


def _identity(config_path,config):
    identity=dict(code_sha256=source_digest(),config_sha256=file_sha(config_path))
    for key in ('stage1','stage2'):
        identity[key]=require_sha(config[key],config[key+'_sha256'])
    identity['original_protocol']=require_sha(Path(config['original_root'])/'protocol.json',config['original_protocol_sha256'])
    for role in ('train','validation'):
        identity[f'index_{role}']=require_sha(Path(config['original_root'])/f'cache-{role}'/'index.json',config['index_sha256'][role])
    for name,path in config['label_maps'].items():identity['map_'+name]=require_sha(path,config['label_map_sha256'])
    identity['frontend_assets']=frontend_identity(config)
    return identity


def frontend_identity(config):
    legacy=json.loads((Path(config['original_root'])/'protocol.json').read_text())['frontend_assets']
    require_frontend_assets(legacy)
    config_sha=require_sha(config['frontend_config'],legacy['config_sha256'])
    return dict(config_sha256=config_sha,model_dir=legacy['model_dir'],assets=legacy['assets'])


def verify_summary_artifact(root,recomputed):
    path=Path(root)/'evaluation-summary.json'
    if not path.is_file():raise ValueError('evaluation summary missing')
    completion=json.loads((Path(root)/'evaluate-completed.json').read_text())['result']
    require_sha(path,completion['summary_sha256'])
    if json.loads(path.read_text())!=recomputed:
        raise ValueError('evaluation summary differs from recomputed prediction arrays')


def _checkpoint(root,model,step):
    rows=json.loads((root/'train-completed.json').read_text())['result']['runs']
    matching=[r for r in rows if r['arm']==model.arm and r['seed']==model.seed]
    if len(matching)!=1:raise ValueError('completed training identity missing or duplicated')
    path=root/'train'/str(model.seed)/model.arm/f'step{step:04d}.pt'
    require_sha(path,matching[0]['checkpoint_sha256'][path.name])
    payload=torch.load(path,map_location='cpu',weights_only=False)
    validate_optimizer_checkpoint(payload,model,step)
    return payload


def _load_records(config):
    from ..v26.p1 import _load_original,_commands
    from ..v26.targets import prediction_contract
    original,_=_load_original(config)
    maps={key:json.loads(Path(path).read_text()) for key,path in config['label_maps'].items()}
    validate_class_maps(maps['legacy'],maps['expanded'])
    records={};protected={}
    for role in ('train','validation'):
        index_path=original/f'cache-{role}'/'index.json'
        index=json.loads(index_path.read_text());rows=[]
        manifest=Path(config['manifests'][role])
        manifest_rows=[json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
        protected[str(manifest)]=file_sha(manifest)
        for entry in index['records']:
            require_sha(entry['path'],entry['sha256'])
            raw=torch.load(entry['path'],map_location='cpu',weights_only=False)
            if raw['metadata'] not in manifest_rows:raise ValueError('cache source not in original role manifest')
            source=resolve_recording(manifest,raw['metadata'])
            require_sha(source,raw['source_sha256']);require_sha(raw['metadata']['source_commands'],raw['command_sha256'])
            with np.load(source,allow_pickle=False) as data:labels=torch.from_numpy(data['labels'].copy())
            observation=(raw['radar_mask'].numpy().all(-1) & np.isfinite(raw['radar'].numpy()).all((1,2)))
            physical=with_previous_observation(prediction_contract(raw['intervals'].numpy(),observation,_commands(raw)),raw['intervals'])
            raw['source_lock']={str(source):raw['source_sha256'],str(raw['metadata']['source_commands']):raw['command_sha256'],
                                entry['path']:entry['sha256']}
            record=build_joint_record(raw,labels,physical,role)
            for h in (1,3,5):
                record.groups[f'deadline_cross_h{h}']=physical['proxy_boundary_cross'][h]
                record.groups[f'mixed_current_h{h}']=physical['mixed_current'][h]
                record.groups[f'mixed_future_h{h}']=physical['mixed_future'][h]
            protected.update(record.source_lock);rows.append(record)
        records[role]=rows
    fit_people={r.subject for r in records['train']};val_people={r.subject for r in records['validation']}
    if fit_people & val_people:raise ValueError('subject crosses fit/validation roles')
    counts={role:(len(rows),sum(len(r.common_start) for r in rows),
                  sum(int(r.semantic_target[r.common_start].ge(0).sum()) for r in rows)) for role,rows in records.items()}
    if counts!={'train':(18,61299,59540),'validation':(6,20075,19609)}:
        raise ValueError(f'locked population counts differ: {counts}')
    normalization_path=original/'normalization.pt'
    protected[str(normalization_path)]=file_sha(normalization_path)
    stats=fit_motion_contract(records['train'],torch.load(normalization_path,map_location='cpu',weights_only=False))
    return records,stats,protected,counts


def _frontend(config):
    frontend,_,report=load_frozen_frontend(config['frontend_config'],config['stage1'],config['stage2'],
                         {k:config[k+'_sha256'] for k in ('stage1','stage2')},config['device'])
    if report['frontend_sha256']!=config['frontend_state_sha256']:
        raise ValueError('fitted frontend state differs from P1-C frozen reference')
    return frontend


def _prepared(root):
    completion=json.loads((root/'prepare-completed.json').read_text())['result']
    require_sha(root/'prepared.pt',completion['prepared_sha256'])
    value=torch.load(root/'prepared.pt',map_location='cpu',weights_only=False)
    verify_source_lock(value['protected'])
    return value['records'],value['stats']


def _model(frontend,config,stats,arm,seed):
    model=JointPilotModel(frontend,arm,seed=seed).to(config['device'])
    model.set_stats(stats)
    return model


def _smoke(frontend,config,root,records,stats):
    from .contract import prefix_bounds
    long=[]
    for ri,record in enumerate(records['train']):
        for t in record.common_start.tolist():
            if record.semantic_target[t]>=0 and t+1-prefix_bounds(record,t)[0]==32:
                long.append((ri,t))
                if len(long)==2:break
        if len(long)==2:break
    if len(long)!=2:raise RuntimeError('two maximal causal prefixes required for resource smoke')
    options=dict(config['training'],steps=5,max_attempts=15,smoke=True)
    results=[];device=config['device'];amp=device.startswith('cuda')
    for arm in ARMS:
        model=_model(frontend,config,stats,arm,1042)
        batch=make_joint_batch(records['train'],stats,long,device)
        model.eval()
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            initial=model.forward_prefix(batch['F'],batch['mask'],batch['m_ref'])
            if not torch.equal(model.shared_adapter(batch['F']),batch['F']):raise RuntimeError('Adapter not identity')
            if 'mean' in initial and not torch.equal(initial['mean'],batch['m_ref'][:,None].expand(-1,3,-1)):
                raise RuntimeError('initial forecast not exact Persistence')
            teacher=model.teacher_prefix(batch['F'],batch['mask']) if arm in ('B','AB') else {}
            initial_error=float((initial['logits']-teacher['logits']).abs().max()) if teacher else None
            if initial_error is not None and initial_error>1e-4:raise RuntimeError('initial semantic teacher mismatch')
        del initial
        run=train_joint_job(model,records['train'],stats,[long]*5,options,root/'smoke'/arm)
        model.train()
        with torch.autocast('cuda',dtype=torch.float16,enabled=amp):
            output=model.forward_prefix(batch['F'],batch['mask'],batch['m_ref'])
            losses=joint_losses(output,teacher,batch,stats,arm)
        gradient=gradient_diagnostic(model,losses,run['logs'][-1]['scaler_scale'])
        if arm in ('M','A','AB') and gradient['motion_norm']<=0:raise RuntimeError('motion to shared Adapter disconnected')
        if arm in ('B','AB') and gradient['semantic_norm']<=0:raise RuntimeError('semantics to shared Adapter disconnected')
        run.update(initial_semantic_error=initial_error,final_shared_gradient=gradient)
        results.append(run)
        del model,output,losses,batch
        if amp:torch.cuda.empty_cache()
    seconds=sum(r['seconds'] for r in results)
    return dict(passed=True,actual_optimizer_steps=sum(r['actual_optimizer_steps'] for r in results),
                runs=results,estimated_main_seconds=seconds/5*1000*3,
                longest_context=32,gradient_tail=16,batch=2)


def _train(frontend,config,root,records,stats):
    results=[]
    for seed in config['seeds']:
        schedule=json.loads((root/'schedules'/f'{seed}.json').read_text())
        expected=joint_schedule(records['train'],seed)
        if schedule!=json.loads(json.dumps(expected)):raise ValueError('locked sampling table changed')
        for arm in ARMS:
            model=_model(frontend,config,stats,arm,seed)
            results.append(train_joint_job(model,records['train'],stats,schedule,config['training'],root/'train'/str(seed)/arm))
            del model
            if config['device'].startswith('cuda'):torch.cuda.empty_cache()
    if sum(r['actual_optimizer_steps'] for r in results)!=12000:raise RuntimeError('formal training count mismatch')
    return dict(passed=True,actual_optimizer_steps=12000,runs=results)


def _evaluate(frontend,config,root,records,stats):
    results={};probe_steps=0;reference_done=set()
    for seed in config['seeds']:
        for arm in ARMS:
            job=root/'train'/str(seed)/arm
            model=_model(frontend,config,stats,arm,seed)
            checkpoint=_checkpoint(root,model,1000);model.load_learned_state(checkpoint['learned_state'])
            dest=root/'evaluation'/str(seed)/arm;dest.mkdir(parents=True,exist_ok=False)
            arrays=[]
            for record in records['validation']:
                prediction=predict_joint_record(model,record,stats)
                verify_prediction_array(prediction,reference_arrays(record,stats),arm)
                np.savez_compressed(dest/f'{record.uid}.npz',**prediction);arrays.append(prediction)
            results[f'{seed}/{arm}']=summarize_joint(arrays,stats)
            features=extract_shared_features(model.shared_adapter,records['train']+records['validation'],stats,config['device'])
            probe=fit_shared_probes(features,records['train'],stats,seed,device=config['device'])
            validate_probe_checkpoint(probe,seed);probe_steps+=sum(probe['actual_optimizer_steps'].values())
            torch.save(probe,dest/'probes.pt')
            probe_arrays=[]
            for record in records['validation']:
                prediction=predict_probe_record(probe,features,record,stats)
                np.savez_compressed(dest/f'probe-{record.uid}.npz',**prediction);probe_arrays.append(prediction)
            results[f'{seed}/{arm}/probes']=summarize_joint(probe_arrays,stats)
            if seed not in reference_done:
                ref_features={record.uid:(record.motion_ref-stats['mean'])/stats['std'] for role in records.values() for record in role}
                probe_ref=fit_shared_probes(ref_features,records['train'],stats,seed,device=config['device'])
                validate_probe_checkpoint(probe_ref,seed);probe_steps+=sum(probe_ref['actual_optimizer_steps'].values())
                ref_dest=root/'evaluation'/str(seed)/'reference';ref_dest.mkdir(parents=True,exist_ok=False)
                torch.save(probe_ref,ref_dest/'probes.pt');ref_arrays=[]
                for record in records['validation']:
                    prediction=predict_probe_record(probe_ref,ref_features,record,stats)
                    np.savez_compressed(ref_dest/f'probe-{record.uid}.npz',**prediction);ref_arrays.append(prediction)
                results[f'{seed}/reference/probes']=summarize_joint(ref_arrays,stats);reference_done.add(seed)
            write_json(root/'evaluation-progress.json',dict(completed=list(results),probe_optimizer_steps=probe_steps))
            del model,features,probe,arrays,probe_arrays
            if config['device'].startswith('cuda'):torch.cuda.empty_cache()
    if probe_steps!=6000:raise RuntimeError('probe budget mismatch')
    write_json(root/'evaluation-summary.json',results)
    return dict(passed=True,probe_optimizer_steps=probe_steps,result_keys=list(results),summary_sha256=file_sha(root/'evaluation-summary.json'),
                probe_sha256={p.relative_to(root).as_posix():file_sha(p) for p in (root/'evaluation').rglob('probes.pt')})


def _verify(frontend,config,root,records,stats):
    checks=[];checkpoints=[];recomputed={}
    evaluation=json.loads((root/'evaluate-completed.json').read_text())['result']
    for seed in config['seeds']:
        for arm in ARMS:
            job=root/'train'/str(seed)/arm
            model=_model(frontend,config,stats,arm,seed)
            for step in (0,100,300,1000):
                payload=_checkpoint(root,model,step)
                checkpoints.append(dict(path=str(job/f'step{step:04d}.pt'),sha256=file_sha(job/f'step{step:04d}.pt'),step=step))
            model.load_learned_state(payload['learned_state'])
            dest=root/'evaluation'/str(seed)/arm
            feature=extract_shared_features(model.shared_adapter,records['validation'],stats,config['device'])
            require_sha(dest/'probes.pt',evaluation['probe_sha256'][(dest/'probes.pt').relative_to(root).as_posix()])
            probe=torch.load(dest/'probes.pt',map_location='cpu',weights_only=False)
            validate_probe_checkpoint(probe,seed)
            native_arrays=[];probe_arrays=[]
            for record in records['validation']:
                for prefix in ('','probe-'):
                    path=dest/f'{prefix}{record.uid}.npz'
                    with np.load(path,allow_pickle=False) as data:actual={key:data[key] for key in data.files}
                    verify_prediction_array(actual,reference_arrays(record,stats),'probe' if prefix else arm)
                    replay=predict_probe_record(probe,feature,record,stats) if prefix else predict_joint_record(model,record,stats)
                    if set(actual)!=set(replay) or any(not np.array_equal(actual[key],replay[key]) for key in replay):
                        raise ValueError(f'saved weight replay mismatch: {path}')
                    checks.append(dict(path=str(path),sha256=file_sha(path),replayed=True))
                    (probe_arrays if prefix else native_arrays).append(actual)
            recomputed[f'{seed}/{arm}']=summarize_joint(native_arrays,stats)
            recomputed[f'{seed}/{arm}/probes']=summarize_joint(probe_arrays,stats)
            del model
            if config['device'].startswith('cuda'):torch.cuda.empty_cache()
        dest=root/'evaluation'/str(seed)/'reference'
        require_sha(dest/'probes.pt',evaluation['probe_sha256'][(dest/'probes.pt').relative_to(root).as_posix()])
        probe=torch.load(dest/'probes.pt',map_location='cpu',weights_only=False);validate_probe_checkpoint(probe,seed)
        features={record.uid:(record.motion_ref-stats['mean'])/stats['std'] for record in records['validation']}
        reference_predictions=[]
        for record in records['validation']:
            path=dest/f'probe-{record.uid}.npz'
            with np.load(path,allow_pickle=False) as data:actual={key:data[key] for key in data.files}
            replay=predict_probe_record(probe,features,record,stats)
            verify_prediction_array(actual,reference_arrays(record,stats),'probe')
            if any(not np.array_equal(actual[key],replay[key]) for key in replay):raise ValueError('reference probe replay mismatch')
            checks.append(dict(path=str(path),sha256=file_sha(path),replayed=True))
            reference_predictions.append(actual)
        recomputed[f'{seed}/reference/probes']=summarize_joint(reference_predictions,stats)
    verify_summary_artifact(root,recomputed)
    finite=verify_joint_outputs(root/'evaluation')
    return dict(passed=True,arrays=checks,checkpoints=checkpoints,finite=finite,summary_recomputed=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--mode',choices=list(PREDECESSOR),required=True);args=parser.parse_args()
    config_path=Path(args.config);config=yaml.safe_load(config_path.read_text())
    validate_protocol(config);torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(config['output_dir']);identity=_identity(config_path,config)
    begin_stage(root,args.mode,identity)
    try:
        if args.mode=='audit':
            records,stats,protected,counts=_load_records(config)
            write_json(root/'input-lock.json',dict(protected=protected,counts=counts))
            result=dict(passed=True,counts=counts,protected_files=len(protected),optimizer_updates=0,
                        input_lock_sha256=file_sha(root/'input-lock.json'))
        elif args.mode=='prepare':
            audit=json.loads((root/'audit-completed.json').read_text())['result']
            require_sha(root/'input-lock.json',audit['input_lock_sha256'])
            records,stats,protected,counts=_load_records(config)
            if protected!=json.loads((root/'input-lock.json').read_text())['protected']:raise ValueError('audit source identity changed')
            torch.save(dict(records=records,stats=stats,protected=protected),root/'prepared.pt')
            (root/'schedules').mkdir(exist_ok=False)
            for seed in config['seeds']:write_json(root/'schedules'/f'{seed}.json',joint_schedule(records['train'],seed))
            result=dict(passed=True,prepared_sha256=file_sha(root/'prepared.pt'),optimizer_updates=0,
                        schedule_sha256={str(seed):file_sha(root/'schedules'/f'{seed}.json') for seed in config['seeds']})
        else:
            records,stats=_prepared(root);frontend=_frontend(config)
            if args.mode=='smoke':result=_smoke(frontend,config,root,records,stats)
            elif args.mode=='train':result=_train(frontend,config,root,records,stats)
            elif args.mode=='evaluate':result=_evaluate(frontend,config,root,records,stats)
            else:result=_verify(frontend,config,root,records,stats)
        finish_stage(root,args.mode,identity,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('runs','arrays','checkpoints')},ensure_ascii=False),flush=True)
    except Exception as error:
        write_json(root/f'{args.mode}-failure.json',dict(type=type(error).__name__,message=str(error)))
        raise


if __name__=='__main__':main()
