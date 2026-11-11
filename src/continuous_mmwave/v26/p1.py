"""P1-A index contract audit and P1-B residual preparation on immutable v4 inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml
from .targets import prediction_contract, future_indices, build_command_episodes
from .predictors import make_predictor, make_residual_predictor
from .prediction_runner import ARMS, load_records, make_batch, forward, sample_schedule, predict_record
from .losses import gaussian_nll, reduce_prediction_losses
from .source import file_sha, require_sha, write_json, source_digest, require_frontend_assets, state_digest


def audit_prediction_record(record,command_starts):
    radar=record['radar'].numpy(); motion=record['motion'].numpy()
    observed=(record['radar_mask'].numpy().all(-1) & np.isfinite(radar).all(axis=(1,2)) & np.isfinite(motion).all(-1))
    value=prediction_contract(record['intervals'].numpy(),observed,command_starts,(1,3,5))
    selected=record['selected'].numpy(); old_future=record['future'].numpy()
    assert np.array_equal(value['future'][1][selected],old_future)
    assert value['event_assignment_valid'][selected].all()
    assert len(record['event']) == len(selected)
    arrays={key:value[key] for key in ('observation_valid','event_assignment_valid','proxy_episode_id','mixed_update','common_start')}
    arrays.update(intervals=record['intervals'].numpy(),command_starts=np.asarray(command_starts),old_selected=selected,old_future=old_future)
    report=dict(updates=len(motion),invalid_observation_updates=int((~value['observation_valid']).sum()),
                mixed_updates=int(value['mixed_update'].sum()),old_selected_preserved=len(selected),
                cached_event_current_positions=len(selected),common_starts=len(value['common_start']),horizons={})
    old_full=future_indices(record['intervals'].numpy(),record['valid'].numpy(),(1,3,5))
    for h,mapping in value['future'].items():
        good=np.flatnonzero(mapping >= 0); old_good=np.flatnonzero(old_full[h] >= 0)
        added=np.setdiff1d(good,old_good)
        assert np.array_equal(mapping[old_good],old_full[h][old_good])
        available=np.isin(good,selected)
        report['horizons'][str(h)]=dict(pairs=len(good),old_full_pairs=len(old_good),restored_vs_old_full=len(added),
              deadline_cross=int(value['proxy_boundary_cross'][h].sum()),
              mixed_current=int(value['mixed_current'][h].sum()),mixed_future=int(value['mixed_future'][h].sum()),
              boundary_support_union=int((value['proxy_boundary_cross'][h]|value['mixed_current'][h]|value['mixed_future'][h]).sum()),
              cached_event_available=int(available.sum()),missing_cached_event=int((~available).sum()))
        arrays[f'future_h{h}']=mapping; arrays[f'deadline_cross_h{h}']=value['proxy_boundary_cross'][h]
        arrays[f'mixed_current_h{h}']=value['mixed_current'][h];arrays[f'mixed_future_h{h}']=value['mixed_future'][h]
        arrays[f'restored_start_h{h}']=added
    return arrays,report


def _commands(record):
    require_sha(record['metadata']['source_commands'],record['command_sha256'])
    intervals=record['intervals'].numpy()
    episodes=build_command_episodes(record['metadata'],np.array([intervals[0,0],intervals[-1,1]-.09]))
    return np.array([row['start'] for row in episodes])


def _digest_array(value):
    x=value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(x.tobytes()).hexdigest()


def _load_original(config):
    require_sha(config['p0_snapshot'],config['p0_snapshot_sha256'])
    snapshot=json.loads(Path(config['p0_snapshot']).read_text())
    for path,sha in snapshot['protected_before'].items():require_sha(path,sha)
    root=Path(config['original_root']);lock=json.loads((root/'protocol.json').read_text())
    expected=config['original_protocol_sha256'];require_sha(root/'protocol.json',expected)
    for relative,sha in json.loads((root/'cache-lock.json').read_text()).items():require_sha(root/relative,sha)
    require_frontend_assets(lock['frontend_assets'])
    for name in ('stage1','stage2'):require_sha(config[name],lock['checkpoints'][name])
    return root,lock


def require_same_schedule(actual,expected):
    # JSON preserves order and integer indices while canonicalizing tuple/list containers.
    if json.dumps(actual,separators=(',',':')) != json.dumps(expected,separators=(',',':')):
        raise ValueError('original sample schedule differs')


def _fixed_probe(records,schedule):
    unique={tuple(pair) for batch in schedule for pair in batch}
    result=[]
    for ri,record in enumerate(records):
        indices=sorted(si for r,si in unique if r==ri)
        if not indices:continue
        row=dict(record)
        for key in ('selected','future','prefix_start','age','event'):row[key]=record[key][indices]
        result.append(row)
    return result


@torch.no_grad()
def _probe(model,records,stats,device,require_identity=False):
    rows=[]
    for record in records:
        mean,lv,target=predict_record(model,record,stats,device)
        x=(record['motion'][record['selected']]-stats['mean'])/stats['std']
        if require_identity and not torch.equal(mean,x):raise RuntimeError('zero-step residual identity failed')
        d=(target-x).double();a=(mean-x).double();err=(mean-target).double()
        rows.append(dict(uid=record['uid'],subject_id=record['metadata']['subject_id'],samples=len(mean),
                         mse=float(err.square().mean()),persistence_mse=float(d.square().mean()),
                         correction_energy=float(a.square().mean()),alignment=float((a*d).mean()),
                         nll=float(gaussian_nll(mean,lv,target).mean())))
    subjects={key:[r for r in rows if r['subject_id']==key] for key in sorted({r['subject_id'] for r in rows})}
    by_subject={key:{m:float(np.mean([r[m] for r in group])) for m in ('mse','persistence_mse','correction_energy','alignment','nll')}
                for key,group in subjects.items()}
    macro={key:float(np.mean([row[key] for row in by_subject.values()])) for key in next(iter(by_subject.values()))}
    return dict(recordings=rows,by_subject=by_subject,subject_macro=macro)


def run(config_path,mode):
    config_path=Path(config_path);config=yaml.safe_load(config_path.read_text())
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
    original,oldlock=_load_original(config);out=Path(config['output_dir'])
    if mode=='audit':
        out.mkdir(parents=True,exist_ok=False);(out/'indexes').mkdir()
        protected={str(original/'protocol.json'):file_sha(original/'protocol.json')}
        for p in original.glob('cache-*/*.pt'):protected[str(p)]=file_sha(p)
        for name in ('normalization.pt','persistence-logvar.pt','train/completed.json','evaluation/results.json'):
            protected[str(original/name)]=file_sha(original/name)
        for p in original.glob('train/*/*.pt'):protected[str(p)]=file_sha(p)
        protected.update(json.loads(Path(config['p0_snapshot']).read_text())['protected_before'])
        protocol=dict(config_sha256=file_sha(config_path),code_sha256=source_digest(),
              original_root=str(original),original_protocol_sha256=config['original_protocol_sha256'],
              original_code_sha256=oldlock['code_sha256'],protected=protected,
              primary_view='exact_original_selected_h1',event_reset='historical_command_proxy_oracle',
              new_pair_event_states='not_replayed_or_fabricated',formal_go=False,test_arrays_read=False)
        write_json(out/'protocol.json',protocol)
        audits={};index_rows=[]
        stats=torch.load(original/'normalization.pt',weights_only=False)
        for role in ('train','validation'):
            audits[role]=[]
            for record in load_records(original/f'cache-{role}/index.json'):
                arrays,report=audit_prediction_record(record,_commands(record));uid=record['uid']
                report.update(uid=uid,role=role,radar_sha256=_digest_array(record['radar']),
                              motion_sha256=_digest_array(record['motion']),event_sha256=_digest_array(record['event']))
                # Data-only baseline for the complete restored view. No head receives offline cross labels.
                report['persistence']={}
                for h in (1,3,5):
                    mapping=arrays[f'future_h{h}'];t=np.flatnonzero(mapping>=0);f=mapping[t]
                    d=((record['motion'][f]-record['motion'][t])/stats['std']).double()
                    per=d.square().mean(-1).numpy();cross=arrays[f'deadline_cross_h{h}'][t]
                    report['persistence'][str(h)]=dict(all_mse=float(per.mean()),
                          deadline_cross_mse=float(per[cross].mean()) if cross.any() else None)
                path=out/'indexes'/f'{role}-{uid}.npz';np.savez_compressed(path,**arrays)
                index_rows.append(dict(path=str(path.resolve()),sha256=file_sha(path)))
                audits[role].append(report)
                print(f'P1-A {role} {uid}: {report["horizons"]["1"]["deadline_cross"]} deadline-cross pairs',flush=True)
        assert all(file_sha(Path(p))==sha for p,sha in protected.items())
        result=dict(records=audits,cross_recording_pairs=0,feature_values_unchanged=True,
                    future_features_used_by_heads=False,event_extension='unavailable without audited state replay',
                    optimizer_steps=0,formal_go=False)
        write_json(out/'audit.json',result);write_json(out/'index-lock.json',index_rows)
        return result
    lock=json.loads((out/'protocol.json').read_text())
    if lock['code_sha256']!=source_digest() or lock['config_sha256']!=file_sha(config_path):raise ValueError('immutable P1 source/config differs')
    for p,sha in lock['protected'].items():require_sha(p,sha)
    for row in json.loads((out/'index-lock.json').read_text()):require_sha(row['path'],row['sha256'])
    records=load_records(original/'cache-train/index.json');stats=torch.load(original/'normalization.pt',weights_only=False)
    options=config['training'];schedule=sample_schedule(records,options['steps'],options['batch_size'],options['seed'])
    for arm in ARMS:
        require_same_schedule(schedule,json.loads((original/f'train/{arm}/training.json').read_text())['schedule'])
    probes=_fixed_probe(records,schedule)
    if mode=='prepare':
        destination=out/'prepare';destination.mkdir(exist_ok=False);result={}
        for arm in ARMS:
            folder=destination/arm;folder.mkdir()
            torch.manual_seed(options['seed']);model=make_residual_predictor(arm,oldlock['architectures'][arm]).to(config['device'])
            count=sum(p.numel() for p in model.parameters())
            assert count==oldlock['architectures'][arm]['parameter_count']
            zero=_probe(model.eval(),probes,stats,config['device'],require_identity=True)
            torch.save(dict(model_state=model.state_dict(),architecture=model.config,kind=arm,parameterization='residual_zero',
                            actual_optimizer_steps=0,code_sha256=source_digest()),folder/'step000.pt')
            model.train();optimizer=torch.optim.AdamW(model.parameters(),lr=options['lr'],weight_decay=options['weight_decay'])
            batch=make_batch(records,stats,schedule[0],config['device']);optimizer.zero_grad(set_to_none=True)
            mu,lv=forward(model,batch);loss=reduce_prediction_losses(gaussian_nll(mu,lv,batch['target']),batch['recording_ids'],
                                                                 torch.ones(len(mu),dtype=torch.bool,device=config['device']))
            if not torch.isfinite(loss):raise RuntimeError('nonfinite smoke loss')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),options['clip_norm'])
            assert torch.isfinite(norm) and model.head[-1].weight.grad[:model.config['motion_dim']].abs().sum()>0
            hidden_gradient_zero=bool(model.head[0].weight.grad.abs().sum()==0)
            optimizer.step();assert all(int(v['step'])==1 for v in optimizer.state.values())
            torch.save(dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),kind=arm,architecture=model.config,
                            parameterization='residual_zero',actual_optimizer_steps=1,code_sha256=source_digest()),folder/'smoke.pt')
            result[arm]=dict(parameters=count,zero_identity_exact=True,zero_probe=zero,smoke_loss=float(loss.detach()),
                       gradient_norm=float(norm),first_hidden_gradient_zero=hidden_gradient_zero,
                       initial_sha256=file_sha(folder/'step000.pt'),smoke_sha256=file_sha(folder/'smoke.pt'))
            print(f'P1-B {arm}: exact zero identity and 1-step CUDA smoke passed',flush=True)
        write_json(destination/'completed.json',result)
        write_json(out/'prepare-lock.json',{str(p.relative_to(out)):file_sha(p) for p in destination.rglob('*') if p.is_file()})
        return result
    raise ValueError('supported preparatory modes: audit, prepare')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--mode',required=True,choices=('audit','prepare'))
    args=parser.parse_args();print(json.dumps(run(args.config,args.mode),ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':main()
