"""Source-locked Stage2 initialization, pilot and matched Stage3 driver."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .joint_runner import prepare
from .reset_data import reset_targets,split_policy_records
from .reset_model import build_reset,detached
from .reset_training import cache_tokens,fit_heads,fit_policy,source_digest,working,load_working,_future_loss
from .reset_evaluation import rollout,evaluate_record
from .boundary import combine_scores
from .training import seed_everything
from .preprocess import write_json,sha256


def validate_config(c):
    expected={'seeds':[42,43,44],'arms':['control','full'],'head_steps':1000,'head_batch':64,
              'initialization_steps':300,'pilot_steps':50,'steps':1000,'chunk':32,'thresholds':[.25,.5,.75,.9]}
    if any(c[k]!=v for k,v in expected.items()): raise ValueError('fixed Reset experiment contract changed')


def choose_threshold(rows,budget):
    rows={float(k):v for k,v in rows.items()}
    feasible=[k for k,v in rows.items() if v['boundary']['fp_per_known_minute']<=budget]
    if feasible:
        chosen=max(feasible,key=lambda t:(rows[t]['boundary']['recall'],t))
        return chosen,True
    return min(rows,key=lambda t:(rows[t]['boundary']['fp_per_known_minute'],-t)),False


def binding(c):
    manifest=Path(__file__).resolve().parents[3]/'source-manifest.json'
    return {'config_sha256':hashlib.sha256(json.dumps(c,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            'source_manifest_sha256':sha256(manifest)}


def verify_replay(first,second):
    first=Path(first); second=Path(second); names=sorted(p.name for p in first.glob('record-*.npz'))
    if not names or names!=sorted(p.name for p in second.glob('record-*.npz')): raise RuntimeError('Reset replay file mismatch')
    maximum=0.
    for name in names:
        with np.load(first/name) as a,np.load(second/name) as b:
            if set(a.files)!=set(b.files): raise RuntimeError('Reset replay keys changed')
            for key in a.files:
                if a[key].shape!=b[key].shape or not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all():
                    raise RuntimeError('Reset replay shape/nonfinite mismatch')
                if a[key].size: maximum=max(maximum,float(np.max(np.abs(a[key]-b[key]))))
    if maximum!=0: raise RuntimeError(f'Reset replay output changed {maximum}')
    return maximum


def verify_sources(c):
    for row in [*c['locks'].values(),*c['sources'].values()]:
        if sha256(row['path'])!=row['sha256']: raise ValueError('Reset source lock changed')


def prepare_records(c,path):
    verify_sources(c); records,stats,_=prepare(c,path)
    metadata=json.loads(Path(c['locks']['continuous_manifest']['path']).read_text())
    for r,meta in zip(records,metadata):
        if r['file']!=meta['file']: raise ValueError('Reset metadata order changed')
        r['reset']=reset_targets(meta)
        np.testing.assert_array_equal(r['reset']['labels'],r['semantic'])
    fit,cal=split_policy_records(records)
    with np.load(c['sources']['motion_stats']['path']) as old:
        for key in stats: np.testing.assert_array_equal(stats[key],old[key])
    write_json(Path(path)/'reset-contract.json',{'fit_recordings':fit,'calibration_recordings':cal,
        'validation_recordings':[i for i,r in enumerate(records) if r['role']=='validation'],
        'calibration_pretraining_exposure':True,'test_signal_values_read':False,
        'boundary_positive':'official start and stop; same-update deduplicated','unknown_is_negative':False,
        'anchor_audit':[{'file':r['file'],'merged':r['reset']['merged_anchor_count'],
                        'censored':r['reset']['excluded_recording_edge_anchors'],
                        'groups':r['reset']['anchor_groups']} for r in records]})
    return records,stats,fit,cal


def mechanism_probe(model,record,stats,path,device='cuda'):
    model.phase('policy'); before=source_digest(model); model.train()
    if device=='cuda': torch.cuda.reset_peak_memory_stats()
    state=model.initial(1,device)
    with torch.no_grad():
        for t in range(8):
            state,_=model.step(record['za'][t:t+1].to(device),record['zb'][t:t+1].to(device),
                torch.as_tensor(record['normalized'][t:t+1],device=device),state,forced=torch.zeros(1,device=device))
        args=(record['za'][8:9].to(device),record['zb'][8:9].to(device),torch.as_tensor(record['normalized'][8:9],device=device),state)
        keep,a=model.step(*args,forced=torch.zeros(1,device=device))
        reset,b=model.step(*args,forced=torch.ones(1,device=device))
        torch.testing.assert_close(keep['G'],reset['G'],atol=0,rtol=0)
        torch.testing.assert_close(a['keep'],b['keep'],atol=0,rtol=0)
        torch.testing.assert_close(a['birth'],b['birth'],atol=0,rtol=0)
        gap=float((a['mean']-b['mean']).abs().max())
    if gap<=1e-9: raise RuntimeError('Reset does not affect actual future predictor')
    state=detached(state); final=None
    for t in range(8,40):
        state,final=model.step(record['za'][t:t+1].to(device),record['zb'][t:t+1].to(device),
                              torch.as_tensor(record['normalized'][t:t+1],device=device),state)
    target=torch.as_tensor(np.stack([record['normalized'][39+h] for h in (1,3,5)])[None],device=device)
    losses={'future':_future_loss(final['mean'],target,stats),'semantic':final['logits'].square().mean()}
    parameters=list(model.policy.parameters()); gradients={}
    for name,loss in losses.items():
        values=torch.autograd.grad(loss,parameters,retain_graph=True,allow_unused=True)
        finite=[g for g in values if g is not None]
        if not finite or not all(torch.isfinite(g).all() for g in finite): raise RuntimeError('nonfinite real Reset gradients')
        maximum=max(float(g.abs().max()) for g in finite)
        if maximum<=1e-12: raise RuntimeError(f'actual {name} misses Policy')
        gradients[name]=maximum
    if before!=source_digest(model): raise RuntimeError('mechanism probe changed frozen sources')
    peak=torch.cuda.max_memory_allocated() if device=='cuda' else 0
    if peak>9*1024**3: raise RuntimeError('Reset probe exceeds 9GiB allocated')
    report={'passed':True,'forced_reset_prediction_max_abs':gap,'policy_gradient_max':gradients,
            'source_unchanged':True,'global_preserved':True,'prior_distribution_decision_invariant':True,
            'unrolled_updates':32,'peak_cuda_allocated':peak}
    write_json(path,report); model.zero_grad(set_to_none=True); return report


def evaluate_set(model,records,ids,stats,path,threshold=.5,mode='autonomous',limit=None,device='cuda'):
    path=Path(path); path.mkdir(parents=True,exist_ok=False); rows={}
    for rid in ids:
        arr=rollout(model,records[rid],threshold,mode,limit,device)
        np.savez_compressed(path/f'record-{rid:02d}.npz',**arr)
        rows[str(rid)]=evaluate_record(arr,records[rid],stats)
        write_json(path/'status.json',{'completed_recordings':len(rows),'total_recordings':len(ids)})
    boundary=combine_scores([r['boundary'] for r in rows.values()])
    known=sum(2*r['n_updates']-r['boundary']['unknown_frames'] for r in rows.values())
    boundary['fp_per_known_minute']=boundary['fp']/(known/900) if known else None
    counts={k:sum(r['episodes'][k] for r in rows.values()) for k in ('tp','fp','fn','unscored_unknown')}
    counts['f1']=2*counts['tp']/max(1,2*counts['tp']+counts['fp']+counts['fn'])
    semantic=[r['semantic']['macro_f1'] for r in rows.values() if r['semantic'] is not None]
    instance_semantic=[r['instance_semantic']['macro_f1'] for r in rows.values() if r['instance_semantic'] is not None]
    result={'threshold':threshold,'mode':mode,'limit':limit,'records':rows,'boundary':boundary,'episodes':counts,
            'record_macro_f1':float(np.mean(semantic)) if semantic else None,
            'record_instance_macro_f1':float(np.mean(instance_semantic)) if instance_semantic else None,
            'record_skill':np.mean([r['skill'] for r in rows.values()],0).tolist()}
    write_json(path/'complete.json',result); return result


def run(c,output,prepare_only=False):
    validate_config(c); output=Path(output); started=time.time(); prior=None
    if output.exists():
        if prepare_only or not (output/'preflight.json').exists() or (output/'stage2a').exists(): raise RuntimeError('refusing duplicate/incomplete Reset run')
        prior=json.loads((output/'preflight.json').read_text())
        if any(prior.get(k)!=v for k,v in binding(c).items()): raise ValueError('prepared Reset config/source changed')
    else: output.mkdir(parents=True,exist_ok=False)
    try:
        if not (output/'config.json').exists(): write_json(output/'config.json',c)
        if prior:
            import tempfile
            with tempfile.TemporaryDirectory(prefix='reset_data_') as temporary:
                records,stats,fit,cal=prepare_records(c,Path(temporary)/'data')
                assert json.loads((output/'data/reset-contract.json').read_text())==json.loads((Path(temporary)/'data/reset-contract.json').read_text())
        else: records,stats,fit,cal=prepare_records(c,output/'data')
        validation=[i for i,r in enumerate(records) if r['role']=='validation']
        seed_everything(42,4); model=build_reset(c); model.variance.copy_(torch.as_tensor(np.exp(stats['logvar'][0]),device='cuda'))
        write_json(output/'status.json',{'phase':'caching_frozen_tokens','updated_unix':time.time()})
        cache_tokens(model,records,stats,'cuda')
        if not prior:
            result=mechanism_probe(model,records[fit[0]],stats,output/'mechanism-initial.json')
            write_json(output/'preflight.json',{'passed':True,**binding(c),**result})
        if prepare_only:
            write_json(output/'status.json',{'phase':'prepared','updated_unix':time.time()}); return
        write_json(output/'status.json',{'phase':'stage2a','updated_unix':time.time()})
        fit_heads(model,records,fit,stats,c['head_steps'],c['head_batch'],output/'stage2a')
        mechanism_probe(model,records[fit[0]],stats,output/'mechanism-fitted-heads.json')
        write_json(output/'status.json',{'phase':'stage2b','updated_unix':time.time()})
        fit_policy(model,records,fit,stats,c['initialization_steps'],c['chunk'],42,output/'stage2b',True,False,
                   checkpoints=(0,c['initialization_steps']))
        initial=working(model)
        mechanism_probe(model,records[fit[0]],stats,output/'mechanism-initialized-policy.json')
        pilots={}
        for arm in c['arms']:
            load_working(model,initial); seed_everything(42,4)
            write_json(output/'status.json',{'phase':'pilot','current_job':arm,'updated_unix':time.time()})
            pilots[arm]=fit_policy(model,records,fit,stats,c['pilot_steps'],c['chunk'],42,output/'pilot'/arm,False,arm=='full',
                                  checkpoints=(0,c['pilot_steps']))
        write_json(output/'pilot/complete.json',{'passed':True,'jobs':pilots,'technical_gate_only':True})
        jobs={}
        for seed in c['seeds']:
            for arm in c['arms']:
                name=f'{arm}-seed{seed}'; load_working(model,initial); seed_everything(seed,4)
                write_json(output/'status.json',{'phase':'stage3','current_job':name,'completed_jobs':len(jobs),'total_jobs':6,'updated_unix':time.time()})
                def evaluate(current,step,path):
                    return evaluate_set(current,records,validation,stats,path/f'evaluation-{step:04d}')
                done=fit_policy(model,records,fit,stats,c['steps'],c['chunk'],seed,output/'jobs'/name,False,arm=='full',evaluate=evaluate)
                job_path=output/'jobs'/name
                write_json(job_path/'status.json',{'phase':'replay','step':c['steps'],'attempts':c['steps']})
                saved=torch.load(job_path/'final.pt',map_location='cpu',weights_only=False)
                load_working(model,saved['working'])
                evaluate_set(model,records,validation,stats,job_path/'replay-validation')
                done['prediction_replay_max_abs']=verify_replay(job_path/'evaluation-1000',job_path/'replay-validation')
                write_json(job_path/'status.json',{'phase':'calibration','step':c['steps'],'attempts':c['steps']})
                calibration={}
                for threshold in c['thresholds']:
                    if any(records[i]['role']!='train' for i in cal): raise RuntimeError('calibration role error')
                    calibration[str(threshold)]=evaluate_set(model,records,cal,stats,output/'jobs'/name/f'calibration-{threshold}',
                                                            threshold=threshold,limit=1024)
                threshold,feasible=choose_threshold(calibration,10.)
                final=evaluate_set(model,records,validation,stats,output/'jobs'/name/'calibrated-validation',threshold)
                done.update(calibration_threshold=threshold,calibration_budget_met=feasible,calibrated_validation=final)
                write_json(job_path/'complete.json',done)
                write_json(job_path/'status.json',{'phase':'complete','step':c['steps'],'attempts':c['steps']})
                jobs[name]=done; write_json(output/'completed-jobs.json',jobs)
        references={}
        load_working(model,initial)
        for mode in ('oracle','always_keep','always_reset'):
            write_json(output/'status.json',{'phase':'reference','current_job':mode,'updated_unix':time.time()})
            references[mode]=evaluate_set(model,records,validation,stats,output/'references'/mode,mode=mode)
        verify_sources(c)
        result={'phase':'complete','jobs':jobs,'references':references,'stage3_successful_steps':sum(r['successful_steps'] for r in jobs.values()),
                'test_signal_values_read':False,'completed_unix':time.time(),'seconds':time.time()-started}
        write_json(output/'complete.json',result); write_json(output/'status.json',{'phase':'complete','updated_unix':time.time()})
    except BaseException as exc:
        row={'phase':'failed','error':repr(exc),'traceback':traceback.format_exc(),'updated_unix':time.time()}
        write_json(output/'failed.json',row); write_json(output/'status.json',row); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output',required=True)
    p.add_argument('--prepare-only',action='store_true'); args=p.parse_args()
    run(json.loads(Path(args.config).read_text()),args.output,args.prepare_only)
