"""Controlled pilot, exact resume audit and subject-grouped formal experiments."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from .training import (RecordingSampler,seed_everything,make_training_state,optimize_step,
                       save_checkpoint,load_checkpoint,bn_snapshot,bn_feature_diagnostic)
from .evaluation import EvaluationPool,evaluate
from .audit import write_json
from .model import parameter_audit,SplitBiGRU


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_fingerprint(train,val):
    return {role:[{'recording_id':r['recording_id'],'subject_id':r['subject_id'],
                   'sha256':r.get('npz_sha256') or digest(r['path'])} for r in sorted(rows,key=lambda r:r['recording_id'])]
            for role,rows in (('train',train),('validation',val))}


def source_code_hashes():
    return {p.name:digest(p) for p in sorted(Path(__file__).parent.glob('*.py'))}


def append_json(path,row):
    with Path(path).open('a',encoding='utf-8') as f:f.write(json.dumps(row,allow_nan=False)+'\n')


def cpu_model(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}


def run_training(kind,train_rows,val_rows,output,config,*,device='cuda',resume=None,capture_step=None):
    if set(r['subject_id'] for r in train_rows)&set(r['subject_id'] for r in val_rows):
        raise ValueError('train/validation subject leakage')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    config=copy.deepcopy(config);seed_everything(config['seed'])
    net,opt,sched=make_training_state(kind,device)
    sampler=RecordingSampler(train_rows,seed=config['seed'],min_length=config.get('min_length',100),max_length=config.get('max_length',1000))
    valpool=EvaluationPool(val_rows)
    provenance={'experiment':'split_birnn_reconstructed','kind':kind,'config':config,
                'source_roles':source_fingerprint(train_rows,val_rows),'code_hashes':source_code_hashes(),
                'clock':'approximate_0.09s_first_command_origin','hospital_status':'unconfirmed',
                'cnn_trainable':7283,'criterion':'pooled_coverage_matched_final_weighted_f1',
                'sampling':sampler.expected_distribution(),'torch':torch.__version__,'device':str(device)}
    write_json(output/'provenance.json',provenance)
    best_score,best_loss,best_step,bad_checks,best_model=-1.,float('inf'),0,0,None
    start=0
    if resume:
        state=load_checkpoint(resume,net,opt,sched,sampler);extra=state['extra']
        if extra['provenance']!=provenance:raise ValueError('resume configuration, source or code provenance changed')
        start=state['step'];best_score=extra['best_score'];best_loss=extra['best_loss'];best_step=extra['best_step']
        bad_checks=extra['bad_checks'];best_model=extra['best_model']
    if start>=config['max_optimizer_steps']:raise ValueError('resume step outside training budget')
    fixed=torch.from_numpy(sampler.arrays[0][0][:min(512,len(sampler.arrays[0][0]))]).unsqueeze(0)
    write_json(output/'bn-initial.json',{'snapshot':bn_snapshot(net),'feature_diagnostic':bn_feature_diagnostic(net,fixed)})
    wall=time.perf_counter();step=start;last_validation=None
    for step in range(start+1,config['max_optimizer_steps']+1):
        sample=sampler.sample(config['batch_size']);entry=optimize_step(net,opt,sched,sample,device=device)
        entry.update(step=step,elapsed_s=time.perf_counter()-wall)
        append_json(output/'train.jsonl',entry)
        if step%64==0:
            snapshot=bn_snapshot(net)
            if any(not np.isfinite(v['variance']).all() or np.min(v['variance'])<=0 for v in snapshot.values()):
                raise ValueError('invalid BN running variance')
            append_json(output/'bn.jsonl',{'step':step,'snapshot':snapshot,'feature_diagnostic':bn_feature_diagnostic(net,fixed)})
        do_validation=step%config['validation_interval_steps']==0 or step==config['max_optimizer_steps']
        if do_validation:
            before=bn_snapshot(net)
            last_validation=evaluate(net,valpool,window=config.get('window',512),hop=config.get('hop',171))
            if bn_snapshot(net)!=before:raise ValueError('validation mutated BN statistics')
            score=last_validation['score'];loss=last_validation['loss']
            if score is None or loss is None:raise ValueError('empty validation predictions or loss')
            improved=score>best_score or (score==best_score and loss<best_loss)
            if improved:
                best_score,best_loss,best_step,bad_checks=score,loss,step,0;best_model=cpu_model(net)
            elif step>=config['minimum_steps_before_early_stopping']:bad_checks+=1
            append_json(output/'validation.jsonl',{'step':step,'score':score,'loss':loss,
                'branch_losses':last_validation['branch_losses'],'metrics':last_validation['metrics'],
                'best_step':best_step,'bad_checks':bad_checks,'BN_unchanged':True})
            print(json.dumps({'kind':kind,'step':step,'train_loss':entry['loss'],'val_score':score,
                              'val_loss':loss,'best_step':best_step,'elapsed_s':time.perf_counter()-wall}),flush=True)
        extra={'provenance':provenance,'best_score':best_score,'best_loss':best_loss,'best_step':best_step,
               'bad_checks':bad_checks,'best_model':best_model}
        if do_validation or step==capture_step:
            save_checkpoint(output/'last.pt',net,opt,sched,sampler,step=step,extra=extra)
        if step==capture_step:save_checkpoint(output/f'step-{step}.pt',net,opt,sched,sampler,step=step,extra=extra)
        if do_validation and bad_checks>=config['early_stopping_patience_checks']:
            break
    # If early stop happened at validation, last.pt already contains the exact state.
    if best_model is None:raise ValueError('no checkpoint selected')
    best={'model':best_model,'kind':kind,'step':best_step,'provenance':provenance}
    torch.save(best,output/'best.pt')
    trained=cpu_model(net);net.load_state_dict(best_model);net.eval()
    yy_before=net.forward_sequence(fixed.to(device))['forward'].detach() if kind=='split' else net(fixed.to(device)).detach()
    restored,_,_=make_training_state(kind,device);restored.load_state_dict(torch.load(output/'best.pt',map_location='cpu',weights_only=False)['model']);restored.eval()
    with torch.no_grad():yy_after=restored.forward_sequence(fixed.to(device))['forward'] if kind=='split' else restored(fixed.to(device))
    replay={'parameters_exact':all(torch.equal(v,restored.state_dict()[k].cpu()) for k,v in best_model.items()),
            'predictions_exact':bool(torch.equal(yy_before,yy_after)),'selected_step':best_step}
    if not all(replay[k] for k in ('parameters_exact','predictions_exact')):raise ValueError('checkpoint replay mismatch')
    net.load_state_dict(trained)
    write_json(output/'sampling.json',{'expected':sampler.expected_distribution(),'actual':sampler.state_dict()})
    write_json(output/'bn-final.json',{'snapshot':bn_snapshot(net),'feature_diagnostic':bn_feature_diagnostic(net,fixed)})
    summary={'kind':kind,'steps':step,'best_step':best_step,'best_score':best_score,'best_loss':best_loss,
             'elapsed_s':time.perf_counter()-wall,'stopped_early':step<config['max_optimizer_steps'],
             'checkpoint_replay':replay,'best_sha256':digest(output/'best.pt'),'last_sha256':digest(output/'last.pt')}
    write_json(output/'summary.json',summary)
    return summary


def tree_equal(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,np.ndarray):return isinstance(b,np.ndarray) and np.array_equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return type(a)==type(b) and len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def load_sources(data_root):
    data_root=Path(data_root)
    rows=[json.loads(s) for s in (data_root/'manifest.jsonl').read_text().splitlines()]
    rows=[{**r,'path':str((data_root/r['path']).resolve())} for r in rows]
    folds=json.loads((data_root/'candidate_folds.json').read_text())['folds']
    for f in folds:
        roles=[set(f[k+'_subjects']) for k in ('train','validation','test')]
        if any(roles[i]&roles[j] for i in range(3) for j in range(i)):raise ValueError('fold subjects overlap')
        if set.union(*roles)!={r['subject_id'] for r in rows}:raise ValueError('fold subjects not exhaustive')
    return rows,folds


def role_rows(rows,fold,role):
    ids=set(fold[role+'_recordings']);out=[r for r in rows if r['recording_id'] in ids]
    if set(r['recording_id'] for r in out)!=ids or {r['subject_id'] for r in out}!=set(fold[role+'_subjects']):
        raise ValueError('role source identity mismatch')
    return out


def run_pilot(data_root,output,*,device='cuda'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows,folds=load_sources(data_root);fold=folds[0]
    train,val=role_rows(rows,fold,'train'),role_rows(rows,fold,'validation')
    config={'seed':42,'max_optimizer_steps':300,'validation_interval_steps':100,'batch_size':8,
            'minimum_steps_before_early_stopping':10000,'early_stopping_patience_checks':15,'window':512,'hop':171}
    full=run_training('split',train,val,output/'uninterrupted',config,device=device,capture_step=150)
    resumed=run_training('split',train,val,output/'resumed',config,device=device,resume=output/'uninterrupted'/'step-150.pt')
    a=torch.load(output/'uninterrupted'/'last.pt',map_location='cpu',weights_only=False)
    b=torch.load(output/'resumed'/'last.pt',map_location='cpu',weights_only=False)
    exact={key:tree_equal(a[key],b[key]) for key in ('model','optimizer','scheduler','sampler','rng','step','extra')}
    if not all(exact.values()):raise ValueError(f'nonexact pilot resume: {exact}')
    net,_,_=make_training_state('split',device)
    net.load_state_dict(torch.load(output/'uninterrupted'/'best.pt',map_location='cpu',weights_only=False)['model'])
    validation=evaluate(net,EvaluationPool(val),output=output/'validation-predictions',diagnostics=True)
    result={'passed':True,'scope':'300step_fold0_training_chain','fold':fold,'steps':300,
            'resume_from_step':150,'resume_exact':exact,'uninterrupted':full,'resumed':resumed,
            'validation_score':validation['score'],'criterion':'matched_final_weighted_f1',
            'test_not_evaluated':True,'source_code_hashes':source_code_hashes(),
            'data_manifest_sha256':digest(Path(data_root)/'manifest.jsonl'),
            'folds_sha256':digest(Path(data_root)/'candidate_folds.json')}
    write_json(output/'pilot-proof.json',result)
    return result


def run_formal(data_root,output,pilot_proof,*,device='cuda'):
    proof=json.loads(Path(pilot_proof).read_text())
    if not proof['passed'] or not all(proof['resume_exact'].values()) or proof['source_code_hashes']!=source_code_hashes():
        raise ValueError('pilot gates or source identity not met')
    if (proof['data_manifest_sha256']!=digest(Path(data_root)/'manifest.jsonl')
            or proof['folds_sha256']!=digest(Path(data_root)/'candidate_folds.json')):raise ValueError('data/folds changed after pilot')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows,folds=load_sources(data_root)
    config={'max_optimizer_steps':6400,'validation_interval_steps':128,'batch_size':8,
            'minimum_steps_before_early_stopping':1280,'early_stopping_patience_checks':15,'window':512,'hop':171,
            'seed':42}
    write_json(output/'frozen-config.json',{'config':config,'experiment':'split_birnn_reconstructed',
        'formal_training_enabled':True,'source_code_hashes':source_code_hashes(),'pilot_proof_sha256':digest(pilot_proof),
        'data_manifest_sha256':proof['data_manifest_sha256'],'folds_sha256':proof['folds_sha256'],
        'selection':'validation matched/final pooled F1','diagnostics':'prespecified_shift_collar_tail_no_tuning'})
    write_json(output/'frozen-folds.json',folds)
    results=[]
    for fold in folds:
        train,val,test=(role_rows(rows,fold,role) for role in ('train','validation','test'))
        for kind in ('split','baseline'):
            d=output/f"fold{fold['fold']}"/kind;d.parent.mkdir(exist_ok=True)
            cfg={**config,'seed':42+fold['fold']}
            summary=run_training(kind,train,val,d,cfg,device=device)
            net,_,_=make_training_state(kind,device)
            net.load_state_dict(torch.load(d/'best.pt',map_location='cpu',weights_only=False)['model'])
            result=evaluate(net,EvaluationPool(test),output=d/'test-predictions',diagnostics=True)
            # Save validation predictions for reproducibility without selecting from test.
            evaluate(net,EvaluationPool(val),output=d/'validation-predictions',diagnostics=False)
            results.append({'fold':fold['fold'],'kind':kind,'summary':summary,'metrics':result['metrics'],'coverage':result['coverage']})
            write_json(output/'progress.json',{'completed':results,'target_runs':10,'state':'running'})
            del net
            if torch.cuda.is_available():torch.cuda.empty_cache()
    write_json(output/'progress.json',{'completed':results,'target_runs':10,'state':'complete'})
    return results


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('pilot','formal'))
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pilot-proof',type=Path);p.add_argument('--device',default='cuda')
    a=p.parse_args()
    if a.mode=='pilot':run_pilot(a.data_root,a.output,device=a.device)
    else:
        if not a.pilot_proof:p.error('--pilot-proof required')
        run_formal(a.data_root,a.output,a.pilot_proof,device=a.device)
