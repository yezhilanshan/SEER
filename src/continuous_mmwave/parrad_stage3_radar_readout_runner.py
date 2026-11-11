"""Offline, train-only raw Radar readout pilot on immutable natural traces."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from continuous_mmwave import parrad_stage3_radar_readout as pilot
from continuous_mmwave.parrad_stage3_linear_probe import SUBJECTS, binary_metrics
from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source, select_endpoints, sha256
from continuous_mmwave.parrad_stage3_sequence_features import iter_history_blocks
from continuous_mmwave.parrad_stage3_sequence_probe import stratified_metrics

GROUPS=('natural-fixed-4','natural-fixed-6','natural-fixed-7','natural-terminal-min3')


def _write(file,value):
    file.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def prepare(source,checked):
    endpoints=[]
    for key,rows in checked['rows_by_record'].items():
        for age in range(3,14):
            chosen,_=select_endpoints(rows,checked['candidates_by_record'][key],cohort='natural',
                endpoint_kind='fixed_age',age=age,history_lengths=(3,),task='update')
            endpoints.extend(chosen)
    if not endpoints: raise ValueError('no eligible clean natural online updates')
    dimension=checked['manifest']['layer_dimensions']['F']
    current=np.empty((len(endpoints),3,dimension),np.float32)
    reference=np.empty_like(current)
    for indices,c,a in iter_history_blocks(source,endpoints,('F',),3):
        if c['F'].shape[2] != dimension: raise ValueError('Radar dimension differs from manifest')
        current[indices],reference[indices]=c['F'],a['F']
    rows=[e.metadata for e in endpoints]
    pilot.candidate_weights(rows)  # also reject duplicate endpoints before fitting
    targets=np.array([r['update_target'] for r in rows],bool)
    group_masks={f'natural-fixed-{age}':np.array([r['age_updates']==age for r in rows]) for age in (4,6,7)}
    group_masks['natural-terminal-min3']=np.array([r['terminal_decision'] in ('confirm','reject')
        and r['age_updates']==r['terminal_age'] for r in rows])
    return rows,current,reference,targets,group_masks


def aggregate(records):
    grouped=defaultdict(list)
    for r in records: grouped[r['group'],r['model']].append(r)
    out=[]
    for (group,model),values in grouped.items():
        per_seed=[]
        for seed in sorted({v['seed'] for v in values},key=str):
            vv=[v for v in values if v['seed']==seed]
            m={}
            for metric in ('auroc','average_precision','balanced_accuracy'):
                nums=[v['metrics'][metric] for v in vv if v['metrics'].get(metric) is not None]
                m[metric]={'subject_macro':statistics.mean(nums) if nums else None,'subjects_defined':len(nums)}
            ops={}
            for target in pilot.SETTINGS['target_fprs']:
                label=str(target)
                ops[label]={}
                for metric in ('false_positive_rate','recall','precision','balanced_accuracy'):
                    nums=[v['operating_points'][label][metric] for v in vv if v['operating_points'][label][metric] is not None]
                    ops[label][metric]={'subject_macro':statistics.mean(nums) if nums else None,'subjects_defined':len(nums)}
            per_seed.append({'seed':seed,'metrics':m,'operating_points':ops})
        result={'group':group,'model':model,'per_seed':per_seed,'metrics':{},'operating_points':{}}
        for metric in ('auroc','average_precision','balanced_accuracy'):
            nums=[v['metrics'][metric]['subject_macro'] for v in per_seed if v['metrics'][metric]['subject_macro'] is not None]
            result['metrics'][metric]={'seed_median':statistics.median(nums) if nums else None,'seed_range':[min(nums),max(nums)] if nums else None}
        for target in pilot.SETTINGS['target_fprs']:
            label=str(target); result['operating_points'][label]={}
            for metric in ('false_positive_rate','recall','precision','balanced_accuracy'):
                nums=[v['operating_points'][label][metric]['subject_macro'] for v in per_seed if v['operating_points'][label][metric]['subject_macro'] is not None]
                result['operating_points'][label][metric]={'seed_median':statistics.median(nums) if nums else None,'seed_range':[min(nums),max(nums)] if nums else None}
        out.append(result)
    return out


def paired(records):
    baselines={(r['group'],r['held_subject'],r['seed']):r for r in records if r['model']=='projected_logistic'}
    results=[]
    for r in records:
        if r['model']=='projected_logistic': continue
        seeds=pilot.SETTINGS['seeds'] if r['seed'] is None else [r['seed']]
        for seed in seeds:
            baseline=baselines.get((r['group'],r['held_subject'],seed))
            if baseline is None: continue
            if r['endpoint_fingerprint']!=baseline['endpoint_fingerprint']:
                raise RuntimeError('paired model comparisons differ in endpoints or labels')
            delta={k:r['metrics'][k]-baseline['metrics'][k] if r['metrics'][k] is not None and baseline['metrics'][k] is not None else None
                   for k in ('auroc','average_precision','balanced_accuracy')}
            results.append({'group':r['group'],'model':r['model'],'seed':seed,'held_subject':r['held_subject'],
                            'endpoint_fingerprint':r['endpoint_fingerprint'],'delta':delta})
    return results


def evaluate(source:Path,output:Path):
    source,output=Path(source),Path(output)
    if output.exists(): raise FileExistsError(output)
    checked=load_sequence_source(source)
    code=Path(__file__).parent
    code_hashes={f:sha256(code/f) for f in ('parrad_stage3_radar_readout.py','parrad_stage3_radar_readout_runner.py')}
    output.mkdir(parents=True,exist_ok=False); (output/'fits').mkdir()
    manifest={'status':'preparing','source_split':'train','diagnostic_only':True,'test_evaluation_allowed':False,
              'source_dir':str(source.resolve()),'source_manifest_sha256':checked['manifest_sha256'],
              'settings':json.loads(json.dumps(pilot.SETTINGS)),'model_specs':pilot.MODEL_SPECS,
              'code_sha256':code_hashes,'started_unix':time.time(),'fits':[],
              'torch_version':torch.__version__,'numpy_version':np.__version__,'device':'cpu',
              'calibration_policy':'set003 of the two fitting subjects; both sensors; never held-subject labels',
              'threshold_application':'one all-online calibration threshold per fit, shared across endpoint groups',
              'baseline_note':'old 0.679 is contextual only: all heads now fit identical Candidate-weighted online updates with session calibration holdout'}
    _write(output/'readout-manifest.json',manifest)
    results=[]
    try:
        rows,current,reference,y,groups=prepare(source,checked)
        ages=np.array([r['age_seconds'] for r in rows])
        ids=np.array(pilot.endpoint_ids(rows))
        _write(output/'endpoints.json',rows)
        manifest.update(status='fitting',online_count=len(rows),candidate_count=len({(r['recording_key'],r['candidate_id']) for r in rows}))
        _write(output/'readout-manifest.json',manifest)
        with (output/'readout-results.jsonl').open('x',encoding='utf-8') as stream:
            for held_subject in SUBJECTS:
                fit,cal,held=pilot.fold_masks(rows,held_subject)
                if not fit.any() or not cal.any() or not held.any() or not (~y[cal]).any():
                    raise ValueError(f'incomplete fitting/calibration/held fold {held_subject}')
                weights=pilot.candidate_weights([rows[i] for i in np.flatnonzero(fit)])
                residual=np.asarray(current[fit,-1],np.float64)-reference[fit,-1]
                scale=max(float(np.sqrt(np.average(np.square(residual).sum(axis=1),weights=weights))),1e-6)
                for name,kind in pilot.MODEL_SPECS:
                    seeds=pilot.SETTINGS['seeds'] if name=='projected_logistic' or kind=='mlp' else [None]
                    for seed in seeds:
                        started=time.monotonic()
                        identifier=f'{len(manifest["fits"])+1:03d}'
                        manifest['active_fit']={'fit_id':identifier,'model':name,'seed':seed,'held_subject':held_subject}
                        _write(output/'readout-manifest.json',manifest)
                        x=pilot.features(current,reference,ages,name,seed or pilot.SETTINGS['seeds'][0],scale)
                        fitted=pilot.fit_readout(x[fit],y[fit],weights,x,kind=kind,seed=seed or pilot.SETTINGS['seeds'][0])
                        scores=fitted['scores']
                        thresholds={str(rate):pilot.calibrate_threshold(y[cal],scores[cal],rate) for rate in pilot.SETTINGS['target_fprs']}
                        file=output/'fits'/f'{identifier}.npz'
                        arrays={
                            'fit_endpoint_ids':ids[fit],'calibration_endpoint_ids':ids[cal],'held_endpoint_ids':ids[held],
                            'fit_scores':scores[fit],'calibration_scores':scores[cal],'held_scores':scores[held],
                            'fit_targets':y[fit],'calibration_targets':y[cal],'held_targets':y[held],
                            'mean':fitted['mean'],'std':fitted['std'],'class_weights':fitted['class_weights'],
                            'base_weights':weights,'sample_weights':fitted['sample_weights'],'losses':fitted['losses'],
                            'source_scale':np.array(scale),'kind':np.array(kind),'hidden':np.array(fitted['hidden']),
                            'model':np.array(name),**{f'state_{k}':v for k,v in fitted['state'].items()}}
                        np.savez_compressed(file,**arrays)
                        fit_record={'fit_id':identifier,'model':name,'seed':seed,'held_subject':held_subject,
                            'fit_count':int(fit.sum()),'calibration_count':int(cal.sum()),'held_count':int(held.sum()),
                            'source_scale':scale,'parameter_count':fitted['parameter_count'],
                            'optimizer_status':fitted['optimizer_status'],'converged':fitted['converged'],
                            'iterations':fitted['iterations'],'gradient_inf':fitted['gradient_inf'],
                            'initial_recorded_loss':float(fitted['losses'][0]),'final_loss':float(fitted['losses'][-1]),
                            'fit_metrics':binary_metrics(y[fit].tolist(),scores[fit].tolist()),
                            'calibration_metrics':binary_metrics(y[cal].tolist(),scores[cal].tolist()),
                            'calibration_operating_points':{label:pilot.operating_metrics(y[cal],scores[cal],threshold) for label,threshold in thresholds.items()},
                            'thresholds':thresholds,'artifact':str(file.relative_to(output)),'artifact_sha256':sha256(file),
                            'seconds':time.monotonic()-started}
                        for group,mask in groups.items():
                            selected=held&mask
                            group_ids=ids[selected].tolist(); targets=y[selected].tolist(); predictions=scores[selected].tolist()
                            fingerprint=hashlib.sha256(json.dumps([group_ids,targets]).encode()).hexdigest()
                            group_rows=[rows[i] for i in np.flatnonzero(selected)]
                            record={**fit_record,'group':group,'endpoint_fingerprint':fingerprint,'held_group_count':int(selected.sum()),
                                'metrics':binary_metrics(targets,predictions),'strata':stratified_metrics(group_rows,targets,predictions),
                                'operating_points':{label:pilot.operating_metrics(targets,predictions,threshold) for label,threshold in thresholds.items()}}
                            stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush(); results.append(record)
                        manifest['fits'].append(fit_record)
                        manifest['active_fit']=None
                        _write(output/'readout-manifest.json',manifest)
                        print(json.dumps({'fit':identifier,'model':name,'seed':seed,'held':held_subject,
                            'optimizer_status':fitted['optimizer_status'],'seconds':fit_record['seconds']}),flush=True)
                        del x,fitted
        expected=sum(len(pilot.SETTINGS['seeds']) if name=='projected_logistic' or kind=='mlp' else 1 for name,kind in pilot.MODEL_SPECS)*len(SUBJECTS)
        if len(manifest['fits'])!=expected or len(results)!=expected*len(GROUPS):
            raise RuntimeError('incomplete readout grid')
        final=load_sequence_source(source)
        if final['manifest_sha256']!=checked['manifest_sha256'] or code_hashes!={f:sha256(code/f) for f in code_hashes}:
            raise RuntimeError('source or readout code changed during pilot')
        summary={'status':'complete','source_split':'train','diagnostic_only':True,'test_evaluation_allowed':False,
            'source_manifest_sha256':checked['manifest_sha256'],'fit_count':expected,'record_count':len(results),
            'online_count':len(rows),'candidate_count':manifest['candidate_count'],'aggregates':aggregate(results),'paired':paired(results),
            'nonconverged_logistic_fits':[f['fit_id'] for f in manifest['fits'] if f['converged'] is False],
            'interpretation':'Probe-only train-subject holdout; terminal timing and survivors depend on old controller. No production promotion or calibration on held subjects.'}
        _write(output/'readout-summary.json',summary)
        manifest.update(status='complete',completed_unix=time.time(),record_count=len(results),
            summary_sha256=sha256(output/'readout-summary.json'),results_sha256=sha256(output/'readout-results.jsonl'),
            endpoints_sha256=sha256(output/'endpoints.json'))
        _write(output/'readout-manifest.json',manifest)
        return summary
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        _write(output/'readout-manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=evaluate(args.source_dir,args.output_dir)
    print(json.dumps({'status':result['status'],'fit_count':result['fit_count'],'record_count':result['record_count']}))


if __name__=='__main__': main()
