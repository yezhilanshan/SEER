"""Frozen semantic/physical complementarity pilot: TRAIN-only CPU diagnostics."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from continuous_mmwave import parrad_stage3_complementarity as pilot
from continuous_mmwave.parrad_stage3_linear_probe import SUBJECTS
from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source, sha256
from continuous_mmwave.parrad_stage3_sequence_features import SOURCE_SETS, features_from_pack

CONTROLLER_SHA='b3c37594f78b85c2342fb5de17ff2662f48ed5cfd1adcbfdbce8f68dc8153b44'


def _write(file,value):
    file.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def code_hashes():
    code=Path(__file__).parent
    names=('parrad_stage3_complementarity.py','parrad_stage3_complementarity_runner.py',
           'parrad_stage3_radar_readout.py','parrad_stage3_linear_probe.py','parrad_stage3_candidate_audit.py',
           'parrad_stage3_sequence_data.py','parrad_stage3_sequence_features.py',
           'parrad_stage3_frozen_layers.py','parrad_controller.py','models/event_former.py')
    hashes={n:sha256(code/n) for n in names}
    if hashes['parrad_controller.py']!=CONTROLLER_SHA:
        raise ValueError('controller differs from frozen trace baseline')
    return hashes


def aggregate(records,*,paired=False):
    groups=defaultdict(list)
    fields=('group','comparison') if paired else ('group','source','history')
    for r in records: groups[tuple(r[f] for f in fields)].append(r)
    result=[]
    for identity,values in groups.items():
        by_seed=[]
        metric_names=list(values[0]['delta'] if paired else pilot.flattened_metrics(values[0]))
        for seed in pilot.SETTINGS['projection_seeds']:
            selected=[r for r in values if r['seed']==seed]; metrics={}
            for name in metric_names:
                numbers=[(r['delta'] if paired else pilot.flattened_metrics(r))[name] for r in selected]
                numbers=[v for v in numbers if v is not None]
                metrics[name]={'subject_macro':statistics.mean(numbers) if numbers else None,
                               'subjects_defined':len(numbers),'subjects_expected':3,
                               'subject_range':[min(numbers),max(numbers)] if numbers else None}
            by_seed.append({'seed':seed,'metrics':metrics})
        summary={**dict(zip(fields,identity)),'per_seed':by_seed,'metrics':{}}
        for name in metric_names:
            numbers=[r['metrics'][name]['subject_macro'] for r in by_seed if r['metrics'][name]['subject_macro'] is not None]
            summary['metrics'][name]={'seed_median':statistics.median(numbers) if numbers else None,
                                     'seed_range':[min(numbers),max(numbers)] if numbers else None}
        if not paired:
            summary['undefined_records']=sum(r['status']=='undefined' for r in values)
            summary['nonconverged_fits']=sum(r['converged'] is False for r in values)
        result.append(summary)
    return result


def group_pairs(group,lookup):
    result=[]
    for seed in pilot.SETTINGS['projection_seeds']:
        for subject in SUBJECTS:
            comparisons=[]
            for source in ('F','E','z','F+E','F+z'):
                comparisons.extend((f'{source}:K{k}-K1',(source,1),(source,k)) for k in group['histories'] if k!=1)
            for k in group['histories']:
                comparisons.extend((f'{fusion}-{part}:K{k}',(part,k),(fusion,k))
                                   for fusion,parts in (('F+E',('F','E')),('F+z',('F','z'))) for part in parts)
            for name,a,b in comparisons:
                left,right=lookup[(*a,seed,subject)],lookup[(*b,seed,subject)]
                result.append({'group':group['name'],'comparison':name,'seed':seed,'held_subject':subject,
                               'left_fit':left['fit_id'],'right_fit':right['fit_id'],
                               'endpoint_fingerprint':left['endpoint_fingerprint'],'role_fingerprint':left['role_fingerprint'],
                               'delta':pilot.paired_delta(left,right),
                               'both_converged':left['converged'] is True and right['converged'] is True})
    return result


def evaluate(source:Path,output:Path):
    source,output=Path(source),Path(output)
    if output.exists(): raise FileExistsError(output)
    checked=load_sequence_source(source); hashes=code_hashes(); grid=pilot.experiment_grid()
    torch.set_num_threads(pilot.SETTINGS['cpu_threads'])
    output.mkdir(parents=True,exist_ok=False); (output/'fits').mkdir(); (output/'cohorts').mkdir()
    upstream=checked['manifest']
    manifest={'status':'preparing','source_split':'train','diagnostic_only':True,'test_evaluation_allowed':False,
              'source_dir':str(source.resolve()),'source_manifest_sha256':checked['manifest_sha256'],
              'upstream_provenance':{k:v for k,v in upstream.items() if 'sha256' in k or k in
                  ('checkpoint','stage2_checkpoint','config','threshold_artifact','cache_fingerprint','layer_dimensions')},
              'code_sha256':hashes,'settings':pilot.SETTINGS,'grid':grid,'device':'cpu',
              'torch_version':torch.__version__,'numpy_version':np.__version__,
              'solver':'float64 LBFGS strong_wolfe; fit-only class balanced ridge logistic',
              'started_unix':time.time(),'completed_groups':[],'active_fit':None}
    _write(output/'complementarity-manifest.json',manifest)
    records=[]; pairs=[]; counts=[]; identifier=0
    try:
        with (output/'complementarity-results.jsonl').open('x',encoding='utf-8') as stream:
            for group in grid:
                started=time.monotonic()
                manifest.update(status='preparing',active_group=group['name'])
                _write(output/'complementarity-manifest.json',manifest)
                endpoints,excluded=pilot.select_group(checked,group)
                rows=[e.metadata for e in endpoints]; y=np.asarray([r['update_target'] for r in rows],bool)
                cohort_file=output/'cohorts'/f"{group['name']}.json"
                _write(cohort_file,[{'metadata':e.metadata,'history_rows':e.history_rows,'endpoint_row':e.endpoint_row} for e in endpoints])
                counts.append({'group':group['name'],'count':len(rows),'positives':int(y.sum()),'negatives':int((~y).sum()),
                               'prevalence':float(y.mean()) if len(y) else None,'exclusions':excluded,
                               'artifact':str(cohort_file.relative_to(output)),'artifact_sha256':sha256(cohort_file)})
                lookup={}
                for history in group['histories']:
                    packs=pilot.prepare(source,endpoints,history)
                    for sources in SOURCE_SETS:
                        for seed in pilot.SETTINGS['projection_seeds']:
                            pack=packs.get((sources,seed))
                            for subject in SUBJECTS:
                                fit,_,_=pilot.fold_masks(rows,subject)
                                scales=pilot.source_scales(pack,fit)
                                x=features_from_pack(pack,scales) if scales else np.empty((len(rows),130))
                                identifier+=1
                                descriptor={'group':group['name'],'cohort':group['cohort'],'source':'+'.join(sources),
                                            'history':history,'seed':seed,'held_subject':subject}
                                manifest.update(status='fitting',active_fit={'fit_id':f'{identifier:06d}',**descriptor})
                                _write(output/'complementarity-manifest.json',manifest)
                                record=pilot.fit_record(output,f'{identifier:06d}',descriptor,rows,y,x,scales)
                                stream.write(json.dumps(record,allow_nan=False)+'\n'); stream.flush(); records.append(record)
                                lookup[('+'.join(sources),history,seed,subject)]=record
                    del packs
                pairs.extend(group_pairs(group,lookup))
                manifest['completed_groups'].append({'name':group['name'],'seconds':time.monotonic()-started,'count':len(rows)})
                manifest['active_fit']=None
                _write(output/'endpoint-counts.json',counts); _write(output/'complementarity-manifest.json',manifest)
                print(json.dumps(manifest['completed_groups'][-1]),flush=True)
        final=load_sequence_source(source)
        if checked['manifest_sha256']!=final['manifest_sha256'] or hashes!=code_hashes():
            raise RuntimeError('source or frozen/runtime code changed during pilot')
        expected=sum(len(g['histories'])*len(SOURCE_SETS)*3*3 for g in grid)
        if len(records)!=expected or len(manifest['completed_groups'])!=len(grid):
            raise RuntimeError('incomplete pilot matrix')
        summary={'status':'complete','source_split':'train','diagnostic_only':True,'test_evaluation_allowed':False,
                 'source_manifest_sha256':checked['manifest_sha256'],'record_count':len(records),'expected_records':expected,
                 'fitted_count':sum(r['status']=='fitted' for r in records),'counts':counts,
                 'aggregates':aggregate(records),'paired':pairs,'paired_aggregates':aggregate(pairs,paired=True),
                 'nonconverged_fits':[r['fit_id'] for r in records if r['converged'] is False],
                 'undefined_records':[r['fit_id'] for r in records if r['status']=='undefined'],
                 'undefined_calibration':[{'fit_id':r['fit_id'],'point':label,'reason':p['reason']}
                     for r in records for label,p in r['operating_points'].items() if p['status']=='undefined'],
                 'interpretation':pilot.SETTINGS['interpretation'],
                 'decision_guard':'No best held threshold/direction selection. Inspect paired recall with actual held FPR, AP, all subjects/seeds; forced late gains do not imply natural early competence.'}
        _write(output/'complementarity-summary.json',summary)
        manifest.update(status='complete',completed_unix=time.time(),record_count=len(records),
                        summary_sha256=sha256(output/'complementarity-summary.json'),
                        results_sha256=sha256(output/'complementarity-results.jsonl'),counts_sha256=sha256(output/'endpoint-counts.json'))
        _write(output/'complementarity-manifest.json',manifest)
        return summary
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        _write(output/'complementarity-manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir',type=Path,required=True); parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args(); result=evaluate(args.source_dir,args.output_dir)
    print(json.dumps({'status':result['status'],'records':result['record_count'],'fitted':result['fitted_count']}))


if __name__=='__main__': main()
