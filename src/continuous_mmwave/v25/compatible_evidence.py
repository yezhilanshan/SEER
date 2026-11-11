"""Development-only Evidence500 on a frozen positive-coordinate frontend."""
from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

import numpy as np
import torch

from .cache import BalancedRunSampler
from .models import PositiveCoordinateGate, install_positive_coordinate_gate
from .checkpoint import file_sha, module_digest, state_digest
from .data import require_array_access, DATASET_V2, SENSORS


def validate_source_roles(index):
    if any(row['role'] not in ('fit','dev') for row in index['records']+index['runs']):
        raise ValueError('only fit/dev sources permitted')
    records={r['uid']:r for r in index['records']}
    if len(records)!=len(index['records']):
        raise ValueError('duplicate source records')
    for row in index['records']:
        require_array_access(row['metadata'],allowed_roles=('fit','dev_regression','dev_new'))
        role='dev' if row['metadata']['role'] in ('dev_regression','dev_new') else row['metadata']['role']
        if row['role']!=role:
            raise ValueError('record role differs from locked metadata partition')
    for run in index['runs']:
        source=records[run['record_uid']]
        if run['role']!=source['role'] or any(run[k]!=source['metadata'][k] for k in ('subject_id','sensor_id','session_id')):
            raise ValueError('run/source identity mismatch')
    fit_cells={(r['subject_id'],r['sensor_id']) for r in index['runs'] if r['role']=='fit'}
    if fit_cells!={(s,sensor) for s in DATASET_V2['fit'] for sensor in SENSORS}:
        raise ValueError('locked fit subject/sensor cells missing')


def validate_event_features(payload, frontend_digest):
    if payload.get('architecture')!=PositiveCoordinateGate.architecture or payload.get('schema_version')!=25:
        raise ValueError('exact compatible feature architecture required')
    if payload.get('frontend_digest')!=frontend_digest:
        raise ValueError('feature frontend digest mismatch')
    seen=set()
    for row in payload['events']:
        if row['role'] not in ('fit','dev') or row['uid'] in seen:
            raise ValueError('fit/dev unique events required')
        seen.add(row['uid'])
        for path in ('stream','commit'):
            if (row.get(path+'_visible_range')!=[row.get('start'),row.get('end')]
                or row.get(path+'_target_range')!=[2*row.get('start',-1),2*row.get('end',-1)]):
                raise ValueError('explicit event coverage required')
            if not torch.isfinite(row[path+'_tokens']).all():
                raise ValueError('nonfinite frozen event tokens')
        if row['target'].shape!=(4,) or row['target_mask'].shape!=(4,) or not torch.isfinite(row['target']).all():
            raise ValueError('four finite proxy targets required')
    if not seen:
        raise ValueError('event features missing')
    return payload


class PairedEvidenceBatches:
    """Eight balanced fit events each appear on both paths: effective batch16."""
    def __init__(self, events, *, seed, device):
        self.events=[r for r in events if r['role']=='fit']
        self.sampler=BalancedRunSampler(self.events,seed=seed)
        self.device=device
        self.counts=Counter(); self.draws=[]

    def __call__(self, stage):
        rows=self.sampler.next_group(8)
        paired=[(row,path) for row in rows for path in ('stream','commit')]
        self.draws.extend(row['uid'] for row in rows)
        for row,path in paired:
            self.counts[(row['subject_id'],row['sensor_id'],path)]+=1
        tokens=torch.stack([row[path+'_tokens'].detach() for row,path in paired]).to(self.device)
        return stage.evidence_step(dict(tokens=tokens,
            target=torch.stack([row['target'] for row,path in paired]).to(self.device),
            mask=torch.stack([row['target_mask'] for row,path in paired]).to(self.device)))['loss']

    def coverage(self):
        return [dict(subject_id=s,sensor_id=sensor,path=path,draws=count)
                for (s,sensor,path),count in sorted(self.counts.items())]


def score_evidence_rows(fit, predictions):
    """NumPy median from fit only; preserve missing baselines and empty groups."""
    medians={}
    for sensor in sorted({r['sensor_id'] for r in fit}):
        rows=[r for r in fit if r['sensor_id']==sensor]
        medians[sensor]=[float(np.median([r['target'][j] for r in rows if r['mask'][j]]))
            if any(r['mask'][j] for r in rows) else None for j in range(4)]
    saved=copy.deepcopy(predictions)
    for row in saved:
        row['baseline']=medians.get(row['sensor_id'],[None]*4)
        for j in range(4):
            if row['mask'][j] and row['baseline'][j] is None:
                raise ValueError('valid dev attribute has no fit baseline')
            if row['mask'][j] and not all(np.isfinite(row[k][j]) for k in ('prediction','target','baseline')):
                raise ValueError('nonfinite valid Evidence prediction/target/baseline')
    def measure(rows):
        errors=[abs(r['prediction'][j]-r['target'][j]) for r in rows for j in range(4) if r['mask'][j]]
        baseline=[abs(r['baseline'][j]-r['target'][j]) for r in rows for j in range(4) if r['mask'][j]]
        mae=float(np.mean(errors)) if errors else None
        constant=float(np.mean(baseline)) if baseline else None
        gain=None if constant is None or constant==0 else 1-mae/constant
        return dict(count=len(errors),masked_mae=mae,fit_median_baseline_mae=constant,
                    improvement_fraction=gain,passed=bool(gain is not None and gain>=.1-1e-12))
    groups={'all':saved}
    for key in ('sensor_id','subject_id','path'):
        prefix={'sensor_id':'sensor','subject_id':'subject','path':'path'}[key]
        for value in sorted({r[key] for r in saved}):
            groups[f'{prefix}/{value}']=[r for r in saved if r[key]==value]
    for subject,sensor in sorted({(r['subject_id'],r['sensor_id']) for r in saved}):
        groups[f'subject-sensor/{subject}/{sensor}']=[r for r in saved if (r['subject_id'],r['sensor_id'])==(subject,sensor)]
    attrs={}
    for name,rows in groups.items():
        attrs[name]=[]
        for j in range(4):
            valid=[r for r in rows if r['mask'][j]]
            attrs[name].append(dict(attribute=j,count=len(valid),
                mae=float(np.mean([abs(r['prediction'][j]-r['target'][j]) for r in valid])) if valid else None,
                baseline_mae=float(np.mean([abs(r['baseline'][j]-r['target'][j]) for r in valid])) if valid else None))
    return dict(**measure(saved),groups={k:measure(v) for k,v in groups.items()},
                group_attribute_mae=attrs,predictions=saved,fit_medians=medians)


@torch.no_grad()
def evaluate_evidence(stage, events):
    stage.eval()
    fit=[dict(sensor_id=e['sensor_id'],target=e['target'].tolist(),mask=e['target_mask'].tolist())
         for e in events if e['role']=='fit']
    dev=[e for e in events if e['role']=='dev']
    if not fit or not dev:
        raise ValueError('Evidence fit/dev missing')
    device=next(stage.parameters()).device
    rows=[]
    for event in dev:
        for path in ('stream','commit'):
            value=stage.heads.evidence(event[path+'_tokens'][None].to(device))[0].cpu()
            rows.append(dict(uid=event['uid'],path=path,prediction=value.tolist(),target=event['target'].tolist(),
                mask=event['target_mask'].tolist(),sensor_id=event['sensor_id'],subject_id=event['subject_id'],
                visible_range=event[path+'_visible_range'],target_range=event[path+'_target_range']))
    return score_evidence_rows(fit,rows)


def paired_retrieval(events, reference, output):
    """New features must match the prior C-exact keys, with all cells reported."""
    from .training import fixed_query_retrieval
    from .prepare import write_json
    rows={r['uid']:r for r in events if r['role']=='dev'}
    if set(rows)!=set(reference['uid'].tolist()):
        raise ValueError('diagnostic/current event UID mismatch')
    arrays={k:reference[k].copy() for k in reference.files}
    for path in ('stream','commit'):
        current=np.stack([rows[str(uid)][path+'_key'].numpy() for uid in arrays['uid']])
        prior=arrays[path+'_key']
        if current.shape!=prior.shape or current.dtype!=prior.dtype or current.tobytes()!=prior.tobytes():
            raise ValueError('compatible gate keys differ from C-exact diagnostic')
        arrays[path+'_key']=current
    for i,uid in enumerate(arrays['uid']):
        row=rows[str(uid)]
        if (row['label'],row['subject_id'],row['sensor_id'])!=(int(arrays['labels'][i]),str(arrays['subject'][i]),str(arrays['sensor'][i])):
            raise ValueError('diagnostic/current metadata mismatch')
    queries=torch.from_numpy(arrays['queries'])
    masks={'dev':np.ones(len(arrays['uid']),dtype=bool),'dev_regression':arrays['role']=='dev_regression'}
    for s,sensor in sorted(set(zip(arrays['subject'],arrays['sensor']))):
        masks[f'{s}/{sensor}']=(arrays['subject']==s)&(arrays['sensor']==sensor)
    groups={}; passed=True
    for name,mask in masks.items():
        if not mask.any():
            raise ValueError('required retrieval group empty')
        groups[name]={}
        for path in ('stream','commit'):
            labels=torch.from_numpy(arrays['labels'][mask])
            current=fixed_query_retrieval(torch.from_numpy(arrays[path+'_key'][mask]),labels,queries)
            original=fixed_query_retrieval(torch.from_numpy(arrays['teacher_'+path+'_key'][mask]),labels,queries)
            delta={k:None if current[k] is None or original[k] is None else current[k]-original[k]
                   for k in ('macro_map','macro_r_at_1')}
            ok=all(v is not None and v>=-(.01 if name in ('dev','dev_regression') else .03)-1e-10 for v in delta.values())
            passed &= ok
            groups[name][path]=dict(current=current,original=original,delta=delta,passed=ok)
    np.savez_compressed(Path(output)/'retrieval-arrays.npz',**arrays)
    report=dict(passed=bool(passed),metrics=groups,paired_diagnostic_keys_bitwise_equal=True,
                arrays_sha256=file_sha(Path(output)/'retrieval-arrays.npz'),evaluation='fixed-GT-run-development')
    write_json(Path(output)/'retrieval-gate.json',report)
    return report


def main():
    import argparse
    import yaml
    from .cache import load_original
    from .checkpoint import code_digest, save_bundle, read_bundle
    from .controller import ControllerConfig
    from .evidence import EvidenceConfig
    from .gate_diagnostic import transferred_gate,validate_inference_source,write_status
    from .prepare import write_json
    from .stage3 import V25Stage3
    from .train_heads import build_event_features,train_fixed_head
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    config=yaml.safe_load(Path(args.config).read_text())
    status=dict(phase='source-check',actual_optimizer_steps=0,holdout_access=False,official_data_access=False,formal_go=False)
    def update(**kwargs):
        status.update(kwargs);write_status(output/'status.json',status)
        print(json.dumps(status),flush=True)
    try:
        if (config['architecture']!=PositiveCoordinateGate.architecture or config['seed']!=42
            or config['actual_optimizer_steps']!=500 or config['batch_events']!=8
            or config['optimizer']!=dict(lr=.001,weight_decay=.01,clip_norm=1.) or not config['sealed_holdouts']):
            raise ValueError('locked Evidence500 configuration required')
        for name in ('source_checkpoint','stage1_checkpoint','diagnostic_arrays'):
            if file_sha(config[name+'_path'])!=config[name+'_sha256']:
                raise ValueError(name+' source SHA mismatch')
        prep=Path(config['native_preparation'])
        index_path=prep/'encoder/index.json';teacher_path=prep/'teachers/index.json'
        for path,key in ((index_path,'encoder_index_sha256'),(teacher_path,'teachers_index_sha256')):
            if file_sha(path)!=config[key]:
                raise ValueError(key+' mismatch')
        index=json.loads(index_path.read_text());validate_source_roles(index)
        config_sha=validate_inference_source(config,index)
        frontend,tokenizer=load_original(config['frontend_config'],config['source_checkpoint_path'],'cuda')
        teacher_gate_digest=module_digest(frontend.event_former.state_gate)
        expected=transferred_gate(frontend.event_former.state_gate,maturity=1.,exact=True)
        install_positive_coordinate_gate(frontend.event_former)
        frontend.event_former.state_gate.float()
        if module_digest(expected)!=module_digest(frontend.event_former.state_gate):
            raise RuntimeError('formal gate weights differ from exact diagnostic')
        del expected
        frontend.eval().requires_grad_(False)
        frozen=module_digest(frontend)
        source=dict(config_sha256=file_sha(args.config),frontend_config_sha256=config_sha,
            encoder_index_sha256=file_sha(index_path),teachers_index_sha256=file_sha(teacher_path),
            source_checkpoint_sha256=config['source_checkpoint_sha256'],stage1_checkpoint_sha256=config['stage1_checkpoint_sha256'],
            diagnostic_arrays_sha256=config['diagnostic_arrays_sha256'],code_digest=code_digest(),
            architecture=PositiveCoordinateGate.architecture,teacher_gate_digest=teacher_gate_digest,
            frontend_digest=frozen,gate_optimizer_steps=0,stop_gradient=True,
            target_kind='MD-statistic-proxy',holdout_access=False,official_data_access=False)
        write_json(output/'source.json',source)
        update(phase='frozen-event-features')
        events=build_event_features(frontend,index,output/'features')
        feature_path=output/'features/events.pt';feature_sha=file_sha(feature_path)
        validate_event_features(torch.load(feature_path,map_location='cpu',weights_only=False),frozen)
        if module_digest(frontend)!=frozen:
            raise RuntimeError('feature building changed frozen frontend')
        update(phase='paired-retrieval-check')
        with np.load(config['diagnostic_arrays_path'],allow_pickle=False) as ref:
            retrieval=paired_retrieval(events,ref,output)
        if not retrieval['passed']:
            update(phase='stopped',reason='initial-retrieval-gate-failed')
            return
        torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        cfg=frontend.event_former.config
        stage=V25Stage3(radar_dim=cfg.radar_dim,event_dim=cfg.llm_dim,semantic_dim=cfg.semantic_dim).to('cuda')
        stage.set_phase('evidence')
        initial=module_digest(stage.heads.evidence)
        source.update(feature_sha256=feature_sha)
        with (output/'evidence-step0.pt').open('xb') as stream:
            torch.save(dict(stage_state=stage.state_dict(),optimizer_steps=dict(stage.optimizer_steps),source=source),stream)
        zero=evaluate_evidence(stage,events);write_json(output/'evidence-step0.json',zero)
        batches=PairedEvidenceBatches(events,seed=42,device='cuda')
        update(phase='evidence-training',budget=500)
        def report_step(row):
            status['actual_optimizer_steps']=row['actual_optimizer_steps']
            write_status(output/'status.json',status)
        trained=train_fixed_head(stage,batches,phase='evidence',steps=500,output=output/'training',source=source,on_step=report_step)
        update(phase='result-verification',actual_optimizer_steps=500)
        frozen_after=module_digest(frontend)
        if frozen_after!=frozen or file_sha(feature_path)!=feature_sha or module_digest(stage.heads.evidence)==initial:
            raise RuntimeError('frozen source changed or Evidence did not update')
        counts=batches.coverage()
        if len(counts)!=20 or any(row['draws']!=400 for row in counts):
            raise RuntimeError('paired subject/sensor/path balance failed')
        write_json(output/'sampling.json',dict(coverage=counts,event_draws=batches.draws,
            draws_by_event=dict(Counter(batches.draws)),actual_optimizer_steps=500))
        final=evaluate_evidence(stage,events);write_json(output/'evidence-step500.json',final)
        selected=trained['checkpoints'][-1]
        saved=torch.load(selected['path'],map_location='cpu',weights_only=False)
        if (selected['step']!=500 or saved['optimizer_steps']!=dict(physical=0,semantic=0,evidence=500)
            or module_digest(stage)!=state_digest(saved['stage_state'])):
            raise RuntimeError('final head checkpoint readback mismatch')
        bundle=output/'compatible-evidence500.pt'
        bundle_sha=save_bundle(bundle,frontend,stage,ControllerConfig(),EvidenceConfig(),
            provenance={**source,'frontend_config_sha256':config_sha,'data_sha256':index['dataset_provenance_sha256']})
        read_bundle(bundle,expected_sha256=bundle_sha,allow_development=True)
        result=dict(actual_optimizer_steps=500,gate_optimizer_steps=0,selected_checkpoint=selected,
            bundle_sha256=bundle_sha,frontend_before=frozen,frontend_after=frozen_after,
            frozen_unchanged=True,feature_sha256=feature_sha,source=source,
            evidence_passed=final['passed'],evidence_step0={k:v for k,v in zero.items() if k!='predictions'},
            evidence_step500={k:v for k,v in final.items() if k!='predictions'},
            retrieval=retrieval,retrieval_before_after_identical=True,
            holdout_access=False,official_data_access=False,formal_go=False,
            next_action='report-only-no-automatic-next-training')
        write_json(output/'result-readback-checks-v1.json',result)
        update(phase='completed',evidence_passed=final['passed'],retrieval_preserved=True)
    except BaseException as exc:
        partial=output/'training/failure-head.pt'
        if partial.exists():
            payload=torch.load(partial,map_location='cpu',weights_only=False)
            status['actual_optimizer_steps']=payload['optimizer_steps']['evidence']
            status['failure_checkpoint_sha256']=file_sha(partial)
        if 'batches' in locals():
            write_json(output/'failure-sampling.json',dict(coverage=batches.coverage(),event_draws=batches.draws))
        update(phase='failed',error=repr(exc))
        raise


if __name__=='__main__':
    main()
