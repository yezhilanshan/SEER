"""Exclusive fixed-budget pilot. Failure gates stop all later phases/seeds."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch
import yaml

from .cache import load_original,chunk_schedule,build_teacher_cache
from .calibrate import calibrate_and_replay
from .checkpoint import file_sha,module_digest,code_digest,save_bundle,read_bundle
from .collect import collect_candidate_episodes
from .controller import ControllerConfig
from .evaluate import evaluate_continuous,evaluate_original,load_memory_and_qa,compare_boundary,compare_qa
from .evidence import EvidenceConfig
from .prepare import write_json
from .stage3 import V25Stage3
from .train_heads import build_event_features,build_oracle_pairs,PairSampler,EvidenceBatches,PhysicalBatches,train_fixed_head,evidence_gate
from .train_stage2 import train_gate


def successful_updates(output):
    counts={}
    for path in Path(output).rglob('training.jsonl'):
        values=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        counts[str(path.parent)]=max((int(row['actual_optimizer_steps']) for row in values),default=0)
    for path in Path(output).rglob('failure-readback.json'):
        value=json.loads(path.read_text())
        counts[str(path.parent)]=max(counts.get(str(path.parent),0),value['max_adamw_parameter_step'])
    return sum(counts.values())


def terminal_lock(results,*,planned_seeds):
    if {r['seed'] for r in results}!=set(planned_seeds):
        raise ValueError('all planned seeds must have terminal results before holdout unlock')
    if any(r['status']!='dev-qualified' or r['actual_optimizer_steps']!=3750 for r in results):
        raise ValueError('all planned seeds must be qualified fixed-budget terminals')
    for result in results:
        if any(len(result.get(key,''))!=64 for key in ('bundle_sha256','config_sha256')):
            raise ValueError('terminal weight/config SHA required')
    return dict(planned_seeds=list(planned_seeds),results=results,holdout_unlock_allowed=True,formal_go=False)


def _read(path):
    return json.loads(Path(path).read_text())


def _stop(seed,output,stage,reason,source,details):
    result=dict(seed=seed,status='dev-failed',stop_phase=stage,reason=reason,
        actual_optimizer_steps=successful_updates(output),holdout_unlock_allowed=False,source=source,details=details)
    write_json(Path(output)/'result.json',result)
    return result


def run_seed(config,index,teachers,ready,root,seed):
    output=Path(root)/f'seed{seed}'; output.mkdir(parents=True,exist_ok=False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    source={key:config[key] for key in config if key.endswith('_sha256')}
    source.update(code_digest=code_digest(),seed=seed,micro_batch=ready['micro_batch'],accumulation=ready['accumulation'])
    device=torch.device('cuda')
    frontend,tokenizer=load_original(config['frontend_config'],config['source_checkpoint_path'],device)
    if module_digest(frontend)!=teachers['original_frontend_digest'] or module_digest(frontend.radar_encoder)!=index['encoder_digest']:
        raise ValueError('native preparation and original current frontend differ')
    stage=V25Stage3().to(device)
    size=ready['micro_batch']
    if seed==42:
        schedule=_read(Path(config['native_preparation'])/f'schedule-micro{size}.json')
    else:
        schedule=chunk_schedule(index['runs'],seed=seed,micro_batch=size,accumulation=16//size)
        teachers=build_teacher_cache(frontend,index,[schedule],output/'original-teachers')
    write_json(output/'locked-schedule.json',schedule)
    source['schedule_sha256']=file_sha(output/'locked-schedule.json')
    result_A=train_gate(frontend,tokenizer,stage.heads,index,teachers,schedule,output/'A',micro_batch=size,seed=seed,source=source)
    if not result_A['retrieval_gate']['passed']:
        return _stop(seed,output,'A','retrieval-retention',source,result_A)
    events=build_event_features(frontend,index,output/'current-gate-events')
    result_B=train_fixed_head(stage,EvidenceBatches(events,seed=seed,device=device),phase='evidence',steps=500,
                             output=output/'B',source=source)
    physical_qualified=evidence_gate(stage,events,output/'B')
    if not physical_qualified['passed']:
        return _stop(seed,output,'B','Evidence-MAE',source,physical_qualified)
    result_C=train_fixed_head(stage,PhysicalBatches(index,device=device),phase='physical',steps=1000,output=output/'C',source=source)
    oracle=build_oracle_pairs(events)
    torch.save(oracle,output/'oracle-fit-pairs.pt')
    source['oracle_sha256']=file_sha(output/'oracle-fit-pairs.pt')
    oracle_sampler=PairSampler(oracle,seed=seed)
    train_fixed_head(stage,lambda module:module.oracle_step(oracle_sampler.batch(device))['loss'],
        phase='oracle',steps=250,output=output/'D1',source=source)
    write_json(output/'D1'/'sampling-coverage.json',oracle_sampler.coverage())
    collected=collect_candidate_episodes(frontend,stage,index,output/'fit-closed-loop-once',role='fit')
    closed_sampler=PairSampler(collected['pairs'],seed=seed)
    oracle_sampler=PairSampler(oracle,seed=seed+1)
    def semantic_batch(module):
        batch=closed_sampler.batch(device); extra=oracle_sampler.batch(device)
        batch.update({f'oracle_{key}':value for key,value in extra.items()})
        batch['oracle_weight']=.3
        return module.semantic_step(batch)['loss']
    train_fixed_head(stage,semantic_batch,phase='semantic',steps=1000,output=output/'D2',source=source)
    write_json(output/'D2'/'sampling-coverage.json',dict(closed=closed_sampler.coverage(),oracle=oracle_sampler.coverage()))
    if successful_updates(output)!=3750:
        raise RuntimeError('independent total successful AdamW count differs from3750')
    calibrated=calibrate_and_replay(frontend,stage,index,output/'E')
    if not calibrated['passed']:
        return _stop(seed,output,'E',calibrated['stop'],source,calibrated)
    controller=ControllerConfig(**calibrated['controller_config']); evidence=EvidenceConfig(**calibrated['evidence_config'])
    bundle=output/'fixed-terminal.pt'
    bundle_sha=save_bundle(bundle,frontend,stage,controller,evidence,
        provenance=dict(frontend_config_sha256=file_sha(config['frontend_config']),data_sha256=config['dataset_provenance_sha256'],
                        training_sources=source),frontend_eventformer_steps=1000,initialization_probe=calibrated['probe'],calibration=calibrated['calibration'])
    read_bundle(bundle,expected_sha256=bundle_sha)
    memory,qa=load_memory_and_qa(config,frontend,tokenizer,device)
    current=evaluate_continuous(frontend,stage,index,memory,qa,tokenizer,output/'F-current',
                               controller_config=controller,evidence_config=evidence)
    original=evaluate_original(config,index,output/'F-original')
    boundary=compare_boundary(current['boundary'],original['boundary'])
    qa_gate=compare_qa(current['qa'],original['qa'])
    # Native pace remains a complete recording, with no questions/generation
    # blocking the clock. First eligible dev recording must cover at least600s.
    long=[]
    for record in index['records']:
        if record['role']=='dev':
            payload=torch.load(record['path'],map_location='cpu',weights_only=False)
            if len(payload['radar'])*.18>=600:
                long.append(record)
    if not long:
        return _stop(seed,output,'F','no-ten-minute-dev-recording',source,{})
    pace_index={**index,'records':[long[0]]}
    paced=evaluate_continuous(frontend,stage,pace_index,memory,qa,tokenizer,output/'F-paced',
                             controller_config=controller,evidence_config=evidence,paced=True)
    gates=dict(boundary=boundary,qa=qa_gate,engineering=current['engineering_passed'],realtime=paced['realtime_passed'])
    if not boundary['passed'] or not qa_gate['passed'] or not current['engineering_passed'] or not paced['realtime_passed']:
        return _stop(seed,output,'F','complete-pipeline-development-gates',source,gates)
    result=dict(seed=seed,status='dev-qualified',actual_optimizer_steps=3750,bundle_path=str(bundle),bundle_sha256=bundle_sha,
                config_sha256=config['_sha256'],source=source,gates=gates,holdout_unlock_allowed=False,formal_go=False)
    write_json(output/'result.json',result)
    del frontend,stage,memory,qa,oracle,events
    gc.collect(); torch.cuda.empty_cache()
    return result


def run_training_plan(config_path,output):
    config=yaml.safe_load(Path(config_path).read_text())
    config['_sha256']=file_sha(config_path)
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    # An existing output is always an intentional stop; stale locks never trigger
    # another worker. The exclusive lock is retained with the original evidence.
    lock=output/'worker.lock'
    with lock.open('x') as stream:
        json.dump(dict(pid=os.getpid(),started_unix=time.time(),config_sha256=config['_sha256']),stream)
    write_json(output/'config-lock.json',config)
    ready=_read(Path(config['native_preparation'])/'ready.json')
    if ready['phase']!='native-cache-and-preflight-ready' or ready['actual_optimizer_steps']!=0:
        raise ValueError('completed native0-step preflight certificate required')
    index_path=Path(config['native_preparation'])/'encoder/index.json'
    teacher_path=Path(config['native_preparation'])/'teachers/index.json'
    if file_sha(index_path)!=ready['encoder_index_sha256'] or file_sha(teacher_path)!=ready['teachers_index_sha256']:
        raise ValueError('native cache index SHA changed')
    for name in ('source_checkpoint','stage1_checkpoint','dataset_provenance','qa','compressor','legacy_stage3','legacy_threshold'):
        if file_sha(config[name+'_path'])!=config[name+'_sha256']:
            raise ValueError(f'locked source changed: {name}')
    index,teachers=_read(index_path),_read(teacher_path)
    if index['dataset_provenance_sha256']!=config['dataset_provenance_sha256']:
        raise ValueError('native encoder cache dataset source changed')
    if index['config_sha256']!=file_sha(config['frontend_config']):
        raise ValueError('native cache frontend config changed')
    results=[]
    try:
        primary=run_seed(config,index,teachers,ready,output,42); results.append(primary)
        if primary['status']!='dev-qualified':
            write_json(output/'terminal-results.json',dict(results=results,holdout_unlock_allowed=False,formal_go=False))
            return results
        repeat=run_seed(config,index,teachers,ready,output,43); results.append(repeat)
        if repeat['status']!='dev-qualified':
            write_json(output/'terminal-results.json',dict(results=results,holdout_unlock_allowed=False,formal_go=False))
            return results
        locked=terminal_lock(results,planned_seeds=(42,43))
        write_json(output/'all-terminal-lock.json',locked)
        # Holdout conversion/evaluation follows the lock only, through a distinct
        # source manifest. It never changes fit normalization or any checkpoint.
        evaluate_locked_holdouts(config,index,results,output)
        write_json(output/'terminal-results.json',dict(results=results,holdout_unlock_allowed=True,formal_go=False))
        return results
    except BaseException:
        (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
        write_json(output/'failure-status.json',dict(pid=os.getpid(),actual_optimizer_steps=successful_updates(output),
            status='failed',holdout_unlock_allowed=False,formal_go=False))
        raise


def evaluate_locked_holdouts(config,index,results,output):
    # Definition below is intentionally entered only after a verified all-seed
    # terminal lock. An absent/failed terminal cannot call this independently.
    lock=_read(Path(output)/'all-terminal-lock.json')
    if not lock.get('holdout_unlock_allowed') or lock['results']!=results:
        raise ValueError('verified all-seed terminal lock required')
    from .holdout import evaluate_locked
    return evaluate_locked(config,index,results,output)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    try:
        run_training_plan(args.config,args.output)
    except BaseException:
        output=Path(args.output)
        lock=output/'worker.lock'
        # Preserve preflight exceptions as well; never modify another worker's
        # or an existing output's failure evidence.
        if lock.is_file() and _read(lock)['pid']==os.getpid():
            if not (output/'failure.txt').exists():
                (output/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8')
            if not (output/'failure-status.json').exists():
                write_json(output/'failure-status.json',dict(pid=os.getpid(),status='failed',
                    actual_optimizer_steps=successful_updates(output),holdout_unlock_allowed=False,formal_go=False))
        raise


if __name__=='__main__':
    main()
