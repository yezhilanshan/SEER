"""One locked evaluation, with no refit or checkpoint selection."""
from __future__ import annotations

import gc
import json
from pathlib import Path
import shutil

import torch

from .cache import build_encoder_cache,build_teacher_cache,load_original
from .checkpoint import file_sha,load_bundle
from .controller import ControllerConfig
from .data import require_array_access
from .evaluate import evaluate_original,evaluate_continuous,load_memory_and_qa,compare_boundary,compare_qa
from .evidence import EvidenceConfig
from .models import install_age_free_gate
from .prepare import write_json
from .stage3 import V25Stage3
from .train_stage2 import retrieval_gate


def validate_unlock(lock):
    results=lock.get('results',[])
    if not lock.get('holdout_unlock_allowed') or {r['seed'] for r in results}!={42,43} or len(results)!=2:
        raise ValueError('both planned seed terminals must be locked')
    for result in results:
        if result['status']!='dev-qualified' or result['actual_optimizer_steps']!=3750:
            raise ValueError('both planned seeds must pass development gates')
        path=Path(result['bundle_path'])
        if not path.is_file() or file_sha(path)!=result['bundle_sha256']:
            raise ValueError('locked weight SHA changed')
    return results


def evaluate_locked(config,index,results,output):
    from ..parrad_convert import convert_recording,read_command_annotations
    output=Path(output)/'locked-holdout'; output.mkdir(exist_ok=False)
    lock=json.loads((output.parent/'all-terminal-lock.json').read_text())
    if validate_unlock(lock)!=results:
        raise ValueError('terminal results differ from lock')
    source=Path(config['dataset_dir'])
    provenance=json.loads((source/'dataset-provenance.json').read_text())
    if file_sha(source/'dataset-provenance.json')!=config['dataset_provenance_sha256']:
        raise ValueError('sealed data provenance changed')
    dataset=output/'canonical'; dataset.mkdir()
    for name in ('normalization.json','label_map.json'):
        if file_sha(source/name)!=provenance[name.removesuffix('.json')+'_sha256']:
            raise ValueError('fit normalization/vocabulary changed')
        shutil.copyfile(source/name,dataset/name)
    known={row['command'] for row in json.loads((dataset/'label_map.json').read_text())['classes']}
    sources={item['path']:item['sha256'] for row in provenance['sources'] for item in row['files']}
    converted={}; roles=('session_holdout','subject_holdout')
    for role in roles:
        if file_sha(source/f'{role}.jsonl')!=provenance['manifests'][role]:
            raise ValueError('sealed manifest changed')
        rows=[json.loads(line) for line in (source/f'{role}.jsonl').read_text().splitlines() if line.strip()]
        actual=[]
        for row in rows:
            require_array_access(row,allowed_roles=roles,terminal_locked=True)
            if 'path' in row:
                actual.append(row); continue
            key=row['subject_id'],row['session_id']
            if key not in converted:
                for name in ('source_h5','source_commands'):
                    if file_sha(row[name])!=sources[row[name]]:
                        raise ValueError('sealed raw source SHA changed')
                unknown={r.command for r in read_command_annotations(row['source_commands'])}-known
                if unknown:
                    raise ValueError(f'held-out unknown commands: {sorted(unknown)}')
                destination=dataset/'converted'/key[0]/key[1]; destination.mkdir(parents=True)
                shutil.copyfile(dataset/'label_map.json',destination/'label_map.json')
                manifest=convert_recording(row['source_h5'],row['source_commands'],destination,
                    subject_id=key[0],session_id=key[1],environment_id='source-unknown',frame_period_s=.09,md_only=True)
                converted[key]=[]
                for line in manifest.read_text().splitlines():
                    item=json.loads(line); item.update(role=role,array_access_locked=False)
                    item['path']=str((destination/item['path']).resolve())
                    converted[key].append(item)
            actual.extend(r for r in converted[key] if r['sensor_id']==row['sensor_id'])
        if len(actual)!=8:
            raise ValueError('held-out dual-sensor view count must be8 per role')
        with (dataset/f'{role}.jsonl').open('x') as stream:
            for row in actual: stream.write(json.dumps(row)+'\n')
    write_json(dataset/'dataset-provenance.json',dict(terminal_lock_sha256=file_sha(output.parent/'all-terminal-lock.json'),
        parent_provenance_sha256=config['dataset_provenance_sha256'],fit_normalization_refitted=False,
        official_validation_test_access=False,subject12_unseen_provenance_verified=False,formal_go=False))
    device=torch.device('cuda')
    front,tokenizer=load_original(config['frontend_config'],config['source_checkpoint_path'],device)
    held=build_encoder_cache(front,dataset,output/'encoder',checkpoint_sha256=config['source_checkpoint_sha256'],
        config_sha256=file_sha(config['frontend_config']),roles=roles,terminal_locked=True)
    # GT-run retrieval only: alias evaluation index role, not the sealed source
    # metadata. No fit sampler, normalization or optimization is invoked here.
    metric_index={**held,'runs':[{**r,'role':'dev'} for r in held['runs']],
        'records':[{**r,'role':'dev'} for r in held['records']]}
    teachers=build_teacher_cache(front,metric_index,[],output/'original-teachers')
    del front; gc.collect(); torch.cuda.empty_cache()
    baselines={role:evaluate_original(config,held,output/f'original-{role}',role=role,terminal_locked=True) for role in roles}
    reports=[]
    for result in results:
        front,tokenizer=load_original(config['frontend_config'],config['source_checkpoint_path'],device)
        install_age_free_gate(front.event_former); stage=V25Stage3().to(device)
        payload=load_bundle(result['bundle_path'],front,stage,expected_sha256=result['bundle_sha256'])
        front.eval().requires_grad_(False); stage.eval().requires_grad_(False)
        memory,qa=load_memory_and_qa(config,front,tokenizer,device)
        folder=output/f"seed{result['seed']}"; folder.mkdir()
        retrieval=retrieval_gate(front,tokenizer,metric_index,teachers,folder)
        rows={}
        for role in roles:
            current=evaluate_continuous(front,stage,held,memory,qa,tokenizer,folder/role,role=role,terminal_locked=True,
                controller_config=ControllerConfig(**payload['controller_config']),evidence_config=EvidenceConfig(**payload['evidence_config']))
            rows[role]=dict(boundary=compare_boundary(current['boundary'],baselines[role]['boundary']),
                qa=compare_qa(current['qa'],baselines[role]['qa']),engineering_passed=current['engineering_passed'])
        reports.append(dict(seed=result['seed'],fixed_terminal_sha256=result['bundle_sha256'],roles=rows,retrieval=retrieval))
        del front,stage,memory,qa,payload; gc.collect(); torch.cuda.empty_cache()
    write_json(output/'report.json',dict(results=reports,formal_go=False,
        subject12_unseen_provenance_verified=False,all_planned_terminal_locked=True))
    return reports
