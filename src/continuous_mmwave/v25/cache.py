"""Frozen encoder cache and offline original-path teachers; no old event input."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random

import numpy as np
import torch

from .checkpoint import file_sha, module_digest
from .data import SENSORS, require_array_access, build_evidence_targets, update_evidence_targets, clean_runs


class BalancedRunSampler:
    """Exact round-robin subject/sensor mass; rotate available classes per cell."""
    def __init__(self, rows, *, seed):
        self.rng = random.Random(seed)
        self.cells = defaultdict(lambda: defaultdict(list))
        for row in rows:
            self.cells[(row['subject_id'], row['sensor_id'])][row['label']].append(row)
        self.subjects = sorted({cell[0] for cell in self.cells})
        if not self.subjects or any((s, sensor) not in self.cells for s in self.subjects for sensor in SENSORS):
            raise ValueError('both sensors required per fit subject')
        self.classes = {cell: sorted(by_label) for cell, by_label in self.cells.items()}
        self.turns = defaultdict(int)
        self.draw_index = 0

    def next_group(self, size):
        if size < 2:
            raise ValueError('distinct actions need micro-batch >=2')
        if len({label for labels in self.classes.values() for label in labels}) < 2:
            raise ValueError('distinct actions unavailable')
        group = []
        for _ in range(size):
            index = self.draw_index
            subject = self.subjects[index % len(self.subjects)]
            sensor = SENSORS[(index // len(self.subjects)) % 2]
            cell = subject, sensor
            classes = self.classes[cell]
            label = classes[self.turns[cell] % len(classes)]
            # Prefer an available different label inside this cell without changing
            # the subject/sensor mass. Fail explicitly if the batch is impossible.
            if group and len({r['label'] for r in group}) == 1 and label == group[0]['label']:
                label = next((k for k in classes if k != label), label)
            group.append(self.rng.choice(self.cells[cell][label]))
            self.turns[cell] += 1
            self.draw_index += 1
        if len({r['label'] for r in group}) < 2:
            raise ValueError('distinct actions unavailable in balanced micro-batch')
        return group


def load_original(config_path, checkpoint_path, device):
    from ..train import _build_parrad_stage2, load_experiment_config
    from ..checkpoint import load_training_checkpoint
    frontend, tokenizer = _build_parrad_stage2(load_experiment_config(config_path))
    payload = load_training_checkpoint(checkpoint_path, frontend)
    validate_original_source(frontend,payload)
    frontend.requires_grad_(False)
    frontend.to(device).eval()
    return frontend, tokenizer


def validate_original_source(frontend,payload):
    if payload.get('trainable_only'):
        # _build_parrad_stage2 loads Stage1 strictly before this Stage2 import.
        # The historical trainable-only Stage2 export intentionally omits the
        # frozen radar encoder, but must still contain all fitted former/P keys.
        expected={name for name,parameter in frontend.named_parameters()
                  if parameter.requires_grad and not name.startswith('radar_encoder.')}
        missing=sorted(expected-set(payload['model_state']))
        if missing:
            raise ValueError(f'original checkpoint misses fitted source parameters: {missing}')


@torch.no_grad()
def build_encoder_cache(frontend, dataset_dir, output, *, checkpoint_sha256, config_sha256,
                        roles=('fit','dev'),terminal_locked=False):
    dataset_dir, output = Path(dataset_dir), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    normalization = json.loads((dataset_dir/'normalization.json').read_text())
    device = next(frontend.parameters()).device
    encoder = frontend.radar_encoder
    records, runs = [], []
    if any(role not in ('fit','dev','session_holdout','subject_holdout') for role in roles):
        raise ValueError('fixed dataset roles required')
    for role in roles:
        rows = [json.loads(line) for line in (dataset_dir/f'{role}.jsonl').read_text().splitlines() if line.strip()]
        for row in rows:
            require_array_access(row, allowed_roles=('fit','dev_regression','dev_new','session_holdout','subject_holdout'),
                                 terminal_locked=terminal_locked)
            with np.load(row['path'], allow_pickle=False) as arrays:
                md = np.asarray(arrays['md'], dtype=np.float32)
                labels = np.asarray(arrays['labels'], dtype=np.int64)
                timestamps = np.asarray(arrays['timestamps'], dtype=np.float64)
            if len(md) < 2 or not np.isfinite(md).all() or not np.isfinite(timestamps).all() or (np.diff(timestamps) <= 0).any():
                raise ValueError('malformed canonical recording')
            frames = len(md) // 2 * 2
            proxy, proxy_mask = build_evidence_targets(md[:frames], normalization=normalization[row['sensor_id']])
            target, target_mask = update_evidence_targets(proxy, proxy_mask)
            state, tokens, masks = None, [], []
            for i in range(0, frames, 2):
                rd = torch.empty((1,2,0,0,0), device=device)
                update = encoder.forward_update(rd, torch.from_numpy(md[i:i+2].copy())[None].to(device),
                    sensor_index=torch.tensor([SENSORS.index(row['sensor_id'])],device=device), state=state,
                    frame_mask=torch.ones((1,2),dtype=torch.bool,device=device))
                state = update.state
                tokens.append(update.radar_tokens[0].float().cpu())
                masks.append(update.radar_token_mask[0].cpu())
            uid = f"{row['subject_id']}-{row['session_id']}-{row['sensor_id']}"
            payload = dict(uid=uid, role=role, metadata=row, radar=torch.stack(tokens), radar_mask=torch.stack(masks),
                labels=torch.from_numpy(labels[:frames].reshape(-1,2).copy()),
                timestamps=torch.from_numpy(timestamps[:frames].reshape(-1,2).copy()),
                frame_target=torch.from_numpy(proxy), frame_target_mask=torch.from_numpy(proxy_mask),
                target=torch.from_numpy(target), target_mask=torch.from_numpy(target_mask),
                source_npz_sha256=file_sha(row['path']), trailing_frames=len(md)-frames)
            target_path = output/f'{uid}.pt'
            torch.save(payload, target_path)
            records.append(dict(uid=uid, role=role, path=str(target_path), sha256=file_sha(target_path),
                                metadata=row, trailing_frames=len(md)-frames))
            for number,(start,end,label) in enumerate(clean_runs(payload['labels'].numpy())):
                runs.append(dict(uid=f'{uid}:{number}', record_uid=uid, role=role, start=start,end=end,label=label,
                    subject_id=row['subject_id'],session_id=row['session_id'],sensor_id=row['sensor_id']))
            print(json.dumps(dict(phase='encoder-cache',recording=uid,role=role,updates=len(tokens))),flush=True)
    index = dict(schema_version=25, kind='frozen-encoder-only', checkpoint_sha256=checkpoint_sha256,
        config_sha256=config_sha256, encoder_digest=module_digest(encoder),
        dataset_provenance_sha256=file_sha(dataset_dir/'dataset-provenance.json'), records=records,runs=runs)
    from .prepare import write_json
    write_json(output/'index.json', index)
    return index


def chunk_schedule(runs, *, seed, micro_batch, optimizer_steps=1000, accumulation=4):
    """Immutable, reproducible consecutive chunks; max64, never random windows."""
    sampler = BalancedRunSampler([r for r in runs if r['role']=='fit'], seed=seed)
    cursors = {}
    rows = []
    for step in range(optimizer_steps):
        for micro in range(accumulation):
            group = sampler.next_group(micro_batch)
            remaining = [r['end']-r['start']-cursors.get(r['uid'],0) for r in group]
            size = min(64,min(remaining))
            if size <= 0:
                raise ValueError('invalid chunk cursor')
            chunks=[]
            for run in group:
                start=cursors.get(run['uid'],0)
                end=start+size
                chunks.append(dict(uid=run['uid'],start=start,end=end,reset=start==0))
                cursors[run['uid']] = 0 if end==run['end']-run['start'] else end
            rows.append(dict(step=step+1,micro=micro+1,chunks=chunks))
    return rows


@torch.no_grad()
def build_teacher_cache(frontend, index, schedules, output):
    """Independent same-input original Stream/Commit keys at scheduled endpoints.

    Old Stream uses original online-controller age ramp (.18s/1.2s), recorded
    solely in this frozen teacher. New gate never receives it as a feature.
    Development includes every fixed GT run's complete endpoint.
    """
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    requested=defaultdict(set)
    for schedule in schedules:
        for group in schedule:
            for chunk in group['chunks']:
                requested[chunk['uid']].add(chunk['end'])
    by_record=defaultdict(list)
    for run in index['runs']:
        by_record[run['record_uid']].append(run)
        if run['role']=='dev':
            requested[run['uid']].add(run['end']-run['start'])
    device=next(frontend.parameters()).device
    former,p=frontend.event_former,frontend.semantic_projection
    prompt=former.build_prompt_cache(1)
    entries=[]
    for record in index['records']:
        if file_sha(record['path'])!=record['sha256']:
            raise ValueError('encoder cache SHA changed')
        payload=torch.load(record['path'],map_location='cpu',weights_only=False)
        keys={}
        for run in by_record[record['uid']]:
            endpoints=requested[run['uid']]
            if not endpoints:
                continue
            state=former.initial_state(1)
            radar=payload['radar'][run['start']:run['end']].to(device)
            mask=payload['radar_mask'][run['start']:run['end']].to(device)
            for i in range(max(endpoints)):
                state=former.stream_update(state,radar[i:i+1],min((i+1)*.18/1.2,1.),mask[i:i+1],prompt).event_tokens
                if i+1 in endpoints:
                    commit=former.commit_refine(radar[:i+1].flatten(0,1)[None],mask[:i+1].flatten()[None],prompt).event_tokens
                    keys[f"{run['uid']}:{i+1}"]=dict(stream_key=p(state)[0].float().cpu(),
                        commit_key=p(commit)[0].float().cpu(),visible_start=run['start'],visible_end=run['start']+i+1)
        path=output/f"{record['uid']}.pt"
        torch.save(keys,path)
        entries.append(dict(uid=record['uid'],path=str(path),sha256=file_sha(path),endpoints=len(keys)))
        print(json.dumps(dict(phase='original-teachers',recording=record['uid'],endpoints=len(keys))),flush=True)
    from .prepare import write_json
    manifest=dict(schema_version=25,kind='independent-original-stream-commit',records=entries,
        original_frontend_digest=module_digest(frontend),encoder_index_sha256=file_sha(Path(index['records'][0]['path']).parent/'index.json'),
        original_stream_maturity_rule='min(age_updates*.18/1.2,1)',
        scheduled_endpoint_counts={k:len(v) for k,v in requested.items()})
    write_json(output/'index.json',manifest)
    return manifest


def main():
    """Native cache/preflight only. Formal training is a distinct later command."""
    import argparse
    import yaml
    from .stage3 import V25Stage3
    from .train_stage2 import preflight_schedule,native_gate_preflight
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text())
    for name in ('source_checkpoint','stage1_checkpoint','dataset_provenance'):
        path=config[name+'_path']
        if file_sha(path)!=config[name+'_sha256']:
            raise ValueError(f'locked source SHA mismatch: {name}')
    output=Path(config['preparation_output'])
    output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(42)
    frontend,tokenizer=load_original(config['frontend_config'],config['source_checkpoint_path'],torch.device('cuda'))
    index=build_encoder_cache(frontend,config['dataset_dir'],output/'encoder',
        checkpoint_sha256=config['source_checkpoint_sha256'],config_sha256=file_sha(config['frontend_config']))
    schedules={4:chunk_schedule(index['runs'],seed=42,micro_batch=4,accumulation=4),
               2:chunk_schedule(index['runs'],seed=42,micro_batch=2,accumulation=8)}
    teachers=build_teacher_cache(frontend,index,list(schedules.values())+[preflight_schedule(index,4),preflight_schedule(index,2)],
                               output/'teachers')
    for size,schedule in schedules.items():
        from .prepare import write_json
        write_json(output/f'schedule-micro{size}.json',schedule)
    stage=V25Stage3().cuda()
    flight=native_gate_preflight(frontend,tokenizer,stage.heads,index,teachers,output,seed=42)
    write_json(output/'ready.json',dict(phase='native-cache-and-preflight-ready',actual_optimizer_steps=0,
        encoder_index_sha256=file_sha(output/'encoder/index.json'),teachers_index_sha256=file_sha(output/'teachers/index.json'),
        micro_batch=flight['micro_batch'],accumulation=flight['accumulation'],config_sha256=file_sha(args.config)))


if __name__=='__main__':
    main()
