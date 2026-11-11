"""Stage4 engineering pilot: annotated-event training and real online replay.

No validation/test arrays, upstream fitting, threshold calibration or checkpoint
selection occur here. Outputs explicitly distinguish oracle training segments
from unchanged-controller online events.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F
import yaml

from continuous_mmwave.models.memory import QueryCompressor
from continuous_mmwave.parrad_memory import EventMemory, MemoryUnit
from continuous_mmwave.stages.parrad_stage4 import summary_objective, encode_query, generate_from_memory

SUMMARY_PROMPT = 'Summarize the events in chronological order:\n'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def validate_provenance(expected, current):
    required={'frontend_config','stage1_checkpoint','stage2_checkpoint','stage3_checkpoint','threshold_artifact'}
    if not required.issubset(expected) or expected != current:
        raise ValueError('cached frontend provenance differs from current frozen sources')


def check_pilot_row(row):
    if (row.get('split') != 'train' or row.get('subject_id') not in ('subject_02','subject_03','subject_04')
            or row.get('session_id') not in ('set000','set001','set002')
            or row.get('sensor_id') not in ('TI_xWR14xx','TI_xWR68xx')):
        raise ValueError('recording outside TRAIN pilot cohort')


def label_runs(labels):
    labels = torch.as_tensor(labels)
    result = []; start = 0
    while start < len(labels):
        label = int(labels[start, 0])
        if label < 0 or not labels[start].eq(label).all():
            start += 1; continue
        stop = start + 1
        while stop < len(labels) and labels[stop].eq(label).all(): stop += 1
        result.append((start, stop, label)); start = stop
    return result


def chronological_groups(events, group_size=4):
    if not events: return []
    if len({e['recording'] for e in events}) != 1 or any(a['start'] >= b['start'] for a,b in zip(events, events[1:])):
        raise ValueError('groups require one chronological recording')
    return [events[i:i+group_size] for i in range(0, len(events), group_size)]


def training_summary(events):
    from continuous_mmwave.stages.parrad_stage2 import default_parrad_event_templates
    templates = default_parrad_event_templates()
    return '; then '.join(templates[e['label']][0] for e in events) + '.'


def caption_tensors(tokenizer, prompt, summary, device):
    prefix = tokenizer.encode(prompt, add_special_tokens=False)
    target = tokenizer.encode(summary, add_special_tokens=False) + [tokenizer.eos_token_id]
    if not prefix or len(prefix) + len(target) > 512:
        raise ValueError('empty instruction or excessively long pilot summary')
    ids = torch.tensor([prefix + target], dtype=torch.long, device=device)
    targets = torch.zeros_like(ids, dtype=torch.bool); targets[:, len(prefix):] = True
    return ids, torch.ones_like(ids, dtype=torch.bool), targets


def finalized_tokens(event_former, radar, mask, prompt):
    return event_former.commit_refine(radar, mask, prompt).event_tokens[0]


def boundary_diagnostic(predicted, truth):
    from continuous_mmwave.parrad_stage3_diagnostics import match_boundaries_seconds
    matched = match_boundaries_seconds(predicted, truth, tolerance_s=.36)
    return dict(asdict(matched), precision=matched.precision, recall=matched.recall, f1=matched.f1)


def upstream_digest(runtime):
    digest = hashlib.sha256()
    modules = (runtime.radar_encoder, runtime.controller.event_former, runtime.controller.semantic_projection,
               runtime.controller.local_change, runtime.controller.radar_transition,
               runtime.controller.semantic_difference, runtime.controller.verify)
    for index, module in enumerate(modules):
        for name, tensor in sorted(module.state_dict().items()):
            digest.update(f'{index}:{name}:{tuple(tensor.shape)}:{tensor.dtype}'.encode())
            digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_frontend(settings, device):
    from transformers import AutoTokenizer
    from continuous_mmwave.parrad_stage3_calibrate import build_runtime, _thresholds_from_artifact
    from continuous_mmwave.parrad_stage3_diagnostics import _apply_thresholds
    runtime = build_runtime(Path(settings['frontend_config']), Path(settings['stage3_checkpoint']), device)
    _apply_thresholds(runtime, _thresholds_from_artifact(Path(settings['threshold_artifact'])))
    frontend_raw = yaml.safe_load(Path(settings['frontend_config']).read_text(encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(frontend_raw['llm']['name_or_path'])
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    return runtime, tokenizer, frontend_raw


@torch.no_grad()
def build_event_cache(settings, runtime, frontend_raw, output, provenance):
    from continuous_mmwave.parrad_stage3_runner import _load_cache_items
    cache = Path(frontend_raw['stage3']['cache_dir'])
    index = json.loads((cache/'index.json').read_text())
    keys = []; sidecar_hashes = {}
    for row in index['records']:
        if row['split'] != 'train': continue
        sidecar = cache/'train'/f"{row['key']}.json"
        metadata = json.loads(sidecar.read_text())['metadata']
        if metadata.get('subject_id') not in settings['subjects'] or metadata.get('sensor_id') not in settings['sensors']:
            continue
        if metadata.get('session_id') not in ('set000','set001','set002'): continue
        check_pilot_row(dict(metadata, split='train'))
        keys.append(row['key']); sidecar_hashes[row['key']] = sha(sidecar)
    items, fingerprint = _load_cache_items(cache, Path(frontend_raw['stage3']['stage2_checkpoint']),
                                         splits=('train',), recording_keys=tuple(keys))
    if not keys: raise ValueError('no approved pilot recordings')
    controller = runtime.controller; device = next(controller.event_former.parameters()).device
    prompt = controller.event_former.build_prompt_cache(1)
    events = []; counts = {}
    for item in items['train']:
        cohort = 'dev' if item.metadata['session_id'] == 'set002' else 'fit'
        counts[item.key] = 0
        for number, (start, stop, label) in enumerate(label_runs(item.record.frame_labels)):
            radar = torch.from_numpy(item.record.radar_tokens[start:stop].copy()).flatten(0,1)[None].to(device)
            mask = torch.from_numpy(item.record.radar_mask[start:stop].copy()).flatten()[None].to(device)
            tokens = finalized_tokens(controller.event_former, radar, mask, prompt)
            key = controller.semantic_projection(tokens[None])[0]
            events.append(dict(uid=f'{item.key}:{number}', recording=item.key, cohort=cohort, label=label,
                               start=float(item.record.timestamps[start,0]),
                               end=float(item.record.timestamps[stop-1,-1]) + .09,
                               tokens=tokens.detach().float().cpu(), key=key.detach().float().cpu()))
            counts[item.key] += 1
        print(json.dumps(dict(phase='cache', recording=item.key, events=counts[item.key])), flush=True)
    payload = dict(schema_version=1, events=events, segment_source='GT clean label-run offline supervision',
                   source_split='train', source_fingerprint=fingerprint, source_index_sha256=sha(cache/'index.json'),
                   sidecar_sha256=sidecar_hashes, counts=counts, provenance=provenance,
                   upstream_digest=upstream_digest(runtime))
    torch.save(payload, output/'events.pt')
    atomic_json(output/'cache-manifest.json', {k:v for k,v in payload.items() if k != 'events'} | {'events_sha256':sha(output/'events.pt')})
    return payload


def examples_for(events, cohort, group_size=4):
    examples = []
    for recording in sorted({e['recording'] for e in events if e['cohort'] == cohort}):
        rows = [e for e in events if e['recording'] == recording and e['cohort'] == cohort]
        groups = chronological_groups(rows, group_size)
        examples.extend((1, group) for group in groups if len(group) >= 2)
        for i in range(0, len(groups), group_size):
            group = groups[i:i+group_size]
            if len(group) >= 2:
                examples.append((2, [e for child in group for e in child]))
    return examples


def compress_example(model, example, device):
    level, events = example
    source = torch.stack([e['tokens'].to(device) for e in events])[None]
    times = torch.tensor([[e['start'] for e in events]], device=device)
    if level == 1:
        result = model(source, torch.ones(source.shape[:3], dtype=torch.bool, device=device), times, level=1)
    else:
        groups = []; group_times = []
        for start in range(0, len(events), 4):
            child = source[:, start:start+4]
            groups.append(model(child, torch.ones(child.shape[:3], dtype=torch.bool, device=device),
                                times[:, start:start+4], level=1))
            group_times.append(times[:, start])
        higher = torch.stack(groups, dim=1)
        result = model(higher, torch.ones(higher.shape[:3], dtype=torch.bool, device=device),
                       torch.stack(group_times, dim=1), level=2)
    return result, source.flatten(1,2)


def objective(model, example, adapter, projection, tokenizer, device, retain_weight):
    summary, source = compress_example(model, example, device)
    ids, attention, target = caption_tensors(tokenizer, SUMMARY_PROMPT, training_summary(example[1]), device)
    return summary_objective(adapter, projection, summary, source, ids, attention, target, retain_weight)


@torch.no_grad()
def evaluate(model, examples, adapter, projection, tokenizer, device, retain_weight):
    model.eval(); rows = []
    for example in examples:
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            summary, source = compress_example(model, example, device)
            ids, attention, target = caption_tensors(tokenizer, SUMMARY_PROMPT, training_summary(example[1]), device)
            metrics = summary_objective(adapter, projection, summary, source, ids, attention, target, retain_weight)
            raw = summary_objective(adapter, projection, source, source, ids, attention, target, retain_weight)
            empty = source[:, :0]
            logits = adapter.caption_logits(empty.to(adapter.compute_dtype), ids, attention)
            valid = target[:,1:]
            no_memory = F.cross_entropy(logits[:,:-1][valid].float(), ids[:,1:][valid])
        rows.append(dict(level=example[0], recording=example[1][0]['recording'], source_ids=[e['uid'] for e in example[1]],
                         **{k:float(v) for k,v in metrics.items()}, raw_event_ce=float(raw['summary_loss']), no_memory_ce=float(no_memory)))
    def means(level):
        selected = [r for r in rows if r['level']==level]
        return None if not selected else {k: sum(r[k] for r in selected)/len(selected)
                                          for k in ('loss','summary_loss','retain_loss','raw_event_ce','no_memory_ce')}
    return dict(rows=rows, level1=means(1), level2=means(2))


def train_compressor(settings, runtime, tokenizer, payload, output):
    device = next(runtime.controller.event_former.parameters()).device
    adapter, projection = runtime.controller.event_former.adapter, runtime.controller.semantic_projection
    model = QueryCompressor(adapter.hidden_size, **settings['compressor']).to(device)
    fit = examples_for(payload['events'], 'fit'); dev = examples_for(payload['events'], 'dev')
    if not fit or not dev or {e[0] for e in fit} != {1,2}:
        raise ValueError('pilot requires fit/dev groups and both training levels')
    teacher_before = upstream_digest(runtime)
    if teacher_before != payload['upstream_digest']: raise ValueError('cached upstream tensor digest changed')
    before = evaluate(model, dev, adapter, projection, tokenizer, device, settings['retain_weight'])
    atomic_json(output/'dev-before.json', before)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['learning_rate'], weight_decay=.01)
    scaler = torch.amp.GradScaler('cuda', init_scale=float(settings.get('amp_initial_scale',256.)),
                                 enabled=device.type=='cuda')
    records = []; order = list(range(len(fit))); rng = random.Random(settings['seed'])
    for step in range(1, settings['steps']+1):
        if (step-1)%len(order) == 0: rng.shuffle(order)
        example = fit[order[(step-1)%len(order)]]
        model.train(); optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type=='cuda'):
            metrics = objective(model, example, adapter, projection, tokenizer, device, settings['retain_weight'])
        if not torch.isfinite(metrics['loss']): raise RuntimeError('nonfinite Stage4 loss')
        scale_before = scaler.get_scale()
        scaler.scale(metrics['loss']).backward(); scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm): raise RuntimeError('nonfinite Stage4 gradient; no silent skipped update')
        scaler.step(optimizer); scaler.update()
        if scaler.get_scale() < scale_before: raise RuntimeError('AdamW update skipped by scaler')
        row = dict(step=step, level=example[0], gradient_norm=float(norm), **{k:float(v.detach()) for k,v in metrics.items()})
        records.append(row)
        with (output/'training.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(row)+'\n')
        print(json.dumps(dict(phase='train', **row)), flush=True)
        if step in (20,50,settings['steps']):
            torch.save(dict(schema_version=1, step=step, model_settings=model.settings,
                            model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                            scaler_state=scaler.state_dict(), torch_rng=torch.get_rng_state(),
                            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                            settings=settings, cache_sha256=sha(output/'events.pt'),
                            provenance=payload['provenance'], frozen_upstream_digest=teacher_before),
                       output/f'compressor-step{step}.pt')
    after = evaluate(model, dev, adapter, projection, tokenizer, device, settings['retain_weight'])
    teacher_after = upstream_digest(runtime)
    if teacher_before != teacher_after: raise RuntimeError('frozen upstream weights changed')
    result = dict(updates=len(records), fit_examples=len(fit), dev_examples=len(dev),
                  trainable_parameters=sum(p.numel() for p in model.parameters()),
                  frozen_upstream_digest=teacher_after, upstream_unchanged=True,
                  checkpoints={str(step): sha(output/f'compressor-step{step}.pt')
                               for step in sorted({20,50,settings['steps']}) if (output/f'compressor-step{step}.pt').exists()},
                  before=before, after=after)
    atomic_json(output/'training-result.json', result)
    model.eval()
    return model


def memory_for(settings, model, projection):
    return EventMemory(model, projection, **settings['memory'])


@torch.no_grad()
def replay(settings, runtime, tokenizer, model, frontend_raw, output):
    from continuous_mmwave.data_parrad import PARradManifestDataset
    from continuous_mmwave.train import load_experiment_config
    from continuous_mmwave.parrad_controller import FinalizationJob
    device = next(model.parameters()).device; controller = runtime.controller
    config = load_experiment_config(settings['frontend_config'])
    dataset = PARradManifestDataset(frontend_raw['data']['train_manifest'], config.model,
                                   {'TI_xWR14xx':0, 'TI_xWR68xx':1})
    indices = []
    for i, row in enumerate(dataset.rows):
        if row.get('subject_id') in settings['subjects'] and row.get('session_id') == 'set002' and row.get('sensor_id') in settings['sensors']:
            check_pilot_row(dict(row, split='train')); indices.append(i)
    reports = []
    for index in indices[:settings.get('replay_recordings',1)]:
        sample = dataset[index]; runtime.reset()
        memory = memory_for(settings, model, controller.semantic_projection)
        prompt = controller.event_former.build_prompt_cache(1)
        events = []; decisions = {'proposal':0,'confirm':0,'reject':0}; updates = 0
        recording = '__'.join((sample.subject_id, sample.session_id, sample.sensor_id))
        def finalize(job, now, confidence, source):
            tokens = finalized_tokens(controller.event_former, job.radar_tokens, job.radar_mask, prompt)
            key = controller.semantic_projection(tokens[None])[0]
            uid = f'{recording}:online:{len(events)}'
            memory.add(MemoryUnit(uid, tokens, job.start_time, job.end_time, now, confidence, key))
            events.append(dict(uid=uid, start=job.start_time, end=job.end_time, confirmed_at=now,
                               confidence=confidence, source=source))
        for start in range(0, len(sample.timestamps),2):
            stop=min(start+2,len(sample.timestamps)); valid=stop-start
            rd=sample.rd[start:stop].to(device); md=sample.md[start:stop].to(device)
            if valid<2:
                rd=torch.cat((rd,rd.new_zeros((2-valid,*rd.shape[1:])))); md=torch.cat((md,md.new_zeros((2-valid,md.shape[1]))))
            now=float(sample.timestamps[start])
            result=runtime.step(rd[None],md[None],torch.tensor([sample.sensor_index],device=device),
                                torch.arange(2,device=device)[None]<valid,now,
                                {k:getattr(sample,k) for k in ('subject_id','session_id','sensor_id')})
            updates+=1
            if result.decision in decisions: decisions[result.decision]+=1
            if result.finalization_job is not None:
                finalize(result.finalization_job,now,float(result.verify_probability), 'controller_confirm')
            if updates%100==0: print(json.dumps(dict(phase='replay',recording=recording,updates=updates,events=len(events))),flush=True)
        end=float(sample.timestamps[-1])+.09
        # A recording EOF is explicit, not a manufactured transition. Shadow
        # contains the active plus pending candidate evidence when still undecided.
        branch=controller.shadow if controller.candidate is not None else controller.active
        if branch is not None and branch.start_time<end:
            job=FinalizationJob(branch.start_time,end,torch.cat(branch.radar_chunks,1),torch.cat(branch.radar_masks,1),
                                branch.event_tokens,{})
            finalize(job,end,0.,'recording_eof_unconfirmed_tail')
        memory_snapshot=output/f'{recording}-memory.pt'; torch.save(memory.snapshot(),memory_snapshot)
        answers=[]
        for query in settings['queries']:
            query_key=encode_query(controller.event_former.adapter,controller.semantic_projection,tokenizer,query,device)
            context,selected=memory.context(query_key,now=end)
            answer=generate_from_memory(controller.event_former.adapter,tokenizer,context,query,
                                        max_new_tokens=int(settings.get('generation_max_new_tokens',96)))
            answers.append(dict(query=query,answer=answer,context_tokens=context.shape[1],
                                selected=[dict(uid=u.uid,level=u.level,start=u.start,end=u.end,count=u.count) for u in selected]))
        # Labels are consumed only now, for a retrospective boundary diagnostic.
        truth=[float(sample.timestamps[i]) for i in range(1,len(sample.labels))
               if int(sample.labels[i])>=0 and int(sample.labels[i-1])>=0 and sample.labels[i]!=sample.labels[i-1]]
        predicted=[e['end'] for e in events if e['source']=='controller_confirm']
        boundary=boundary_diagnostic(predicted,truth)
        reports.append(dict(recording=recording,updates=updates,decisions=decisions,events=events,
                            memory_counts=[len(x) for x in memory.levels],promotions=memory.promotions,
                            answers=answers,boundary_diagnostic=boundary,memory_sha256=sha(memory_snapshot),
                            online_boundaries='unchanged Stage3; EOF tail separately marked; no GT injection'))
    if not reports: raise ValueError('no approved online replay recordings')
    atomic_json(output/'online-replay.json',dict(status='complete',recordings=reports,
                is_unseen_subject_test=False,formal_go=False))
    return reports


@torch.no_grad()
def hierarchy_demonstration(settings, runtime, tokenizer, model, payload, output):
    """Exercise native-size L2 even if the unchanged online controller undersegments."""
    projection=runtime.controller.semantic_projection; device=next(model.parameters()).device
    recording=next(e['recording'] for e in payload['events'] if e['cohort']=='dev')
    rows=[e for e in payload['events'] if e['recording']==recording]
    memory=memory_for(settings,model,projection)
    for e in rows:
        memory.add(MemoryUnit(e['uid'],e['tokens'].to(device),e['start'],e['end'],e['end'],1.,e['key'].to(device)))
    query=settings['queries'][0]
    key=encode_query(runtime.controller.event_former.adapter,projection,tokenizer,query,device)
    context,selected=memory.context(key,now=rows[-1]['end'])
    answer=generate_from_memory(runtime.controller.event_former.adapter,tokenizer,context,query)
    torch.save(memory.snapshot(),output/'annotated-hierarchy-memory.pt')
    atomic_json(output/'annotated-hierarchy.json',dict(segment_source='GT offline diagnostic; not online controller',
                recording=recording,input_events=len(rows),memory_counts=[len(x) for x in memory.levels],
                promotions=memory.promotions,query=query,answer=answer,context_tokens=context.shape[1],
                selected=[dict(uid=u.uid,level=u.level,count=u.count) for u in selected]))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--mode',choices=('all','cache','train','replay'),default='all')
    args=parser.parse_args(argv)
    settings=yaml.safe_load(args.config.read_text(encoding='utf-8'))['stage4']
    output=Path(settings['output_dir']); output.mkdir(parents=True,exist_ok=True)
    if args.mode in ('all','train') and (output/'training.jsonl').exists():
        raise FileExistsError('refuse duplicate training in existing output')
    torch.manual_seed(settings['seed']); random.seed(settings['seed']); torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    provenance={name:sha(settings[name]) for name in ('frontend_config','stage3_checkpoint','threshold_artifact')}
    runtime,tokenizer,frontend_raw=load_frontend(settings,device)
    provenance['stage1_checkpoint']=sha(frontend_raw['stage']['stage1_checkpoint'])
    provenance['stage2_checkpoint']=sha(frontend_raw['stage3']['stage2_checkpoint'])
    provenance['event_former_code']=sha(Path(__file__).parent/'models/event_former.py')
    provenance['projection_code']=sha(Path(__file__).parent/'models/semantic.py')
    stage4_code={name:sha(Path(__file__).parent/name) for name in
                 ('models/memory.py','parrad_memory.py','stages/parrad_stage4.py','parrad_stage4_runner.py')}
    started=time.time(); status=dict(status='running',mode=args.mode,settings=settings,provenance=provenance,
                                    stage4_code=stage4_code,pilot_config_sha256=sha(args.config),
                                    train_only=True,formal_go=False,started_at=started)
    status_path=output/('status.json' if args.mode=='all' else f'status-{args.mode}.json')
    atomic_json(status_path,status)
    try:
        if args.mode in ('all','cache'):
            if (output/'events.pt').exists(): raise FileExistsError('refuse duplicate cache')
            payload=build_event_cache(settings,runtime,frontend_raw,output,provenance)
        else:
            manifest=json.loads((output/'cache-manifest.json').read_text())
            validate_provenance(manifest.get('provenance',{}),provenance)
            if sha(output/'events.pt')!=manifest['events_sha256']: raise ValueError('cache changed')
            payload=torch.load(output/'events.pt',map_location='cpu',weights_only=False)
        if args.mode in ('all','train'):
            model=train_compressor(settings,runtime,tokenizer,payload,output)
        elif args.mode=='replay':
            checkpoint=torch.load(output/f"compressor-step{settings['steps']}.pt",map_location=device,weights_only=False)
            validate_provenance(checkpoint.get('provenance',{}),provenance)
            if checkpoint['settings'] != settings: raise ValueError('checkpoint/settings mismatch')
            if checkpoint['frozen_upstream_digest'] != upstream_digest(runtime): raise ValueError('checkpoint teacher changed')
            if checkpoint['cache_sha256']!=sha(output/'events.pt'): raise ValueError('checkpoint/cache mismatch')
            model=QueryCompressor(**checkpoint['model_settings']).to(device)
            model.load_state_dict(checkpoint['model_state']); model.eval()
        if args.mode in ('all','replay'):
            hierarchy_demonstration(settings,runtime,tokenizer,model,payload,output)
            replay(settings,runtime,tokenizer,model,frontend_raw,output)
        status.update(status='complete',duration_s=time.time()-started)
    except Exception as error:
        status.update(status='failed',error=repr(error),duration_s=time.time()-started)
        atomic_json(status_path,status); raise
    atomic_json(status_path,status)
    print(json.dumps(dict(phase='complete',duration_s=status['duration_s'])),flush=True)


if __name__=='__main__': main()
