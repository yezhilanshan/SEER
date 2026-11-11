"""Role-gated TRAIN caches for Projection adaptation and semantic rehearsal."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from importlib import metadata as package_metadata
import uuid

import numpy as np
import torch

from continuous_mmwave.parrad_projection_pilot import role_masks
from continuous_mmwave.parrad_stage3_sequence_data import sha256


def _check_roles(roles, held_subject, selection_lock):
    allowed = {'fit', 'dev', 'calibration', 'held', 'probe_fit'}
    if not roles or not set(roles) <= allowed:
        raise ValueError('unknown data role')
    if set(roles) & {'held', 'calibration', 'probe_fit'}:
        if selection_lock is None:
            raise ValueError('final data roles require a saved selection lock')
        lock = json.loads(Path(selection_lock).read_text(encoding='utf-8'))
        if lock.get('status') != 'locked' or lock.get('held_subject') != held_subject:
            raise ValueError('selection lock does not match fold')


def load_transition_rows(source, checked, projection, held_subject, *, roles=('fit', 'dev'),
                         event_tokens=8, llm_dim=896, ages=range(3, 8), selection_lock=None):
    _check_roles(roles, held_subject, selection_lock)
    source = Path(source)
    metadata, current, reference = [], [], []
    max_error = 0.
    device = next(projection.parameters()).device
    for key, rows in checked['rows_by_record'].items():
        masks = role_masks(rows, held_subject)
        approved = np.logical_or.reduce([masks[role] for role in roles])
        selected = [i for i, r in enumerate(rows) if approved[i] and r.get('cohort') == 'natural'
                    and r.get('age_updates') in ages and type(r.get('update_target')) is bool]
        if not selected:
            continue
        if Path(key).name != key or '/' in key or '\\' in key:
            raise ValueError('unsafe recording key')
        with np.load(source / key / 'layers.npz', allow_pickle=False) as arrays:
            c = arrays['current_E'][selected].reshape(-1, event_tokens, llm_dim).copy()
            a = arrays['reference_E'][selected].reshape(-1, event_tokens, llm_dim).copy()
            zc, za = arrays['current_z'][selected], arrays['reference_z'][selected]
        if not np.isfinite(c).all() or not np.isfinite(a).all():
            raise ValueError('nonfinite Event State')
        with torch.no_grad():
            for offset in range(0, len(selected), 128):
                for values, expected in ((c, zc), (a, za)):
                    actual = projection(torch.from_numpy(values[offset:offset+128]).to(device)).cpu().numpy()
                    max_error = max(max_error, float(np.max(np.abs(actual - expected[offset:offset+128]))))
        if max_error > 1e-5:
            raise ValueError(f'epoch0 Projection does not reproduce cached z: {max_error}')
        metadata.extend({**rows[i], 'source_row': i} for i in selected)
        current.append(c); reference.append(a)
    if not metadata:
        raise ValueError('no clean natural updates for requested roles')
    return dict(rows=metadata, current=torch.from_numpy(np.concatenate(current)),
                reference=torch.from_numpy(np.concatenate(reference)), max_old_key_error=max_error)


def filtered_train_manifest(source, target, held_subject, *, roles, selection_lock=None):
    _check_roles(roles, held_subject, selection_lock)
    source, target = Path(source), Path(target)
    if target.exists():
        raise FileExistsError(target)
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines() if line.strip()]
    # Canonical TRAIN rows may not carry source_split; this is the explicit TRAIN file.
    masks = role_masks([{**r, 'source_split': 'train'} for r in rows], held_subject)
    keep = np.logical_or.reduce([masks[role] for role in roles])
    selected = []
    for r, approved in zip(rows, keep):
        if not approved:
            continue
        path = Path(r['path'])
        path = path if path.is_absolute() else (source.parent / path).resolve()
        selected.append({**r, 'path': str(path)})
    if not selected:
        raise ValueError('empty filtered semantic manifest')
    target.write_text('\n'.join(json.dumps(r) for r in selected) + '\n', encoding='utf-8')
    return selected


def load_rehearsal(path, held_subject, role, *, selection_lock=None):
    _check_roles((role,), held_subject, selection_lock)
    path = Path(path)
    rows = json.loads((path / 'rows.json').read_text(encoding='utf-8'))
    masks = role_masks(rows, held_subject)
    if not rows or not masks[role].all():
        raise ValueError('rehearsal contains rows outside requested role')
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('status') != 'complete' or manifest.get('held_subject') != held_subject or manifest.get('role') != role:
        raise ValueError('rehearsal cache manifest mismatch')
    if not manifest.get('dependency_sha256'):
        raise ValueError('semantic cache lacks upstream provenance')
    for dependency, digest in manifest['dependency_sha256'].items():
        if sha256(Path(dependency)) != digest:
            raise ValueError(f'changed semantic dependency: {dependency}')
    if manifest.get('upstream_code_sha256') != upstream_code_hashes():
        raise ValueError('semantic cache upstream code provenance mismatch')
    if manifest.get('runtime_versions') != runtime_versions():
        raise ValueError('semantic cache runtime provenance mismatch')
    for name in ('rows.json', 'semantic.npz'):
        if sha256(path / name) != manifest['artifact_sha256'][name]:
            raise ValueError('changed semantic cache')
    with np.load(path / 'semantic.npz', allow_pickle=False) as archive:
        result = {name: torch.from_numpy(archive[name].copy()) for name in archive.files}
    if len(result['labels']) != len(rows) or set(result['labels'].tolist()) != set(range(manifest['num_classes'])):
        raise ValueError('semantic cache lacks required action coverage')
    if not all(torch.isfinite(result[name]).all() for name in ('stream', 'commit', 'text')):
        raise ValueError('nonfinite semantic hidden states')
    return dict(tensors=result, rows=rows, manifest=manifest)


def upstream_code_hashes():
    root = Path(__file__).parent
    return {str(path.relative_to(root)): sha256(path) for path in sorted(root.rglob('*.py'))
            if not path.name.startswith('parrad_projection_')}


def runtime_versions():
    result = {}
    for name in ('torch', 'numpy', 'transformers', 'peft'):
        try:
            result[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            result[name] = None
    return result


def rehearsal_batch(cache, indices, device):
    tensors = cache['tensors']
    text_indices = tensors['text_index'][indices]
    return dict(stream=tensors['stream'][indices].to(device), commit=tensors['commit'][indices].to(device),
                text=tensors['text'][text_indices].to(device), mask=tensors['mask'][text_indices].to(device),
                labels=tensors['labels'][indices].to(device))


def prepare_rehearsal(source_manifest, output, held_subject, *, role, config_path, stage2_checkpoint,
                      device, selection_lock=None, per_class_record=4):
    """Replay original clean label-run windows with a frozen Stage2 runtime, TRAIN only."""
    from continuous_mmwave.checkpoint import load_training_checkpoint
    from continuous_mmwave.data_parrad import PARradManifestDataset, PARradWindowDataset, parrad_collate
    from continuous_mmwave.train import _build_parrad_stage2, _move_parrad, _tokenize_captions, load_experiment_config
    from continuous_mmwave.stages.parrad_stage2 import default_parrad_event_templates

    _check_roles((role,), held_subject, selection_lock)
    final_output = Path(output)
    if final_output.exists():
        raise FileExistsError(final_output)
    output = final_output.with_name(final_output.name+'.building-'+uuid.uuid4().hex)
    output.mkdir(parents=True)
    filtered = output / 'filtered-train.jsonl'
    selected = filtered_train_manifest(source_manifest, filtered, held_subject, roles=(role,),
                                       selection_lock=selection_lock)
    config = load_experiment_config(Path(config_path))
    module, tokenizer = _build_parrad_stage2(config)
    payload = load_training_checkpoint(stage2_checkpoint, module)
    original = payload['config']
    data_settings = original['data']
    stage_settings = original['stage']
    required = ('contrastive_temperature', 'consistency_weight')
    if any(key not in stage_settings for key in required):
        raise ValueError('checkpoint lacks actual Stage2 semantic objective settings')
    dependencies = {str(Path(path).resolve()): sha256(Path(path)) for path in
                    (config_path, stage2_checkpoint, source_manifest, config.stage['stage1_checkpoint'])}
    if config.llm.get('backend', 'qwen2') != 'tiny':
        backbone = Path(config.llm['name_or_path'])
        model_files = [p for p in backbone.rglob('*') if p.is_file()
                       and p.suffix in ('.json', '.safetensors', '.bin', '.model', '.txt')]
        if not model_files:
            raise ValueError('frozen backbone has no hashable local model files')
        dependencies.update({str(p.resolve()): sha256(p) for p in model_files})
    upstream_hashes = upstream_code_hashes()
    versions = runtime_versions()
    module.to(device).eval().requires_grad_(False)
    sensors = {'TI_xWR14xx': 0, 'TI_xWR68xx': 1}
    recordings = PARradManifestDataset(filtered, config.model, sensors)
    window_frames = int(data_settings.get('window_frames', 64))
    ignore_frames = round(float(data_settings.get('label_boundary_ignore_seconds', 0.)) / config.model.frame_period_s)
    windows = PARradWindowDataset(recordings, window_frames=window_frames, window_stride=window_frames,
                                 label_boundary_ignore_frames=ignore_frames, window_mode='label_run',
                                 min_label_run_frames=int(data_settings.get('min_label_run_frames', 1)))
    # Deterministic finite coverage: evenly spaced clean windows per recording/action.
    groups = defaultdict(list)
    for i, (recording, start, stop) in enumerate(windows.windows):
        valid = windows.recordings[recording].labels[start:stop]
        labels = valid[valid != -100].unique()
        if len(labels) != 1:
            raise ValueError('semantic rehearsal window is not a clean label-run')
        groups[recording, int(labels.item())].append(i)
    indices = []
    for values in groups.values():
        positions = np.linspace(0, len(values)-1, min(len(values), per_class_record), dtype=int)
        indices.extend(values[position] for position in positions)
    templates = default_parrad_event_templates()
    captions = tuple(text for label in sorted(templates) for text in templates[label])
    ids, mask = _tokenize_captions(tokenizer, captions)
    with torch.no_grad():
        text = module.event_former.adapter.encode_text(ids.to(device), mask.to(device)).float().cpu()
    stream, commit, rows, labels, text_indices = [], [], [], [], []
    for number, index in enumerate(sorted(indices)):
        sample = windows[index]
        label = int(sample.labels[sample.labels != -100].unique().item())
        batch = _move_parrad(parrad_collate([sample]), device)
        with torch.no_grad():
            radar = module.radar_encoder.forward_sequence(batch.rd, batch.md, sensor_index=batch.sensor_index,
                                                         frame_mask=batch.frame_mask)
            prompt = module.event_former.build_prompt_cache(1)
            state, _ = module._stream_view(radar.radar_tokens, radar.radar_token_mask, prompt)
            refined = module.event_former.commit_refine(radar.radar_tokens, radar.radar_token_mask, prompt).event_tokens
        stream.append(state.float().cpu()[0]); commit.append(refined.float().cpu()[0])
        recording, start, stop = windows.windows[index]
        row = selected[recording]
        rows.append(dict(source_split='train', subject_id=sample.subject_id, session_id=sample.session_id,
                         sensor_id=sample.sensor_id, label=label, start_frame=start, stop_frame=stop,
                         recording_path=row['path'], recording_sha256=sha256(Path(row['path']))
                         if not any(r.get('recording_path') == row['path'] for r in rows) else
                         next(r['recording_sha256'] for r in rows if r['recording_path'] == row['path'])))
        labels.append(label); text_indices.append(label * 3 + number % 3)
        if number % 32 == 0:
            print(json.dumps(dict(phase='semantic_cache', held_subject=held_subject, role=role,
                                  completed=number+1, total=len(indices))), flush=True)
    if set(labels) != set(range(len(templates))):
        raise ValueError('semantic rehearsal lacks14-class action coverage')
    dependencies.update({r['recording_path']: r['recording_sha256'] for r in rows})
    if any(sha256(Path(path)) != digest for path, digest in dependencies.items()) or upstream_code_hashes() != upstream_hashes:
        raise RuntimeError('semantic upstream dependency changed during cache preparation')
    np.savez_compressed(output / 'semantic.npz', stream=torch.stack(stream).numpy(), commit=torch.stack(commit).numpy(),
                        text=text.numpy(), mask=mask.numpy(), labels=np.array(labels, np.int64),
                        text_index=np.array(text_indices, np.int64))
    (output / 'rows.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    manifest = dict(status='complete', source_split='train', role=role, held_subject=held_subject,
                    num_classes=len(templates), per_class_record=per_class_record,
                    checkpoint=str(stage2_checkpoint), checkpoint_sha256=sha256(Path(stage2_checkpoint)),
                    source_manifest_sha256=sha256(Path(source_manifest)), config_sha256=sha256(Path(config_path)),
                    dependency_sha256=dependencies, upstream_code_sha256=upstream_hashes, runtime_versions=versions,
                    window_settings=data_settings, temperature=float(stage_settings['contrastive_temperature']),
                    consistency_weight=float(stage_settings['consistency_weight']),
                    artifact_sha256={name: sha256(output / name) for name in ('rows.json', 'semantic.npz')})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if final_output.exists():
        raise FileExistsError(final_output)
    output.rename(final_output)
    return manifest
