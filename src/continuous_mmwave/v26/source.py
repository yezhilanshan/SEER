"""Immutable sources, subject-separated manifests, and fitted legacy weights."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def require_sha(path, expected):
    actual = file_sha(path)
    if actual != expected:
        raise ValueError(f'SHA mismatch: {path}')
    return actual


def resolve_recording(manifest: Path, row: dict) -> Path:
    path = Path(row['path'])
    path = (path if path.is_absolute() else Path(manifest).parent / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def read_manifest(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def validate_splits(manifests):
    subjects, sources = {}, {}
    result = {}
    for role, path in manifests.items():
        rows = read_manifest(path)
        if not rows:
            raise ValueError('empty manifest')
        ids = set()
        for row in rows:
            subject = row['subject_id']
            if subject in subjects and subjects[subject] != role:
                raise ValueError('subject crosses split roles')
            subjects[subject] = role
            source = str(resolve_recording(path, row))
            identity = (subject, row['session_id'], row['sensor_id'])
            if source in sources or identity in ids:
                raise ValueError('duplicate source/recording identity')
            ids.add(identity)
            sources[source] = role
        result[role] = dict(rows=len(rows), subjects=sorted({r['subject_id'] for r in rows}), sha256=file_sha(path))
    return result


def validate_fitted_state(frontend, payload):
    state = payload['model_state']
    expected = ({name for name, p in frontend.named_parameters() if p.requires_grad and not name.startswith('radar_encoder.')}
                if payload.get('trainable_only') else set(frontend.state_dict()))
    missing = expected - set(state)
    extra = set(state) - set(frontend.state_dict())
    if missing or extra:
        raise ValueError(f'fitted checkpoint missing={sorted(missing)} unexpected={sorted(extra)}')
    if any(not torch.isfinite(v).all() for v in state.values() if v.is_floating_point()):
        raise ValueError('nonfinite fitted checkpoint')


def source_digest():
    root = Path(__file__).parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def state_digest(module):
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f'{name}:{value.shape}:{value.dtype}'.encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def audit_sources(config: dict) -> dict:
    return dict(checkpoints={key: require_sha(config[key], config[key + '_sha256']) for key in ('stage1', 'stage2')},
                splits=validate_splits(config['manifests']), code_sha256=source_digest(),
                frontend_assets=frontend_asset_lock(config['frontend_config']), formal_go=False)


def frontend_asset_lock(frontend_config):
    path = Path(frontend_config).resolve()
    config = yaml.safe_load(path.read_text())
    model = Path(config['llm']['name_or_path']).resolve()
    assets = [p for p in model.iterdir() if p.is_file() and
              (p.suffix in ('.json','.safetensors','.bin','.model') or p.name.startswith('tokenizer') or p.name=='merges.txt')]
    if not any(p.suffix in ('.safetensors','.bin') for p in assets):
        raise ValueError('local Qwen weights required')
    return dict(config_path=str(path),config_sha256=file_sha(path),model_dir=str(model),
                assets={p.name:file_sha(p) for p in sorted(assets)})


def require_frontend_assets(lock):
    require_sha(lock['config_path'],lock['config_sha256'])
    current=frontend_asset_lock(lock['config_path'])
    if current != lock: raise ValueError('SHA mismatch in Qwen/tokenizer assets')


def load_frozen_frontend(config_path, stage1_path, stage2_path, expected_shas, device):
    from ..train import _build_parrad_stage2, load_experiment_config
    require_sha(stage1_path, expected_shas['stage1'])
    require_sha(stage2_path, expected_shas['stage2'])
    config = load_experiment_config(config_path)
    config.stage['stage1_checkpoint'] = str(stage1_path)
    frontend, tokenizer = _build_parrad_stage2(config)
    payload = torch.load(stage2_path, map_location='cpu', weights_only=False)
    validate_fitted_state(frontend, payload)
    frontend.load_state_dict(payload['model_state'], strict=not payload.get('trainable_only'))
    frontend.requires_grad_(False).eval().to(device)
    former = frontend.event_former
    if former.state_gate.network[0].in_features != 2 * former.config.llm_dim + 1:
        raise ValueError('legacy fitted gate required')
    report = dict(frontend_sha256=state_digest(frontend), code_sha256=source_digest(),
                  frontend_config_sha256=file_sha(config_path), event_dim=former.config.llm_dim,
                  qwen_weight_files={p.name: file_sha(p) for p in sorted(Path(config.llm['name_or_path']).glob('*.safetensors'))})
    return frontend, tokenizer, report
