import importlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch


def api():
    assert importlib.util.find_spec('continuous_mmwave.v26') is not None, 'v26 source contract missing'
    return importlib.import_module('continuous_mmwave.v26.source')


def test_relative_recording_path_and_subject_leak(tmp_path):
    module = api()
    path = tmp_path / 'records' / 'one.npz'
    path.parent.mkdir()
    path.write_bytes(b'npz placeholder')
    assert module.resolve_recording(tmp_path / 'train.jsonl', {'path': 'records/one.npz'}) == path
    row = {'path': str(path), 'subject_id': 's1', 'session_id': 'a', 'sensor_id': 'r1'}
    for role in ('train', 'validation'):
        (tmp_path / f'{role}.jsonl').write_text(json.dumps(row) + '\n')
    with pytest.raises(ValueError, match='subject'):
        module.validate_splits({r: tmp_path / f'{r}.jsonl' for r in ('train', 'validation')})


def test_source_sha_rejects_modified_checkpoint(tmp_path):
    module = api()
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'weights')
    with pytest.raises(ValueError, match='SHA'):
        module.require_sha(path, '0' * 64)


def test_partial_stage2_requires_all_fitted_nonencoder_parameters():
    module = api()
    front = torch.nn.Module()
    front.radar_encoder = torch.nn.Linear(2, 2)
    front.event_former = torch.nn.Linear(2, 2)
    front.semantic_projection = torch.nn.Linear(2, 2)
    state = {k: v for k, v in front.state_dict().items() if not k.startswith('radar_encoder.')}
    module.validate_fitted_state(front, {'trainable_only': True, 'model_state': state})
    state.pop('semantic_projection.weight')
    with pytest.raises(ValueError, match='semantic_projection.weight'):
        module.validate_fitted_state(front, {'trainable_only': True, 'model_state': state})


def test_stage2_unknown_keys_are_rejected():
    module = api()
    front = torch.nn.Linear(2, 2)
    state = dict(front.state_dict(), unknown=torch.zeros(1))
    with pytest.raises(ValueError, match='unexpected'):
        module.validate_fitted_state(front, {'trainable_only': True, 'model_state': state})


def test_frontend_asset_lock_rejects_nested_yaml_and_tokenizer_changes(tmp_path):
    module=api()
    model=tmp_path/'qwen'; model.mkdir()
    (model/'model.safetensors').write_bytes(b'frozen weights')
    tokenizer=model/'tokenizer.json'; tokenizer.write_text('{}')
    frontend=tmp_path/'frontend.yaml'; frontend.write_text(f'llm:\n  name_or_path: {model.as_posix()}\n')
    assert hasattr(module,'frontend_asset_lock'), 'pre-load frontend asset lock missing'
    lock=module.frontend_asset_lock(frontend)
    module.require_frontend_assets(lock)
    tokenizer.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='SHA'): module.require_frontend_assets(lock)
    tokenizer.write_text('{}'); frontend.write_text(frontend.read_text()+'model: {}\n')
    with pytest.raises(ValueError,match='SHA'): module.require_frontend_assets(lock)
