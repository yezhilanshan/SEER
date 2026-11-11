"""Catch accessing numeric held/calibration data before selection is locked."""
import json
import hashlib

import numpy as np
import pytest
import torch

from continuous_mmwave.models.semantic import SemanticProjection
from test_parrad_projection_pilot import row


def test_numeric_trace_loading_opens_only_requested_roles_and_checks_old_keys(tmp_path):
    from continuous_mmwave.parrad_projection_data import load_transition_rows
    projection = SemanticProjection(8, 128)
    metadata = [row(candidate='a', target=False), row(candidate='b', target=True),
                row(subject='subject_04')]
    by_record = {}
    for r in metadata:
        r.update(current_update=10 + len(by_record.get(r['recording_key'], [])),
                 active_ref_update=1, proposal_update=8, active_ref_state_id='active:1')
        by_record.setdefault(r['recording_key'], []).append(r)
    for key, rows in by_record.items():
        directory = tmp_path / key; directory.mkdir()
        if key.startswith('subject_04'):
            # A loader that touches held numeric inputs before lock will fail here.
            (directory / 'layers.npz').write_bytes(b'not an archive')
            continue
        current, reference = torch.randn(len(rows), 2, 8), torch.randn(len(rows), 2, 8)
        np.savez(directory / 'layers.npz', current_E=current.reshape(len(rows), -1).numpy(),
                 reference_E=reference.reshape(len(rows), -1).numpy(),
                 current_z=projection(current).detach().numpy(),
                 reference_z=projection(reference).detach().numpy())
    checked = {'rows_by_record': by_record}
    data = load_transition_rows(tmp_path, checked, projection, 'subject_04', roles=('fit',),
                                event_tokens=2, llm_dim=8)
    assert len(data['rows']) == 2
    assert data['max_old_key_error'] < 1e-5
    assert data['current'].shape == (2, 2, 8)
    with pytest.raises(ValueError):
        load_transition_rows(tmp_path, checked, projection, 'subject_04', roles=('held',),
                             event_tokens=2, llm_dim=8)


def test_filtered_semantic_manifest_has_no_forbidden_recording_paths(tmp_path):
    from continuous_mmwave.parrad_projection_data import filtered_train_manifest
    source = tmp_path / 'train.jsonl'
    rows = [dict(row(session=session, subject=subject), path=f'{subject}_{session}.npz', environment_id='room')
            for subject in ('subject_02', 'subject_03', 'subject_04')
            for session in ('set000', 'set001', 'set002', 'set003')]
    source.write_text('\n'.join(json.dumps(r) for r in rows))
    target = tmp_path / 'filtered.jsonl'
    selected = filtered_train_manifest(source, target, 'subject_04', roles=('fit', 'dev'))
    assert len(selected) == 6
    assert {r['session_id'] for r in selected} == {'set000', 'set001', 'set002'}
    assert all(r['subject_id'] != 'subject_04' for r in selected)
    assert all(str(tmp_path) in r['path'] for r in selected)
    with pytest.raises(FileExistsError):
        filtered_train_manifest(source, target, 'subject_04', roles=('fit',))


def test_rehearsal_loader_rejects_calibration_before_numeric_open(tmp_path):
    from continuous_mmwave.parrad_projection_data import load_rehearsal
    cache = tmp_path / 'semantic'
    cache.mkdir()
    metadata = [dict(row(session='set003'), label=0)]
    (cache / 'rows.json').write_text(json.dumps(metadata))
    (cache / 'semantic.npz').write_bytes(b'forbidden numeric data')
    with pytest.raises(ValueError, match='role'):
        load_rehearsal(cache, 'subject_04', 'fit')


def test_complete_cache_rejects_changed_upstream_input_before_reuse(tmp_path):
    from continuous_mmwave.parrad_projection_data import load_rehearsal
    from continuous_mmwave.parrad_stage3_sequence_data import sha256
    cache = tmp_path/'cache'; cache.mkdir()
    upstream = tmp_path/'record.npz'; upstream.write_bytes(b'original-recording')
    rows = [dict(row(), label=0), dict(row(), label=1)]
    (cache/'rows.json').write_text(json.dumps(rows))
    np.savez(cache/'semantic.npz', stream=np.ones((2, 2, 8), np.float32), commit=np.ones((2, 2, 8), np.float32),
             text=np.ones((2, 3, 8), np.float32), mask=np.ones((2, 3), bool),
             labels=np.array([0, 1]), text_index=np.array([0, 1]))
    manifest = dict(status='complete', held_subject='subject_04', role='fit', num_classes=2,
                    dependency_sha256={str(upstream): hashlib.sha256(b'original-recording').hexdigest()},
                    artifact_sha256={name: sha256(cache/name) for name in ('rows.json', 'semantic.npz')})
    (cache/'manifest.json').write_text(json.dumps(manifest))
    upstream.write_bytes(b'changed-recording')
    with pytest.raises(ValueError, match='dependency|provenance'):
        load_rehearsal(cache, 'subject_04', 'fit')


def test_failed_cache_preparation_does_not_publish_final_directory(tmp_path):
    from continuous_mmwave.parrad_projection_data import prepare_rehearsal
    source = tmp_path/'train.jsonl'
    source.write_text(json.dumps(dict(row(), path='missing.npz', environment_id='room')))
    output = tmp_path/'cache'
    with pytest.raises(FileNotFoundError):
        prepare_rehearsal(source, output, 'subject_04', role='fit', config_path=tmp_path/'missing.yaml',
                          stage2_checkpoint=tmp_path/'missing.pt', device=torch.device('cpu'))
    assert not output.exists()
    assert len(list(tmp_path.glob('cache.building-*'))) == 1
