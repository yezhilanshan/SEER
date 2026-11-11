"""A failed epoch replay is isolated, deterministic and cannot expand invalid training."""
from pathlib import Path

import pytest
import torch

from test_parrad_projection_runner import fixture


def test_replay_next_epoch_matches_uninterrupted_optimizer_and_rejects_changed_cache(tmp_path):
    from continuous_mmwave.parrad_projection_runner import train_fold
    from continuous_mmwave.parrad_projection_recovery import replay_next_epoch
    old, data, caches = fixture()
    kwargs = dict(held_subject='subject_04', seed=1701, provenance={'source_manifest_sha256': 'synthetic'}, synthetic=True)
    train_fold(old, data, caches, tmp_path/'continuous', epochs=2, **kwargs)
    train_fold(old, data, caches, tmp_path/'parent', epochs=1, **kwargs)
    parent = tmp_path/'parent'/'last.pt'
    before = parent.read_bytes()
    replay_next_epoch(old, data, caches, parent, tmp_path/'replay')
    expected = torch.load(tmp_path/'continuous'/'last.pt', weights_only=False)
    actual = torch.load(tmp_path/'replay'/'replayed-epoch.pt', weights_only=False)
    assert actual['epoch'] == 2
    for key in expected['model_state']:
        torch.testing.assert_close(actual['model_state'][key], expected['model_state'][key], atol=0, rtol=0)
    assert parent.read_bytes() == before
    caches['fit']['manifest']['artifact_sha256'] = {'semantic.npz': 'changed'}
    with pytest.raises(ValueError, match='replay.*provenance'):
        replay_next_epoch(old, data, caches, parent, tmp_path/'changed')


def test_expansion_requires_improved_and_semantically_eligible_checkpoint():
    from continuous_mmwave.parrad_projection_recovery import can_expand
    assert not can_expand({'selected_epoch': 0, 'improved_dev': False})
    assert not can_expand({'selected_epoch': 2, 'improved_dev': False})
    assert not can_expand({'selected_epoch': 2, 'improved_dev': True}, semantic_complete=False)
    assert can_expand({'selected_epoch': 2, 'improved_dev': True})


def test_invalid_source_does_not_leave_an_output_directory(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_recovery as recovery
    def invalid(*args):
        raise ValueError('invalid source')
    monkeypatch.setattr(recovery, 'load_sequence_source', invalid)
    with pytest.raises(ValueError, match='invalid source'):
        recovery.run_recovery(tmp_path/'source', tmp_path/'parent', tmp_path/'output')
    assert not (tmp_path/'output').exists()
