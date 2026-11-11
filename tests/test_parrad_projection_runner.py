"""Real optimizer/checkpoint smoke; synthetic metrics are never production results."""
import copy

import numpy as np
import torch
import pytest

from continuous_mmwave.models.semantic import SemanticProjection
from test_parrad_projection_pilot import row


def fixture():
    torch.manual_seed(10)
    projection = SemanticProjection(8, 128)
    rows = [row(subject=s, session=t, candidate=f'{label}', age=a, target=bool(label))
            for s in ('subject_02', 'subject_03') for t in ('set000', 'set002')
            for label in (0, 1) for a in (6, 7)]
    for i, r in enumerate(rows):
        r.update(source_row=i, current_update=20+r['age_updates'])
    data = dict(rows=rows, current=torch.randn(len(rows), 2, 8), reference=torch.randn(len(rows), 2, 8))
    caches = {}
    for role, session in [('fit', 'set000'), ('dev', 'set002')]:
        labels = torch.tensor([0, 0, 1, 1] * 2)
        text = torch.randn(2, 4, 8)
        state = torch.randn(8, 2, 8)
        caches[role] = dict(tensors=dict(stream=state, commit=state+.01*torch.randn_like(state),
                                        text=text, mask=torch.ones(2, 4, dtype=torch.bool),
                                        labels=labels, text_index=labels.clone()),
                           rows=[dict(row(subject=s, session=session), label=int(label))
                                 for s in ('subject_02', 'subject_03') for label in labels[:4]],
                           manifest=dict(temperature=.07, consistency_weight=.1))
    return projection, data, caches


def test_fold_smoke_saves_current_optimizer_locked_selection_and_frozen_teacher(tmp_path):
    from continuous_mmwave.parrad_projection_runner import train_fold
    projection, data, caches = fixture()
    old = copy.deepcopy(projection.state_dict())
    result = train_fold(projection, data, caches, tmp_path / 'fold', held_subject='subject_04',
                        seed=1701, epochs=1, provenance={'source_manifest_sha256': 'synthetic'}, synthetic=True)
    assert result['status'] == 'locked'
    assert result['completed_epochs'] == 1
    last = torch.load(tmp_path/'fold'/'last.pt', weights_only=False)
    assert last['optimizer_state']['state']
    assert all(torch.equal(old[k], last['model_state']['teacher.'+k]) for k in old)
    assert any(not torch.equal(old[k], last['model_state']['projection.'+k]) for k in old)
    assert result['head_only']['teacher_unchanged'] is True
    assert (tmp_path/'fold'/'selection-lock.json').is_file()
    assert last['rng_state']


def test_resume_restores_rng_and_optimizer_without_repeating_completed_epochs(tmp_path):
    from continuous_mmwave.parrad_projection_runner import train_fold
    projection, data, caches = fixture()
    provenance = {'source_manifest_sha256': 'synthetic'}
    train_fold(projection, data, caches, tmp_path/'continuous', held_subject='subject_04', seed=1701,
               epochs=2, provenance=provenance, synthetic=True)
    train_fold(projection, data, caches, tmp_path/'resumed', held_subject='subject_04', seed=1701,
               epochs=1, provenance=provenance, synthetic=True)
    train_fold(projection, data, caches, tmp_path/'resumed', held_subject='subject_04', seed=1701,
               epochs=2, provenance=provenance, synthetic=True, resume=True)
    one = torch.load(tmp_path/'continuous'/'last.pt', weights_only=False)
    two = torch.load(tmp_path/'resumed'/'last.pt', weights_only=False)
    assert one['epoch'] == two['epoch'] == 2
    for name in one['model_state']:
        torch.testing.assert_close(one['model_state'][name], two['model_state'][name], atol=0, rtol=0)


def test_resume_rejects_same_rows_with_changed_semantic_artifact_identity(tmp_path):
    from continuous_mmwave.parrad_projection_runner import train_fold
    projection, data, caches = fixture()
    caches['fit']['manifest']['artifact_sha256'] = {'semantic.npz': 'original'}
    train_fold(projection, data, caches, tmp_path/'fold', held_subject='subject_04', seed=1701,
               epochs=1, provenance={'source_manifest_sha256': 'synthetic'}, synthetic=True)
    caches['fit']['manifest']['artifact_sha256']['semantic.npz'] = 'changed'
    with pytest.raises(ValueError, match='provenance'):
        train_fold(projection, data, caches, tmp_path/'fold', held_subject='subject_04', seed=1701,
                   epochs=2, provenance={'source_manifest_sha256': 'synthetic'}, synthetic=True, resume=True)


def test_unconverged_action_probe_records_diagnostics_without_a_score():
    from continuous_mmwave.parrad_projection_runner import _action_probe
    torch.manual_seed(30)
    keys = torch.randn(40, 128)
    labels = torch.arange(40) % 4
    diagnostics = []
    result = _action_probe(keys, labels, keys, labels, max_iter=1, diagnostics=diagnostics)
    assert result['macro_f1'] is None
    assert diagnostics[0]['converged'] is False
    assert diagnostics[0]['iterations'] == 1
    assert diagnostics[0]['gradient_max'] > 1e-7
    assert diagnostics[0]['finite'] is True
    assert diagnostics[0]['fit_classes'] == [10, 10, 10, 10]


def test_undefined_dev_retention_excludes_epoch_without_stopping_training(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_runner as runner
    projection, data, caches = fixture()
    original = runner.semantic_evaluation
    calls = 0
    def evaluate(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls > 1:
            result['subject_02:action_stream']['macro_f1'] = None
        return result
    monkeypatch.setattr(runner, 'semantic_evaluation', evaluate)
    result = runner.train_fold(projection, data, caches, tmp_path/'fold', held_subject='subject_04', seed=1701,
                               epochs=2, provenance={'source_manifest_sha256': 'synthetic'}, synthetic=True)
    assert result['selected_epoch'] == 0
    assert result['completed_epochs'] == 2
    history = __import__('json').loads((tmp_path/'fold'/'history.json').read_text())
    assert history[1]['eligible'] is False
    assert history[1]['semantic_complete'] is False


def test_pending_evaluation_resume_does_not_repeat_optimizer_updates(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_runner as runner
    projection, data, caches = fixture()
    kwargs = dict(held_subject='subject_04', seed=1701, epochs=2,
                  provenance={'source_manifest_sha256': 'synthetic'}, synthetic=True)
    runner.train_fold(projection, data, caches, tmp_path/'continuous', **kwargs)
    original = runner.dev_probes
    calls = 0
    def interrupted(*args, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('evaluation interrupted')
        return original(*args, **kw)
    monkeypatch.setattr(runner, 'dev_probes', interrupted)
    with pytest.raises(RuntimeError, match='evaluation interrupted'):
        runner.train_fold(projection, data, caches, tmp_path/'resumed', **kwargs)
    pending = torch.load(tmp_path/'resumed'/'last.pt', weights_only=False)
    assert pending['epoch'] == 1 and pending['evaluation_complete'] is False
    assert pending['pending_evaluation']['epoch'] == 1
    monkeypatch.setattr(runner, 'dev_probes', original)
    runner.train_fold(projection, data, caches, tmp_path/'resumed', resume=True, **kwargs)
    one = torch.load(tmp_path/'continuous'/'last.pt', weights_only=False)
    two = torch.load(tmp_path/'resumed'/'last.pt', weights_only=False)
    assert two['epoch'] == 2 and two['evaluation_complete'] is True
    for name in one['model_state']:
        torch.testing.assert_close(one['model_state'][name], two['model_state'][name], atol=0, rtol=0)


def test_missing_baseline_retention_stops_before_parameter_training(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_runner as runner
    projection, data, caches = fixture()
    monkeypatch.setattr(runner, 'semantic_evaluation', lambda *a, **kw: {'action': {'macro_f1': None}})
    with pytest.raises(ValueError, match='baseline semantic'):
        runner.train_fold(projection, data, caches, tmp_path/'fold', held_subject='subject_04', seed=1701,
                          epochs=1, provenance={'source_manifest_sha256': 'synthetic'}, synthetic=True)
    assert not (tmp_path/'fold'/'last.pt').exists()
