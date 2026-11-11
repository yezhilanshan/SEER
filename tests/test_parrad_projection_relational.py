"""Teacher relations must preserve old geometry and cross-version compatibility."""
import copy

import pytest
import torch

from continuous_mmwave.models.semantic import SemanticProjection


def keys():
    return dict(stream=torch.eye(2), commit=torch.tensor([[0., 1.], [1., 0.]]),
                text=torch.eye(2))


def test_teacher_equality_is_stationary_without_increasing_stream_commit_similarity():
    from continuous_mmwave.parrad_projection_relational import relation_terms
    old = keys()
    new = {k: v.clone().requires_grad_() for k, v in old.items()}
    terms = relation_terms(new, old, temperature=.07)
    assert terms['relation'].item() == 0
    assert terms['consistency'].item() == 0
    sum(terms.values()).backward()
    assert all(torch.count_nonzero(v.grad) == 0 for v in new.values())


def test_common_rotation_preserves_internal_retrieval_but_cross_version_loss_detects_it():
    from continuous_mmwave.parrad_projection_relational import relation_terms
    old = keys()
    rotation = torch.tensor([[0., -1.], [1., 0.]])
    new = {k: (v @ rotation).requires_grad_() for k, v in old.items()}
    torch.testing.assert_close(new['stream'] @ new['text'].T, torch.eye(2))
    terms = relation_terms(new, old, temperature=1.)
    # Each cross-version matrix has mean squared error1; internal error0.
    assert terms['relation'].item() == pytest.approx(2 / 3)
    terms['relation'].backward()
    assert new['text'].grad.abs().sum() > 0
    assert new['commit'].grad.abs().sum() > 0


def test_fixed_gallery_relation_is_invariant_to_unequal_event_query_chunks():
    from continuous_mmwave.parrad_projection_relational import relation_terms
    torch.manual_seed(12)
    old = {k: torch.nn.functional.normalize(torch.randn(n, 6), dim=-1)
           for k, n in [('stream', 5), ('commit', 5), ('text', 3)]}
    new = {k: torch.nn.functional.normalize(v + .03 * torch.randn_like(v), dim=-1).requires_grad_()
           for k, v in old.items()}
    full = sum(relation_terms(new, old, temperature=.07).values())
    gradients = torch.autograd.grad(full, tuple(new.values()), retain_graph=True)
    chunked = 0
    for first, last in [(0, 2), (2, 5)]:
        n = {k: v if k == 'text' else v[first:last] for k, v in new.items()}
        o = {k: v if k == 'text' else v[first:last] for k, v in old.items()}
        chunked += (last-first) / 5 * sum(relation_terms(n, o, temperature=.07).values())
    torch.testing.assert_close(full, chunked)
    for a, b in zip(gradients, torch.autograd.grad(chunked, tuple(new.values()))):
        torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)


def test_bank_rejects_dev_or_held_rows_before_reading_numeric_tensors():
    from continuous_mmwave.parrad_projection_relational import make_bank
    from test_parrad_projection_pilot import row
    cache = dict(rows=[row(subject='subject_03', session='set002')], manifest={}, tensors={})
    with pytest.raises(ValueError, match='fit'):
        make_bank(SemanticProjection(8, 128), cache, 'subject_02')
    cache['rows'] = [row(subject='subject_02')]
    with pytest.raises(ValueError, match='fit'):
        make_bank(SemanticProjection(8, 128), cache, 'subject_02')


def test_head_warmup_changes_heads_only_then_task_gradient_reaches_projection():
    from continuous_mmwave.parrad_projection_relational import RelationalAdaptation, transition_loss, make_optimizer
    torch.manual_seed(9)
    old = SemanticProjection(8, 128)
    model = RelationalAdaptation(old, [3407], [1.], torch.zeros(1, 130), torch.ones(1, 130))
    old_state = copy.deepcopy(old.state_dict())
    c, a = torch.randn(4, 2, 8), torch.randn(4, 2, 8)
    targets, weights = torch.tensor([0., 1., 0., 1.]), torch.ones(4)
    model.projection.requires_grad_(False)
    opt = make_optimizer(model, joint=False)
    for _ in range(2):
        opt.zero_grad()
        loss = transition_loss(model, c, a, targets, weights)
        loss.backward(); opt.step()
    assert all(torch.equal(old_state[k], v) for k, v in model.projection.state_dict().items())
    assert all(torch.equal(old_state[k], v) for k, v in model.teacher.state_dict().items())
    assert model.heads[0].weight.abs().sum() > 0
    model.projection.requires_grad_(True)
    opt = make_optimizer(model, joint=True)
    opt.zero_grad()
    loss = transition_loss(model, c, a, targets, weights)
    loss.backward()
    assert sum(p.grad.square().sum() for p in model.projection.parameters()) > 0
    opt.step()
    assert any(not torch.equal(old_state[k], v) for k, v in model.projection.state_dict().items())
    assert all(p.grad is None for p in model.teacher.parameters())


def test_three_conditions_share_warmup_and_head_only_really_preserves_projection(tmp_path):
    from continuous_mmwave.parrad_projection_relational import train_small_experiment
    from test_parrad_projection_runner import fixture
    projection, data, caches = fixture()
    before = copy.deepcopy(projection.state_dict())
    result = train_small_experiment(projection, data, caches, tmp_path/'run', held='subject_04',
                                    seed=1701, joint_epochs=1, provenance={'synthetic': True})
    assert len(result) == 3
    warm = torch.load(tmp_path/'run'/'common-warmup.pt', weights_only=False)
    assert all(torch.equal(v, warm['model_state']['projection.'+k]) for k, v in before.items())
    for name in ('head-only', 'joint-relation1', 'joint-relation10'):
        root = tmp_path/'run'/name
        initial = torch.load(root/'fork.pt', weights_only=False)
        assert all(torch.equal(v, initial['model_state'][k]) for k, v in warm['model_state'].items())
        last = torch.load(root/'last.pt', weights_only=False)
        assert last['epoch'] == 2 and last['evaluation_complete']
        assert all(torch.equal(v, last['model_state']['teacher.'+k]) for k, v in before.items())
        if name == 'head-only':
            assert all(torch.equal(v, last['model_state']['projection.'+k]) for k, v in before.items())
        else:
            assert any(not torch.equal(v, last['model_state']['projection.'+k]) for k, v in before.items())


def test_pending_evaluation_resume_does_not_repeat_relation_optimizer_updates(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_relational as module
    from test_parrad_projection_runner import fixture
    projection, data, caches = fixture()
    kwargs = dict(held='subject_04', seed=1701, joint_epochs=1, provenance={'synthetic': True})
    module.train_small_experiment(projection, data, caches, tmp_path/'continuous', **kwargs)
    evaluate = module._evaluate
    def fail_after_optimizer(*args, **kw):
        if args[-1] == 2 and args[-2].name == 'joint-relation1':
            raise RuntimeError('evaluation interrupted')
        return evaluate(*args, **kw)
    monkeypatch.setattr(module, '_evaluate', fail_after_optimizer)
    with pytest.raises(RuntimeError, match='interrupted'):
        module.train_small_experiment(projection, data, caches, tmp_path/'resumed', **kwargs)
    saved = torch.load(tmp_path/'resumed'/'joint-relation1'/'last.pt', weights_only=False)
    assert saved['evaluation_complete'] is False
    monkeypatch.setattr(module, '_evaluate', evaluate)
    module.train_small_experiment(projection, data, caches, tmp_path/'resumed', resume=True, **kwargs)
    for name in ('head-only', 'joint-relation1', 'joint-relation10'):
        one = torch.load(tmp_path/'continuous'/name/'last.pt', weights_only=False)
        two = torch.load(tmp_path/'resumed'/name/'last.pt', weights_only=False)
        for k, v in one['model_state'].items():
            torch.testing.assert_close(v, two['model_state'][k], atol=0, rtol=0)
        assert one['rng_state']['numpy'] == two['rng_state']['numpy']
        assert one['history'][-1]['optimizer_steps'] == two['history'][-1]['optimizer_steps']


def test_resume_without_completed_warmup_preserves_previous_evidence(tmp_path):
    from continuous_mmwave.parrad_projection_relational import train_small_experiment
    from test_parrad_projection_runner import fixture
    old, data, caches = fixture()
    root = tmp_path/'incomplete'
    root.mkdir()
    (root/'preflight.json').write_text('original evidence')
    with pytest.raises(ValueError, match='warmup'):
        train_small_experiment(old, data, caches, root, held='subject_04', seed=1701, joint_epochs=1,
                               provenance={'synthetic': True}, resume=True)
    assert (root/'preflight.json').read_text() == 'original evidence'


def test_warmup_audit_rejects_zero_task_gradient_without_changing_frozen_weights():
    from continuous_mmwave.parrad_projection_relational import audit_warmup, _context
    from test_parrad_projection_runner import fixture
    old, data, _ = fixture()
    model, ctx = _context(old, data, 'subject_04', 1701)
    model.projection.requires_grad_(False)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match='gradient'):
        audit_warmup(model, old, data, ctx)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert not any(p.requires_grad for p in model.projection.parameters())
