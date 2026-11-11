"""Primary fit weighting, actual convex fitting, and interruption-safe P updates."""
import copy
import math

import numpy as np
import pytest
import torch


def fixture():
    from test_parrad_projection_runner import fixture as original
    old, data, caches = original()
    for r in data['rows']:
        r['probe_eligible'] = True
    return old, data, caches


def test_primary_weights_exactly_balance_age_and_class_without_candidate_renormalization():
    from continuous_mmwave.parrad_projection_primary import primary_cohort
    from test_parrad_projection_pilot import row
    rows = []
    for age in (6, 7):
        for target in (False, True):
            for j in range(1 + age - 6 + int(target)):
                rows.append(dict(row(subject='subject_03', age=age, target=target,
                                     candidate=f'{age}-{target}-{j}'), probe_eligible=True))
    rows += [dict(row(subject='subject_03', age=3), probe_eligible=True),
             dict(row(subject='subject_02', age=6), probe_eligible=True),
             dict(row(subject='subject_03', session='set002', age=6), probe_eligible=True),
             dict(row(subject='subject_03', age=6, candidate='excluded'), probe_eligible=False)]
    ids, weights, report = primary_cohort(rows, 'subject_02')
    assert len(ids) == 8 and weights.sum() == pytest.approx(1)
    for age in (6, 7):
        for target in (False, True):
            mask = [rows[i]['age_updates'] == age and rows[i]['update_target'] == target for i in ids]
            assert weights[mask].sum() == pytest.approx(.25)
    assert report['candidate_mass_min'] < report['candidate_mass_max']
    with pytest.raises(ValueError, match='duplicate'):
        primary_cohort(rows + [rows[0]], 'subject_02')
    with pytest.raises(ValueError, match='both classes'):
        primary_cohort([r for r in rows if not r['update_target']], 'subject_02')


def test_real_ridge_solver_converges_and_iteration_cap_is_not_success():
    from continuous_mmwave.parrad_projection_primary import fit_linear_head, weighted_bce
    torch.manual_seed(50)
    x = torch.randn(80, 5, dtype=torch.float64)
    y = (x[:, 0] + .3*x[:, 1] > 0).double()
    w = torch.where(y == 1, .5/y.sum(), .5/(1-y).sum())
    head = torch.nn.Linear(5, 1).double()
    head.weight.data.zero_(); head.bias.data.zero_()
    before = x.clone()
    result = fit_linear_head(head, x, y, w)
    assert result['converged'] and result['objective'] < math.log(2)
    assert result['iterations'] < 400
    torch.testing.assert_close(x, before, atol=0, rtol=0)
    head.weight.data.zero_(); head.bias.data.zero_()
    assert fit_linear_head(head, x, y, w, max_iter=1)['converged'] is False
    logits = head(x).flatten()
    torch.testing.assert_close(weighted_bce(logits, y, w),
        weighted_bce(logits[:17], y[:17], w[:17]) + weighted_bce(logits[17:], y[17:], w[17:]))


def test_already_converged_zero_iteration_head_has_finite_serializable_report(tmp_path):
    from continuous_mmwave.parrad_projection_primary import fit_linear_head
    from continuous_mmwave.parrad_projection_runner import write_json
    head = torch.nn.Linear(3, 1).double()
    head.weight.data.zero_(); head.bias.data.zero_()
    result = fit_linear_head(head, torch.zeros(2, 3), torch.tensor([0., 1.]), torch.tensor([.5, .5]))
    assert result['converged'] and result['iterations'] == 0
    assert result['loss_change'] is None  # No prior loss exists when the solver does no iteration.
    write_json(tmp_path/'head.json', result)


def test_fitted_heads_leave_teacher_and_projection_unchanged_then_task_gradient_reaches_p():
    from continuous_mmwave.parrad_projection_primary import context, fit_heads, task_loss
    old, data, _ = fixture()
    model, ctx = context(old, data, 'subject_04', 1701)
    before = copy.deepcopy(model.state_dict())
    report = fit_heads(model, data, ctx)
    assert report['converged'] and report['objective'] < math.log(2)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()
               if k.startswith(('projection.', 'teacher.')))
    model.projection.requires_grad_(True); model.heads.requires_grad_(False)
    task_loss(model, data, ctx).backward()
    assert sum(p.grad.square().sum() for p in model.projection.parameters()) > 0
    assert all(p.grad is None for p in model.teacher.parameters())


@pytest.mark.parametrize('interrupt_phase', ['head', 'evaluation'])
def test_resume_finishes_pending_phase_without_repeating_projection_step(tmp_path, monkeypatch, interrupt_phase):
    from continuous_mmwave import parrad_projection_primary as module
    old, data, caches = fixture()
    kwargs = dict(held='subject_04', seed=1701, steps=2, milestones=(1, 2), provenance={'synthetic': True})
    module.train_small_experiment(old, data, caches, tmp_path/'continuous', **kwargs)
    original_fit, original_eval = module.fit_heads, module.evaluate
    calls = [0]
    def fit(*args, **kw):
        calls[0] += 1
        if interrupt_phase == 'head' and calls[0] == 2:
            raise RuntimeError('head interrupted')
        return original_fit(*args, **kw)
    def evaluate(*args, **kw):
        if interrupt_phase == 'evaluation' and args[-1] == 1:
            raise RuntimeError('evaluation interrupted')
        return original_eval(*args, **kw)
    monkeypatch.setattr(module, 'fit_heads', fit); monkeypatch.setattr(module, 'evaluate', evaluate)
    with pytest.raises(RuntimeError, match='interrupted'):
        module.train_small_experiment(old, data, caches, tmp_path/'resumed', **kwargs)
    saved = torch.load(tmp_path/'resumed'/'joint-relation1'/'last.pt', weights_only=False)
    assert saved['step'] == 1 and saved['phase'] in ('head_fit_pending', 'evaluation_pending')
    monkeypatch.setattr(module, 'fit_heads', original_fit); monkeypatch.setattr(module, 'evaluate', original_eval)
    module.train_small_experiment(old, data, caches, tmp_path/'resumed', resume=True, **kwargs)
    warm = torch.load(tmp_path/'continuous'/'common-warmup.pt', weights_only=False)
    for name in ('frozen-control', 'joint-relation1', 'joint-relation10'):
        fork = torch.load(tmp_path/'continuous'/name/'fork.pt', weights_only=False)
        assert all(torch.equal(v, fork['model_state'][k]) for k, v in warm['model_state'].items())
        one = torch.load(tmp_path/'continuous'/name/'last.pt', weights_only=False)
        two = torch.load(tmp_path/'resumed'/name/'last.pt', weights_only=False)
        for k, v in one['model_state'].items():
            torch.testing.assert_close(v, two['model_state'][k], atol=0, rtol=0)
        assert one['phase'] == 'ready' and one['step'] == (0 if name == 'frozen-control' else 2)
        for k, v in one['optimizer_state']['state'].items():
            for field, value in v.items():
                torch.testing.assert_close(value, two['optimizer_state']['state'][k][field], atol=0, rtol=0)
        assert all(torch.equal(v, one['model_state']['teacher.'+k]) for k, v in old.state_dict().items())


def test_incomplete_warmup_and_unconverged_heads_do_not_unlock_projection(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_primary as module
    old, data, caches = fixture()
    before = copy.deepcopy(old.state_dict())
    kwargs = dict(held='subject_04', seed=1701, steps=1, milestones=(1,), provenance={'synthetic': True})
    root = tmp_path/'incomplete'; root.mkdir(); (root/'preflight.json').write_text('evidence')
    with pytest.raises(ValueError, match='warmup'):
        module.train_small_experiment(old, data, caches, root, resume=True, **kwargs)
    assert (root/'preflight.json').read_text() == 'evidence'
    monkeypatch.setattr(module, 'fit_heads', lambda *args, **kw: {'converged': False, 'objective': .8})
    with pytest.raises(ValueError, match='head'):
        module.train_small_experiment(old, data, caches, tmp_path/'failed', **kwargs)
    assert not (tmp_path/'failed'/'joint-relation1').exists()
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in before.items())


def test_failed_refit_stops_condition_and_locks_baseline_without_expanding_budget(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_primary as module
    old, data, caches = fixture()
    fit = module.fit_heads
    calls = [0]
    def fail_first_refit(*args, **kw):
        calls[0] += 1
        report = fit(*args, **kw)
        if calls[0] == 2:
            report['converged'] = False
        return report
    monkeypatch.setattr(module, 'fit_heads', fail_first_refit)
    results = module.train_small_experiment(old, data, caches, tmp_path/'run', held='subject_04', seed=1701,
                  steps=2, milestones=(1, 2), provenance={'synthetic': True})
    failed = results[1]
    assert failed['status'] == 'head_fit_failed' and failed['selected_step'] == 0
    assert failed['completed_projection_steps'] == 1
    assert results[2]['completed_projection_steps'] == 2
    saved = torch.load(tmp_path/'run'/'joint-relation1'/'last.pt', weights_only=False)
    assert saved['phase'] == 'stopped' and saved['head_report']['converged'] is False


def test_nonfinite_solver_failure_stops_condition_but_keeps_normal_interruptions_resumable(tmp_path, monkeypatch):
    from continuous_mmwave import parrad_projection_primary as module
    old, data, caches = fixture()
    solver = module.fit_linear_head
    calls = [0]
    def poison_first_refit(head, *args, **kw):
        calls[0] += 1
        if calls[0] == 4:
            head.weight.data.fill_(float('nan'))
        return solver(head, *args, **kw)
    monkeypatch.setattr(module, 'fit_linear_head', poison_first_refit)
    results = module.train_small_experiment(old, data, caches, tmp_path/'run', held='subject_04', seed=1701,
                   steps=1, milestones=(1,), provenance={'synthetic': True})
    assert results[1]['status'] == 'head_fit_failed' and results[1]['selected_step'] == 0
    assert results[2]['completed_projection_steps'] == 1
    saved = torch.load(tmp_path/'run'/'joint-relation1'/'last.pt', weights_only=False)
    assert saved['phase'] == 'stopped' and saved['head_report']['converged'] is False
    assert 'nonfinite' in saved['head_report']['heads'][0]['error']
