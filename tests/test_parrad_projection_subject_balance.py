"""Catch subject domination, split leakage and cross-objective checkpoint reuse."""
import copy
import json

import numpy as np
import pytest
import torch

from test_parrad_projection_pilot import row


def uneven_rows():
    # Every subject/age/class exists; subject02 contributes three times as many rows.
    return [dict(row(subject=s, age=a, target=y, candidate=f'{a}-{y}-{j}'), probe_eligible=True)
            for s, n in [('subject_02', 3), ('subject_03', 1)]
            for a in (6, 7) for y in (False, True) for j in range(n)]


def test_subject_buckets_keep_equal_mass_despite_unequal_counts_and_excluded_rows():
    from continuous_mmwave.parrad_projection_primary import primary_cohort
    rows = uneven_rows()
    excluded = [dict(row(subject='subject_04', age=6), probe_eligible=True),
                dict(row(session='set002', age=6), probe_eligible=True),
                dict(row(session='set003', age=6), probe_eligible=True),
                dict(row(age=5), probe_eligible=True),
                dict(row(age=6, candidate='ineligible'), probe_eligible=False)]
    ids, weights, report = primary_cohort(rows + excluded, 'subject_04', weighting='subject-age-class')
    assert ids.tolist() == list(range(16))
    np.testing.assert_allclose(weights, [1/24] * 12 + [1/8] * 4, rtol=0, atol=1e-16)
    assert weights[:12].sum() == pytest.approx(.5)
    assert weights[12:].sum() == pytest.approx(.5)
    assert len(report['subject_buckets']) == 8
    assert all(b['mass'] == pytest.approx(.125) for b in report['subject_buckets'])
    assert all(b['mass'] == pytest.approx(.25) for b in report['buckets'])
    # Excluded metadata must never change fit weights.
    _, clean, _ = primary_cohort(rows, 'subject_04', weighting='subject-age-class')
    np.testing.assert_array_equal(weights, clean)


@pytest.mark.parametrize('missing', ['subject', 'positive-age7'])
def test_missing_subject_cell_fails_instead_of_silently_redistributing_mass(missing):
    from continuous_mmwave.parrad_projection_primary import primary_cohort
    rows = [r for r in uneven_rows() if not (r['subject_id'] == 'subject_03' and
            (missing == 'subject' or (r['age_updates'] == 7 and r['update_target'])))]
    with pytest.raises(ValueError, match='subject.*bucket'):
        primary_cohort(rows, 'subject_04', weighting='subject-age-class')


def test_unknown_weighting_is_refused_before_creating_output(tmp_path):
    from continuous_mmwave.parrad_projection_primary import run
    with pytest.raises(ValueError, match='weighting'):
        run(tmp_path/'missing', tmp_path/'missing', tmp_path/'out', weighting='wrong')
    assert not (tmp_path/'out').exists()


def test_balanced_fit_changes_only_task_path_and_rejects_cross_weighting_resume(tmp_path):
    from continuous_mmwave import parrad_projection_primary as module
    from test_parrad_projection_primary import fixture
    old, data, caches = fixture()
    before = copy.deepcopy(old.state_dict())
    # Add genuine fit multiplicity, with a distinct Candidate identity.
    index = next(i for i, r in enumerate(data['rows']) if r['subject_id'] == 'subject_02'
                 and r['session_id'] == 'set000' and r['age_updates'] == 6 and r['update_target'])
    extra = dict(data['rows'][index], candidate_id='extra', source_row=999)
    data['rows'].append(extra)
    for key in ('current', 'reference'):
        data[key] = torch.cat([data[key], data[key][index:index+1]], dim=0)
    kwargs = dict(held='subject_04', seed=1701, steps=1, milestones=(1,), provenance={'synthetic': True})
    for label, mode in [('default', None), ('explicit', 'age-class'), ('balanced', 'subject-age-class')]:
        opts = {} if mode is None else {'weighting': mode}
        results = module.train_small_experiment(old, data, caches, tmp_path/label, **kwargs, **opts)
        assert [r['completed_projection_steps'] for r in results] == [0, 1, 1]
    warm = [torch.load(tmp_path/p/'common-warmup.pt', weights_only=False) for p in ('default', 'explicit', 'balanced')]
    assert warm[0]['identity'] == warm[1]['identity']
    assert warm[2]['identity']['settings']['weighting'] == 'subject-age-class'
    assert warm[0]['baseline']['dev'] == warm[2]['baseline']['dev']
    assert warm[0]['baseline']['retention'] == warm[2]['baseline']['retention']
    for k, v in warm[0]['model_state'].items():
        if k.startswith(('projection.', 'teacher.')):
            assert torch.equal(v, warm[2]['model_state'][k])
    for name in ('frozen-control', 'joint-relation1', 'joint-relation10'):
        states = [torch.load(tmp_path/p/name/'last.pt', weights_only=False) for p in ('default', 'explicit', 'balanced')]
        for k, v in states[0]['model_state'].items():
            assert torch.equal(v, states[1]['model_state'][k])
        for k, v in before.items():
            assert torch.equal(v, states[2]['model_state']['teacher.'+k])
        if name != 'frozen-control':
            assert any(not torch.equal(v, states[0]['model_state'][k])
                       for k, v in states[2]['model_state'].items() if k.startswith('projection.'))
            assert {int(s['step']) for s in states[2]['optimizer_state']['state'].values()} == {1}
    evidence = (tmp_path/'balanced'/'common-warmup.pt').read_bytes()
    weights_evidence = (tmp_path/'balanced'/'weights.json').read_bytes()
    with pytest.raises(ValueError, match='provenance'):
        module.train_small_experiment(old, data, caches, tmp_path/'balanced', resume=True, **kwargs)
    assert (tmp_path/'balanced'/'common-warmup.pt').read_bytes() == evidence
    assert (tmp_path/'balanced'/'weights.json').read_bytes() == weights_evidence
    module.train_small_experiment(old, data, caches, tmp_path/'balanced', resume=True,
                                  weighting='subject-age-class', **kwargs)
    assert all(torch.equal(v, old.state_dict()[k]) for k, v in before.items())
