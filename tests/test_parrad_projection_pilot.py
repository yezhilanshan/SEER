"""Protect the adaptation split, equal Candidate mass and the shared semantic space."""
import copy

import numpy as np
import pytest
import torch

from continuous_mmwave.models.semantic import SemanticProjection


def row(subject='subject_02', session='set000', candidate='a', age=3, target=False):
    return dict(source_split='train', subject_id=subject, session_id=session,
                sensor_id='TI_xWR14xx', recording_key=f'{subject}__{session}__TI_xWR14xx',
                candidate_id=candidate, cohort='natural', age_updates=age, update_target=target)


def test_projection_roles_exclude_dev_calibration_and_held_from_parameter_fit():
    from continuous_mmwave.parrad_projection_pilot import role_masks
    rows = [row(), row(session='set001'), row(session='set002'), row(session='set003'),
            row(subject='subject_04')]
    masks = role_masks(rows, 'subject_04')
    assert masks['fit'].tolist() == [True, True, False, False, False]
    assert masks['dev'].tolist() == [False, False, True, False, False]
    assert masks['calibration'].tolist() == [False, False, False, True, False]
    assert masks['held'].tolist() == [False, False, False, False, True]
    assert masks['probe_fit'].tolist() == [True, True, True, False, False]
    with pytest.raises(ValueError):
        role_masks([dict(row(), source_split='test')], 'subject_04')


def test_final_candidate_weights_equal_one_after_age_class_correction():
    from continuous_mmwave.parrad_projection_pilot import balanced_candidate_weights
    rows = [row(candidate='a', age=3), row(candidate='a', age=4, target=True),
            row(candidate='b', age=3, target=True), row(candidate='c', age=4),
            row(candidate='d', age=5, target=True)]
    weights, report = balanced_candidate_weights(rows)
    assert weights[0] + weights[1] == pytest.approx(1)
    assert weights[2] == pytest.approx(1)
    assert weights[3] == pytest.approx(1)
    assert weights[4] == 0
    assert report['excluded_ages'] == [5]
    assert report['candidates'] == 3
    with pytest.raises(ValueError):
        balanced_candidate_weights([row(session='set002')])
    with pytest.raises(ValueError):
        balanced_candidate_weights([row(), row()])


def test_differentiable_features_match_existing_protocol_and_reach_projection():
    from continuous_mmwave.parrad_projection_pilot import relational_features
    from continuous_mmwave.parrad_stage3_sequence_features import project_history_block
    projection = SemanticProjection(8, 128)
    current = projection(torch.randn(5, 3, 8))
    reference = projection(torch.randn(5, 3, 8))
    actual = relational_features(current, reference, seed=3407, scale=2.)
    expected = project_history_block(
        {'z': current.detach().numpy()[:, None]}, {'z': reference.detach().numpy()[:, None]},
        sources=('z',), seed=3407, scales={'z': 2.})
    assert actual.shape == (5, 130)
    np.testing.assert_allclose(actual.detach(), expected, atol=2e-6, rtol=2e-6)
    actual.sum().backward()
    assert projection.semantic_query.grad.abs().sum() > 0
    assert projection.projection[0].weight.grad.abs().sum() > 0


def test_semantic_retention_recognizes_duplicate_action_positives_and_both_directions():
    from continuous_mmwave.parrad_projection_pilot import semantic_metrics
    keys = torch.tensor([[1., 0.], [1., 0.], [0., 1.]])
    metrics = semantic_metrics(keys, keys, torch.tensor([0, 0, 1]))
    assert metrics['event_to_text'] == 1
    assert metrics['text_to_event'] == 1
    assert metrics['margin'] == 1


def test_retention_gate_rejects_one_degraded_branch_not_hidden_by_averaging():
    from continuous_mmwave.parrad_projection_pilot import retention_gate
    old = {'stream': {'text_to_event': .8, 'event_to_text': .8, 'margin': .2},
           'commit': {'text_to_event': .8, 'event_to_text': .8, 'margin': .2},
           'action_stream': {'macro_f1': .8}}
    new = copy.deepcopy(old)
    new['stream']['text_to_event'] = .77
    assert retention_gate(old, new) is False
    new['stream']['text_to_event'] = .79
    assert retention_gate(old, new) is True
    del new['commit']['margin']
    assert retention_gate(old, new) is False


def test_checkpoint_selection_uses_dev_eligibility_and_keeps_earlier_ties():
    from continuous_mmwave.parrad_projection_pilot import choose_checkpoint
    history = [{'epoch': 0, 'score': .6, 'eligible': True},
               {'epoch': 1, 'score': .7, 'eligible': False},
               {'epoch': 2, 'score': .64, 'eligible': True},
               {'epoch': 3, 'score': .64, 'eligible': True}]
    assert choose_checkpoint(history) == 2
    assert choose_checkpoint(history[:2]) == 0


def test_training_step_changes_only_projection_and_disposable_heads():
    from continuous_mmwave.parrad_projection_pilot import ProjectionAdaptation
    torch.manual_seed(6)
    old = SemanticProjection(8, 128)
    model = ProjectionAdaptation(old, [3407, 7211, 12893], [1., 1., 1.],
                                 torch.zeros(3, 130), torch.ones(3, 130))
    before = copy.deepcopy(model.projection.state_dict())
    teacher = copy.deepcopy(model.teacher.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    x, a = torch.randn(4, 3, 8), torch.randn(4, 3, 8)
    rehearsal = {'stream': torch.randn(4, 3, 8), 'commit': torch.randn(4, 3, 8),
                 'text': torch.randn(4, 5, 8), 'mask': torch.ones(4, 5, dtype=torch.bool),
                 'labels': torch.tensor([0, 0, 1, 1])}
    loss, components = model.loss(x, a, torch.tensor([0., 1., 0., 1.]),
                                  torch.ones(4), rehearsal, consistency_weight=.1)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    assert components['semantic'].isfinite()
    assert any(not torch.equal(before[k], v) for k, v in model.projection.state_dict().items())
    assert all(torch.equal(teacher[k], v) for k, v in model.teacher.state_dict().items())
    assert all(p.grad is None for p in model.teacher.parameters())
    assert all(sum(p.numel() for p in h.parameters()) == 131 for h in model.heads)
