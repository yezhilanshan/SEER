"""Mechanism contracts independent of any dataset annotation."""
import math

import numpy as np
import pytest
import torch

from continuous_mmwave.event_gating.mvp.model import (
    ActionHead, EvidenceNet, MVPConfig, rollout, training_feature_inputs,
)


def heads(config, current=1.5):
    action, evidence = ActionHead(config), EvidenceNet(config, use_state=True)
    with torch.no_grad():
        for p in evidence.parameters():
            p.zero_()
        evidence.net[-1].bias.fill_(math.log(current / (config.current_max-current)))
    return action, evidence


def test_lif_exact_decay_variable_dt_and_if_seconds_scaling():
    c = MVPConfig(feature_dim=2, hidden_dim=3, num_classes=2)
    a, e = heads(c)
    x = np.zeros((4, 2), np.float32)
    dt = [0.1, 0.2, 0.4, 0.3]
    lif = rollout(x, a, e, c, threshold=100, tau=0.5, dt_s=dt)
    assert lif['trace']['voltage'] == pytest.approx(
        [0, 1.5*(1-math.exp(-0.4)), 1.5*(1-math.exp(-1.2)), 1.5*(1-math.exp(-1.8))])
    inf = rollout(x, a, e, c, mechanism='if', threshold=100, dt_s=dt)
    assert inf['trace']['voltage'] == pytest.approx([0, .3, .9, 1.35])


def test_hard_reset_frame_ownership_and_eof_not_fire():
    c = MVPConfig(feature_dim=2, hidden_dim=3, num_classes=2, frame_period_s=1)
    a, e = heads(c)
    r = rollout(np.arange(12, dtype=np.float32).reshape(6, 2), a, e, c,
                mechanism='if', threshold=2)
    assert r['trace']['fire'].tolist() == [False, False, True, False, True, False]
    assert r['trace']['voltage'] == pytest.approx([0, 1.5, 3, 1.5, 3, 1.5])
    assert [(v['start_frame'], v['stop_frame'], v['closure']) for v in r['events']] == [
        (0, 2, 'fire'), (2, 4, 'fire'), (4, 6, 'eof')]
    assert r['trace']['state_distance'][3] == pytest.approx(math.sqrt(8))


def test_direct_is_upward_crossing_including_initial_high_suppression():
    c = MVPConfig(feature_dim=1, hidden_dim=2, num_classes=2)
    a, e = heads(c)
    r = rollout(np.ones((8, 1)), a, e, c, mechanism='direct', threshold=1)
    assert not r['trace']['fire'].any()
    assert len(r['events']) == 1


def test_direct_low_high_plateau_fires_once_and_rearms_only_after_low():
    c = MVPConfig(feature_dim=1, hidden_dim=2, num_classes=2)
    a, e = heads(c)
    with torch.no_grad():
        e.net[0].weight[:, 0] = torch.tensor([1., -1.])
        e.net[1].weight.fill_(1.)
        e.net[-1].weight[0, 0] = 4.
        e.net[-1].bias.fill_(-2.)
    x = np.array([[-1], [1], [1], [1], [-1], [1]], np.float32)
    r = rollout(x, a, e, c, mechanism='direct', threshold=1.5)
    assert r['trace']['fire'].tolist() == [False, True, False, False, False, True]
    assert [(v['start_frame'], v['stop_frame']) for v in r['events']] == [(0, 1), (1, 5), (5, 6)]


def test_optional_minimum_duration_counts_seconds_without_default_refractory():
    c = MVPConfig(feature_dim=1, hidden_dim=2, num_classes=2,
                  frame_period_s=.5, minimum_duration_s=1.5)
    a, e = heads(c)
    r = rollout(np.zeros((7, 1)), a, e, c, mechanism='if', threshold=.1)
    assert r['trace']['fire'].tolist() == [False, False, False, True, False, False, True]


def test_state_features_read_before_update_and_oracle_reset_after_read():
    c = MVPConfig(feature_dim=1, hidden_dim=2, num_classes=2,
                  frame_period_s=math.log(2), state_tau_s=1)
    x = torch.tensor([[0.], [2.], [4.], [8.]], requires_grad=True)
    u = training_feature_inputs(x, c, boundaries=[2])
    assert u.detach().numpy() == pytest.approx(np.array([
        [0, 0, 0, 0], [2, 2, 0, 2], [4, 2, 1, 3], [8, 4, 4, 4]], dtype=float))
    u.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_training_autonomous_features_match_rollout_and_oracle_never_forces_fire():
    c = MVPConfig(feature_dim=1, hidden_dim=2, num_classes=2, frame_period_s=.5)
    a, e = heads(c)
    x = np.array([[0.], [1.], [4.], [8.], [2.]], np.float32)
    autonomous = rollout(x, a, e, c, mechanism='if', threshold=1)
    inputs = training_feature_inputs(x, c, fire=autonomous['trace']['fire'])
    np.testing.assert_allclose(inputs[:, -1].abs().numpy(), autonomous['trace']['state_distance'])
    oracle = rollout(x, a, e, c, threshold=100, teacher_boundaries=[2])
    ideal_inputs = training_feature_inputs(x, c, boundaries=[2])
    np.testing.assert_allclose(ideal_inputs[:, -1].abs().numpy(), oracle['trace']['state_distance'])
    assert not oracle['trace']['fire'].any()
    assert oracle['trace']['state_distance'][3] == pytest.approx(4.)


def test_oracle_diagnostic_uses_the_same_state_history_as_warm_start_even_with_fires():
    c=MVPConfig(feature_dim=1,hidden_dim=2,num_classes=2,frame_period_s=.5)
    a,e=heads(c)
    x=np.array([[0.],[1.],[4.],[8.],[2.]],np.float32)
    oracle=rollout(x,a,e,c,mechanism='if',threshold=.1,teacher_boundaries=[3])
    inputs=training_feature_inputs(x,c,boundaries=[3])
    assert oracle['trace']['fire'].any()
    np.testing.assert_allclose(inputs[:,-1].abs().numpy(),oracle['trace']['state_distance'])


def test_local_masks_state_slots_but_preserves_current_and_delta():
    c = MVPConfig(feature_dim=2, hidden_dim=4, num_classes=3)
    local, state = EvidenceNet(c, False), EvidenceNet(c, True)
    state.load_state_dict(local.state_dict())
    assert sum(p.numel() for p in local.parameters()) == sum(p.numel() for p in state.parameters())
    h, prev = torch.tensor([[1., 2.]]), torch.tensor([[0., 1.]])
    assert torch.equal(local(h, prev, torch.zeros_like(h)), local(h, prev, h*20))
    u = torch.cat([h, h-prev, torch.zeros_like(h), torch.zeros_like(h)], -1)
    assert torch.equal(local(h, prev, h*20), local.input_logits(u))
    assert local(h[0], prev[0], h[0]).shape == torch.Size([])


def test_readout_means_probabilities_and_is_shared_across_mechanisms():
    c = MVPConfig(feature_dim=1, hidden_dim=2, num_classes=2)
    a, e = heads(c)
    with torch.no_grad():
        a.linear.weight.copy_(torch.tensor([[1.], [0.]]))
        a.linear.bias.zero_()
    x = np.array([[-1.], [-1.], [3.]], np.float32)
    expected = np.array([(2/(1+math.exp(1))+1/(1+math.exp(-3)))/3, 0])
    expected[1] = 1-expected[0]
    for mechanism in ['direct', 'if', 'lif']:
        r = rollout(x, a, e, c, mechanism=mechanism, threshold=100)
        assert r['events'][0]['probabilities'] == pytest.approx(expected)
        assert r['events'][0]['class_id'] == 1


def test_future_mutation_and_prefix_do_not_change_past_trace():
    torch.manual_seed(4)
    c = MVPConfig(feature_dim=3, hidden_dim=4, num_classes=2)
    a, e = ActionHead(c), EvidenceNet(c, True)
    x = np.random.default_rng(2).normal(size=(12, 3)).astype(np.float32)
    full = rollout(x, a, e, c)
    altered = x.copy(); altered[7:] += 100
    other = rollout(altered, a, e, c)
    prefix = rollout(x[:7], a, e, c)
    for key in full['trace']:
        np.testing.assert_allclose(full['trace'][key][:7], other['trace'][key][:7])
        np.testing.assert_allclose(full['trace'][key][:7], prefix['trace'][key])


def test_empty_and_singleton_have_explicit_eof_and_no_fire():
    c = MVPConfig(feature_dim=2, hidden_dim=3, num_classes=2)
    a, e = heads(c)
    empty = rollout(np.empty((0, 2)), a, e, c)
    assert empty['events'] == []
    assert empty['trace']['probabilities'].shape == (0, 2)
    one = rollout(np.zeros((1, 2)), a, e, c, threshold=.001)
    assert one['trace']['fire'].tolist() == [False]
    assert one['events'][0]['closure'] == 'eof'


def test_unknown_comes_only_from_confidence_and_optional_setting():
    c = MVPConfig(feature_dim=1, hidden_dim=2, num_classes=2, unknown_confidence=.8)
    a, e = heads(c)
    with torch.no_grad():
        a.linear.weight.zero_(); a.linear.bias.zero_()
    r = rollout(np.zeros((3, 1)), a, e, c, threshold=100)
    assert r['events'][0]['class_id'] == -100
    assert r['events'][0]['confidence'] == pytest.approx(.5)
    with pytest.raises(TypeError):
        rollout(np.zeros((3, 1)), a, e, c, labels=[0, 1, 1])


@pytest.mark.parametrize('kwargs', [{'tau': 0}, {'threshold': 0}, {'dt_s': [1, -1]},
                                    {'mechanism': 'adaptive'}, {'teacher_boundaries': [0]}])
def test_rejects_invalid_dynamics_and_noninternal_teacher_boundaries(kwargs):
    c = MVPConfig(feature_dim=2, hidden_dim=3, num_classes=2)
    a, e = heads(c)
    with pytest.raises(ValueError):
        rollout(np.zeros((2, 2)), a, e, c, **kwargs)
