import numpy as np
import pytest

from continuous_mmwave.radar_text.diagnostics import (
    classification_metrics, steady_mask, smooth_predictions, action_instances,
    attribute_instances, match_controls,
)


def test_metrics_keep_absent_classes_and_zero_division():
    r = classification_metrics(np.array([[2, 1, 0], [0, 0, 0], [0, 1, 2]]))
    assert r['frame_accuracy'] == pytest.approx(4 / 6)
    assert r['per_class'][1]['f1'] == 0
    assert r['macro_f1'] == pytest.approx((.8 + 0 + .8) / 3)
    assert r['per_class'][1]['support'] == 0


def test_boundary_mask_uses_full_timestamp_midpoint_and_unknown():
    times = np.arange(8) * .2
    labels = np.array([0, 0, 0, 1, 1, -100, 2, 2])
    mask = steady_mask(labels, times, .3)
    assert mask[0] and not mask[2] and not mask[3] and not mask[5]
    assert not mask[4] and not mask[6]


def test_smoothing_uses_seconds_not_fixed_frame_count_and_never_gt_count():
    times = np.arange(20) * .1
    labels = np.array([0] * 6 + [1] + [0] * 6 + [2] * 7)
    assert smooth_predictions(labels, times, .3, .3) == [0, 2]
    assert smooth_predictions(labels, times, 0, 0) == [0, 1, 0, 2]
    assert smooth_predictions(np.array([1, 2]), np.array([0., .1]), 0, 1.) == []


def test_instances_use_full_run_duration_and_mark_both_truncations():
    times = np.arange(10) * .1
    rows = action_instances(np.array([0] * 5 + [1] * 5), times, 2, 8)
    assert len(rows) == 2
    assert rows[0]['duration'] == pytest.approx(.3)
    assert rows[0]['full_duration'] == pytest.approx(.5)
    assert rows[0]['truncation'] == 'left'
    assert rows[1]['truncation'] == 'right'
    assert rows[0]['relative_start'] == 0
    assert rows[1]['relative_end'] == pytest.approx(1.)
    both = action_instances(np.zeros(10, dtype=int), times, 2, 8)[0]
    assert both['truncation'] == 'both'


def test_repeated_action_coverage_does_not_invent_temporal_alignment():
    rows = [dict(class_id=x) for x in [0, 1, 0]]
    result = attribute_instances(rows, [0, 1])
    assert sum(r['missed'] for r in result) == 1
    assert result[0]['identity_ambiguous'] and result[2]['identity_ambiguous']
    assert all(r['temporal_substitution'] == 'unresolved' for r in result)


def test_mismatch_matches_sensor_duration_and_different_sequence():
    rows = [dict(sensor='a', duration=12., actions=[0]),
            dict(sensor='b', duration=12., actions=[1]),
            dict(sensor='a', duration=8., actions=[1]),
            dict(sensor='a', duration=12.05, actions=[1])]
    assert match_controls(rows, 0, .09) == 3
    assert match_controls(rows, 2, .09) is None


def test_frozen_features_match_original_hidden_without_mutating_weights():
    import torch
    from continuous_mmwave.radar_text.config import RadarTextConfig
    from continuous_mmwave.radar_text.model import RadarEncoder
    from continuous_mmwave.radar_text.diagnostic_models import frozen_features
    torch.set_num_threads(1)
    encoder = RadarEncoder(RadarTextConfig()).eval()
    rd, md, lengths = torch.randn(1, 5, 1, 16, 16), torch.randn(1, 5, 16), torch.tensor([5])
    expected = encoder(rd, md, lengths)[0]
    cnn, gru = frozen_features(encoder, rd, md, lengths)
    assert cnn.shape == (1, 5, 256)
    torch.testing.assert_close(gru, expected)
    assert not gru.requires_grad and not cnn.requires_grad
    assert not encoder.gru._forward_pre_hooks


def test_language_prior_has_only_fixed_prefix_gradients_and_ignores_radar():
    import torch
    from continuous_mmwave.radar_text.config import RadarTextConfig
    from continuous_mmwave.radar_text.runner import make_tiny_model
    from continuous_mmwave.radar_text.diagnostic_models import make_language_prior
    torch.set_num_threads(1)
    model = make_language_prior(make_tiny_model(RadarTextConfig())).eval()
    rd, md, lengths = torch.randn(1, 4, 1, 16, 16), torch.randn(1, 4, 16), torch.tensor([4])
    first = model.prepare_inputs(rd, md, lengths)['inputs_embeds']
    second = model.prepare_inputs(rd * 99, md - 33, lengths)['inputs_embeds']
    torch.testing.assert_close(first, second)
    model(rd, md, lengths, ['The person walks.']).loss.backward()
    names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert names == ['encoder.prefix']
    assert model.encoder.prefix.grad.abs().sum() > 0


def test_real_order_pairs_exclude_synchronized_views_and_overlapping_crops():
    from continuous_mmwave.radar_text.diagnostic_controls import real_order_pairs
    rows = [dict(subject='a',session='s',sensor='60',recording=0,start=0,stop=100,actions=[0,1]),
            dict(subject='a',session='s',sensor='77',recording=1,start=0,stop=101,actions=[0,1]),
            dict(subject='a',session='s',sensor='60',recording=0,start=50,stop=150,actions=[1,0]),
            dict(subject='a',session='s',sensor='60',recording=0,start=200,stop=300,actions=[1,0])]
    pairs, count = real_order_pairs(rows)
    assert count == 3 and pairs == [(0,3)]
