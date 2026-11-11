import numpy as np
import pytest
import torch

from romsl_mvp.romsl.targets import compute_targets, compute_echo_weight
from romsl_mvp.romsl.model import RadarEncoder, RegressionHead, masked_huber
from romsl_mvp.romsl.preprocess import make_windows, sample_points


JOINTS = {'left_hip': 4, 'right_hip': 1, 'groups': {
    'left_arm': [11, 12, 13], 'right_arm': [14, 15, 16],
    'left_leg': [4, 5, 6], 'right_leg': [1, 2, 3]}}


def motion():
    pose = np.zeros((20, 17, 3), dtype=np.float32)
    pose[..., 0] = 2 + np.arange(20)[:, None] * .1
    pose[:, [11, 12, 13], 0] += np.arange(20)[:, None] * .1
    return pose


def test_motion_targets_and_missing_joint():
    pose = motion()
    result = compute_targets(pose, JOINTS, .1)
    np.testing.assert_allclose(result['B'][3:, 0], 1, atol=1e-5)
    np.testing.assert_allclose(result['B'][3:, 1], 1, atol=1e-5)
    np.testing.assert_allclose(result['B'][3:, 2:], 0, atol=1e-5)
    assert not result['B_mask'][:3].any()
    pose[8, 16] = np.nan
    result = compute_targets(pose, JOINTS, .1)
    assert not result['B_mask'][8:12, 2].any()
    assert result['B_mask'][8:12, 1].all()


def test_rotation_invariance_B():
    pose = motion()
    rotation = np.array([[0,-1,0],[1,0,0],[0,0,1]], dtype=np.float32)
    first = compute_targets(pose, JOINTS, .1)
    second = compute_targets(pose @ rotation.T, JOINTS, .1)
    np.testing.assert_allclose(first['B'], second['B'], atol=1e-5)


def test_causal_filter():
    pose = motion()
    first = compute_targets(pose, JOINTS, .1)
    pose[10:] *= 5
    second = compute_targets(pose, JOINTS, .1)
    np.testing.assert_array_equal(first['B'][:10], second['B'][:10])


def test_radial_geometry_gate_and_sign():
    with pytest.raises(ValueError, match='BLOCKED_GEOMETRY'):
        compute_targets(motion(), JOINTS, .1, radial=True)
    geometry = {'verified': True, 'R': np.eye(3), 't': np.zeros(3),
                'source': 'synthetic known transform', 'translation_unit': 'm',
                'fit_recordings': ['synthetic_train'], 'validation_recordings': ['synthetic_val'],
                'mean_error_m': 0, 'p95_error_m': 0}
    result = compute_targets(motion(), JOINTS, .1, radial=True, geometry=geometry)
    np.testing.assert_allclose(result['C'][3:, :2], 1, atol=1e-5)
    pose = motion()
    pose[..., 0] = 5 - np.arange(20)[:, None] * .1
    result = compute_targets(pose, JOINTS, .1, radial=True, geometry=geometry)
    np.testing.assert_allclose(result['C'][3:, 0], -1, atol=1e-5)
    pose = motion()
    pose[:, [11,12,13], 0] = pose[:, [4], 0]
    pose[:, [11,12,13], 1] = np.arange(20)[:, None] * .1
    result = compute_targets(pose, JOINTS, .1, radial=True, geometry=geometry)
    np.testing.assert_allclose(result['C'][3:, 1], 0, atol=1e-5)


def test_echo_weight():
    points = np.zeros((4,5))
    points[:,4] = [0,.2,-.4,.05]
    with pytest.raises(ValueError, match='BLOCKED_DOPPLER'):
        compute_echo_weight([points], 4, 'unverified')
    np.testing.assert_allclose(compute_echo_weight([points], 4, 'm/s'), [1])
    partial = np.zeros((8,5))
    partial[0,4] = .2
    np.testing.assert_allclose(compute_echo_weight([partial, partial[:0]], 4, 'm/s'), [.625, .25])


def test_padding_empty_frames_and_backward():
    torch.manual_seed(42)
    model = RadarEncoder()
    model.eval()
    points = torch.randn(2,16,8,5)
    mask = torch.ones(2,16,8, dtype=torch.bool)
    mask[:,:,5:] = False
    mask[:,4] = False
    first = model(points, mask)
    points[~mask] = 10000
    second = model(points, mask)
    torch.testing.assert_close(first, second, atol=0, rtol=0)
    assert first.shape == (2,16,128) and torch.isfinite(first).all()
    head = RegressionHead(5)
    prediction = head(second)
    target = torch.randn_like(prediction)
    target_mask = torch.ones_like(prediction, dtype=torch.bool)
    target[:,4] = float('nan')
    target_mask[:,4] = False
    loss = masked_huber(prediction, target, target_mask, mask.any(-1))
    loss.backward()
    for module in (model.point_mlp, model.gru):
        gradients = [p.grad for p in module.parameters()]
        assert all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)


def test_window_quality_and_sampling():
    points = np.arange(200*5,dtype=np.float32).reshape(200,5)
    first, mask = sample_points(points, np.zeros(5), np.ones(5), seed=42)
    second, _ = sample_points(points, np.zeros(5), np.ones(5), seed=42)
    np.testing.assert_array_equal(first, second)
    assert mask.all() and len(np.unique(first[:,0])) == 128
    assert first[:,0].max() > points[127,0]
    first, mask = sample_points(points[:2], np.zeros(5), np.ones(5), seed=42)
    assert mask.sum() == 2 and (first[~mask] == 0).all()
    usable = np.ones(30, dtype=bool)
    assert make_windows(usable, usable, size=16, stride=8) == [(0,16), (8,24)]
    usable[:20] = False
    assert not make_windows(usable, usable, size=16, stride=8)
