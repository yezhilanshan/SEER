"""Read-only numeric audit against independent formulas and saved checkpoints."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Import the exact locked implementation that produced these checkpoints.
    sys.path.insert(0, str(args.job / 'source'))
    from romsl.experiment import extract_features, verify_lock
    from romsl.targets import compute_targets

    verify_lock(args.job)
    torch.set_num_threads(4)
    config = json.loads((args.job / 'source/configs/mvp.yaml').read_text())
    joints = json.loads((args.job / 'source/configs/joints.yaml').read_text())
    dt = 1 / config['data']['fps']
    assert dt == .1
    declared_groups = ['left_arm', 'right_arm', 'left_leg', 'right_leg']
    assert list(joints['groups']) == declared_groups
    hips = [joints['left_hip'], joints['right_hip']]
    errors, checked = [], 0
    point_sum, point_square, point_count = np.zeros(5), np.zeros(5), 0
    training_targets, training_masks = [], []
    for cache in sorted((args.job / 'cache').glob('*.npz')):
        scene, subject, action = cache.stem.split('_')
        pose = np.load(args.data_root / scene / subject / action / 'ground_truth.npy').astype(np.float64)
        valid = np.isfinite(pose).all(-1) & (pose != 0).any(-1)
        clean = np.where(valid[..., None], pose, 0)
        smooth = np.zeros_like(pose)
        smooth[2:] = np.median(np.lib.stride_tricks.sliding_window_view(clean, 3, axis=0), axis=-1)
        smooth_valid = np.zeros_like(valid)
        smooth_valid[2:] = np.lib.stride_tricks.sliding_window_view(valid, 3, axis=0).all(-1)
        center = (smooth[:, hips[0]] + smooth[:, hips[1]]) / 2
        relative = smooth - center[:, None]
        root_velocity = np.diff(center, axis=0) / dt
        relative_velocity = np.diff(relative, axis=0) / dt
        reference = np.zeros((len(pose), 5))
        reference[1:, 0] = np.linalg.norm(root_velocity, axis=-1)
        mask = np.zeros_like(reference, dtype=bool)
        hip_ok = smooth_valid[:, hips].all(-1)
        mask[1:, 0] = hip_ok[1:] & hip_ok[:-1]
        for column, name in enumerate(declared_groups, 1):
            group = joints['groups'][name]
            reference[1:, column] = np.linalg.norm(relative_velocity[:, group], axis=-1).mean(-1)
            mask[1:, column] = mask[1:, 0] & smooth_valid[1:, group].all(-1) & smooth_valid[:-1, group].all(-1)
        reference[~mask] = 0
        with np.load(cache) as data:
            assert np.array_equal(mask, data['B_mask']), cache
            difference = float(np.max(np.abs(reference - data['B'])))
            errors.append(difference)
            np.testing.assert_allclose(reference, data['B'], atol=1e-5, rtol=1e-6)
            # A regression is in position units. B is a derivative in position units / second.
            np.testing.assert_array_equal(clean.reshape(-1,51).astype(np.float32), data['A'])
            if subject in config['data']['splits']['train']:
                training_targets.append(data['B'])
                training_masks.append(data['B_mask'])
                for frame in range(1, len(pose) + 1):
                    points = np.fromfile(args.data_root / scene / subject / action / 'mmwave' / f'frame{frame:03}.bin', dtype='<f8').reshape(-1,5)
                    points = points[np.isfinite(points).all(-1)]
                    point_sum += points.sum(0)
                    point_square += (points ** 2).sum(0)
                    point_count += len(points)
            checked += 1
    values, valid = np.concatenate(training_targets), np.concatenate(training_masks)
    count = valid.sum(0)
    expected_mean = (values * valid).sum(0, dtype=np.float64) / count
    expected_std = np.sqrt(np.maximum((((values - expected_mean) ** 2) * valid).sum(0) / count, 1e-12))
    with np.load(args.job / 'artifacts/target_stats.npz') as stats:
        mean, std = stats['B_mean'], stats['B_std']
        np.testing.assert_allclose(mean, expected_mean, atol=1e-12)
        np.testing.assert_allclose(std, expected_std, atol=1e-12)
    with np.load(args.job / 'artifacts/input_stats.npz') as stats:
        np.testing.assert_allclose(stats['mean'], point_sum / point_count, atol=1e-11)
        np.testing.assert_allclose(stats['std'], np.sqrt(np.maximum(point_square / point_count - (point_sum / point_count) ** 2, 1e-12)), atol=1e-11)
    # A known translation of 0.1 coordinate units per 0.1 seconds must produce speed 1.
    synthetic = np.ones((10,17,3))
    synthetic[..., 0] += np.arange(10)[:, None] * .1
    first = compute_targets(synthetic, joints, .1)
    second = compute_targets(synthetic, joints, .2)
    np.testing.assert_allclose(first['B'][3:,0], 1, atol=1e-6)
    np.testing.assert_allclose(second['B'][3:,0], .5, atol=1e-6)
    np.testing.assert_allclose(first['B'][3:,1:], 0, atol=1e-6)
    expected = json.loads((args.job / 'test_results.json').read_text())
    replays = {}
    for exp in ('A','B'):
        data = extract_features(args.job, exp, 'test', torch.device('cpu'))
        head = torch.nn.Linear(128,5)
        state = torch.load(args.job / 'runs' / exp / 'probe_motion_best.pt', map_location='cpu', weights_only=True)
        head.load_state_dict(state['head'])
        head.eval()
        with torch.inference_mode():
            prediction = head(data['features']).numpy()
        # Explicitly invert BOTH prediction and target before subtracting.
        physical_prediction = prediction * std.astype(np.float32) + mean.astype(np.float32)
        physical_target = data['target'].numpy() * std.astype(np.float32) + mean.astype(np.float32)
        error = np.abs(physical_prediction - physical_target)
        recording_sums, recording_counts = {}, {}
        for index, row in enumerate(data['rows']):
            key = row['recording_id']
            mask = data['mask'][index].numpy()
            recording_sums[key] = recording_sums.get(key, np.zeros(5)) + (error[index] * mask).sum(0)
            recording_counts[key] = recording_counts.get(key, np.zeros(5)) + mask.sum(0)
        recording = {key: recording_sums[key] / recording_counts[key] for key in recording_sums}
        subjects = sorted({key.split('/')[1] for key in recording})
        subject_values = [np.mean([v for k,v in recording.items() if k.split('/')[1] == subject], axis=0) for subject in subjects]
        components = np.mean(subject_values, axis=0)
        original = expected[exp]['motion']['per_component']
        np.testing.assert_allclose(components, original, atol=2e-6, rtol=1e-5)
        replays[exp] = {'original': original, 'independent': components.tolist(),
                        'mean_mae': float(components.mean()),
                        'max_component_delta': float(np.max(np.abs(components - original)))}
    report = {
        'status': 'PASS_COMPUTATION', 'audited_recordings': checked,
        'B_formula_max_absolute_error': max(errors), 'B_dimension_order': ['root_speed'] + declared_groups,
        'dt_sec': dt, 'coordinates_scaled_before_differentiation': False,
        'input_stats_recomputed_training_points': point_count,
        'known_motion_test': {'displacement_per_frame': .1, 'dt_sec': .1, 'speed': 1., 'speed_at_dt_point2': .5},
        'MAE_dimension': 'coordinate_unit / second',
        'MAE_if_pose_unit_is_m': 'm/s',
        'absolute_pose_unit_status': 'Nominal meters, supported by scale and community comment; authoritative calibration not supplied',
        'mean': mean.tolist(), 'std': std.tolist(), 'target_stats_subjects': config['data']['splits']['train'],
        'independent_MAE_replays': replays,
        'role': 'Audit replay of existing checkpoints, with no optimization or selection',
    }
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
