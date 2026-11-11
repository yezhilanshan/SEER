import csv
import json
from pathlib import Path

import numpy as np

from .mmfi_io import load_recording
from .plots import write_recording_plot
from .geometry import geometry_verified


def subject_splits(splits):
    if set(splits) != {'train', 'val', 'test'} or not all(splits.values()):
        raise ValueError('Expected nonempty train, val and test splits')
    index = {}
    for split, subjects in splits.items():
        for subject in subjects:
            if subject in index:
                raise ValueError(f'{subject} appears more than once in the split')
            index[subject] = split
    return index


def gate_status(data_verified, geometry_verified, doppler_verified):
    if not data_verified:
        return dict.fromkeys('ABCD', 'BLOCKED_DATA')
    gates = dict.fromkeys('ABCD', 'READY')
    if not geometry_verified:
        gates['C'] = gates['D'] = 'BLOCKED_GEOMETRY'
    elif not doppler_verified:
        gates['D'] = 'BLOCKED_DOPPLER'
    return gates


def quantiles(values):
    finite = values[np.isfinite(values)]
    return dict(zip(('min', 'p01', 'p50', 'p99', 'max'),
                    np.quantile(finite, [0, .01, .50, .99, 1]).tolist())) if finite.size else None


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def run_audit(root, output, config, joints, geometry):
    data = config['data']
    settings = config['audit']
    scene = data['scene']
    split_index = subject_splits(data['splits'])
    actions = [f'A{i:02}' for i in range(1, 28)]
    # Each split has three displays, including the four required motion types across the grid.
    plot_actions = {'train': ('A06', 'A13', 'A17'), 'val': ('A12', 'A14', 'A18'),
                    'test': ('A06', 'A12', 'A17')}
    plot_ids = {
        f'{scene}/{data["splits"][split][0]}/{action}'
        for split, examples in plot_actions.items() for action in examples
    }
    reports = output / 'reports' / 'audit'
    plots = reports / 'plots'
    artifacts = output / 'artifacts'
    plots.mkdir(parents=True, exist_ok=False)
    artifacts.mkdir(parents=True)
    issues, rows, all_points, poses = [], [], [], []
    coverage = {split: set() for split in data['splits']}
    for subject, split in split_index.items():
        for action in actions:
            directory = root / scene / subject / action
            try:
                rec = load_recording(directory)
            except (OSError, ValueError) as exc:
                issues.append({'recording_id': f'{scene}/{subject}/{action}', 'error': str(exc)})
                continue
            counts = np.array([len(points) for points in rec.points_per_frame])
            points = np.concatenate(rec.points_per_frame)
            radar_valid = np.array([np.isfinite(frame).all() and len(frame) > 0
                                    for frame in rec.points_per_frame])
            pose_valid = np.isfinite(rec.pose).all(axis=(1, 2)) & np.any(rec.pose != 0, axis=(1, 2))
            if (radar_valid & pose_valid).any():
                coverage[split].add(action)
            else:
                issues.append({'recording_id': rec.recording_id, 'error': 'No jointly valid radar and pose frames'})
            rows.append({
                'recording_id': rec.recording_id, 'split': split,
                'ground_truth_shape': list(rec.pose.shape), 'ground_truth_dtype': str(rec.pose.dtype),
                'mmwave_frame_count': len(counts), 'mmwave_frame_dtype': 'little-endian float64',
                'mmwave_columns': 5, 'point_count_min': int(counts.min()),
                'point_count_median': float(np.median(counts)), 'point_count_max': int(counts.max()),
                'nan_count': int(np.isnan(points).sum()), 'inf_count': int(np.isinf(points).sum()),
                'empty_frames': int((counts == 0).sum()), 'invalid_pose_frames': int((~pose_valid).sum()),
                'matched_frame_count': len(rec.frame_ids), 'valid_paired_frames': int((radar_valid & pose_valid).sum()),
                'joint_map_verified': bool(joints.get('verified')),
                'geometry_status': 'VERIFIED' if geometry_verified(geometry) else 'BLOCKED_GEOMETRY',
                'doppler_column_status': 'VERIFIED' if settings.get('doppler_verified') else 'UNVERIFIED',
            })
            all_points.append(points)
            poses.append(rec.pose)
            if rec.recording_id in plot_ids:
                write_recording_plot(plots / (rec.recording_id.replace('/', '_') + '.svg'), rec, joints, data['fps'])
    for split, available in coverage.items():
        missing = sorted(set(data['probe_actions']) - available)
        if missing:
            issues.append({'split': split, 'missing_probe_actions': missing})
    layout_verified = joints.get('verified') is True and settings.get('pose_unit') == 'm'
    audit = {
        'scene': scene, 'splits': data['splits'], 'fps': data['fps'],
        'frame_mapping': 'file frame N -> ground_truth[N - 1], only after exact index and length match',
        'frame_index_reference': settings['frame_index_reference'],
        'pose_unit': settings['pose_unit'], 'pose_unit_evidence': settings.get('pose_unit_evidence'),
        'joint_map': joints, 'geometry': geometry,
        'pose_reference_frame': 'RGB camera world, as described in the MM-Fi paper',
        'radar_reference_frame': 'Published right-handed point coordinates; camera-to-radar transform unverified',
        'column_layout': ['x', 'y', 'z', 'feature3', 'feature4'],
        'doppler_col': None, 'intensity_col': None, 'doppler_units': 'unverified', 'doppler_sign': 'unverified',
        'plot_recording_ids': sorted(plot_ids.intersection(row['recording_id'] for row in rows)),
        'file_audit': 'PASS' if not issues else 'BLOCKED_DATA',
        'geometry_gate': 'PASS' if geometry_verified(geometry) else 'BLOCKED_GEOMETRY',
        'doppler_gate': 'PASS' if settings.get('doppler_verified') else 'BLOCKED_DOPPLER',
        'gates': gate_status(not issues and layout_verified, geometry_verified(geometry), settings.get('doppler_verified')),
        'training_blockers': ([] if layout_verified else ['Pose scale and candidate joint map require verification.'])
                             + (['File audit contains errors.'] if issues else []),
        'issues': issues,
        'totals': {
            'recordings': len(rows), 'expected_recordings': len(split_index) * len(actions),
            'matched_frames': sum(row['matched_frame_count'] for row in rows),
            'empty_radar_frames': sum(row['empty_frames'] for row in rows),
            'invalid_pose_frames': sum(row['invalid_pose_frames'] for row in rows),
        },
    }
    audit['totals']['empty_radar_fraction'] = (
        audit['totals']['empty_radar_frames'] / audit['totals']['matched_frames'] if rows else None
    )
    if all_points:
        points = np.concatenate(all_points)
        audit['radar_columns'] = [{
            'index': column, 'quantiles': quantiles(values), 'nan': int(np.isnan(values).sum()),
            'inf': int(np.isinf(values).sum()), 'zero_fraction': float((values == 0).mean()) if values.size else None,
        } for column, values in enumerate(points.T)]
        pose = np.concatenate(poses)
        audit['pose_xyz'] = [quantiles(pose[..., axis]) for axis in range(3)]
        audit['candidate_limb_lengths'] = {}
        audit['candidate_limb_relative_speeds'] = {}
        for name, chain in joints['groups'].items():
            length = np.linalg.norm(np.diff(pose[:, chain], axis=1), axis=-1)
            audit['candidate_limb_lengths'][name] = [quantiles(length[:, i]) for i in range(length.shape[1])]
            speeds = []
            for recording_pose in poses:
                hip = recording_pose[:, [joints['left_hip'], joints['right_hip']]].mean(axis=1)
                relative = recording_pose[:, chain] - hip[:, None, :]
                speeds.append(np.linalg.norm(np.diff(relative, axis=0), axis=-1).mean(axis=1) * data['fps'])
            audit['candidate_limb_relative_speeds'][name] = quantiles(np.concatenate(speeds))
    with (reports / 'recordings.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ['recording_id'])
        writer.writeheader()
        writer.writerows(rows)
    write_json(reports / 'data_audit.json', audit)
    write_json(artifacts / 'split.json', {'scene': scene, **data['splits']})
    write_json(artifacts / 'joints.yaml', joints)
    return audit
