import numpy as np

from .geometry import geometry_verified, proxy_geometry_usable, transform_points, transform_velocities, estimate_los, load_extrinsics
from .doppler import echo_support, load_doppler


def compute_targets(pose, joints, dt, *, radial=False, geometry=None, allow_proxy=False):
    if dt <= 0:
        raise ValueError('dt must be positive')
    if radial and not (geometry_verified(geometry or {}) or (allow_proxy and proxy_geometry_usable(geometry or {}))):
        raise ValueError('BLOCKED_GEOMETRY: a validated world-to-radar transform is required')
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 3 or pose.shape[1:] != (17, 3):
        raise ValueError('Expected pose [T,17,3]')
    valid = np.isfinite(pose).all(-1) & np.any(pose != 0, axis=-1)
    clean = np.where(valid[..., None], pose, 0)
    result = {'A': clean.reshape(len(pose), 51).astype(np.float32),
              'A_mask': np.repeat(valid, 3, axis=1)}
    filtered = np.zeros_like(clean)
    filtered_valid = np.zeros_like(valid)
    for frame in range(2, len(pose)):
        filtered[frame] = np.median(clean[frame-2:frame+1], axis=0)
        filtered_valid[frame] = valid[frame-2:frame+1].all(axis=0)
    hips = [joints['left_hip'], joints['right_hip']]
    root = filtered[:, hips].mean(axis=1)
    root_valid = filtered_valid[:, hips].all(axis=1)
    relative = filtered - root[:, None]
    root_velocity = np.zeros_like(root)
    relative_velocity = np.zeros_like(relative)
    root_velocity[1:] = np.diff(root, axis=0) / dt
    relative_velocity[1:] = np.diff(relative, axis=0) / dt
    velocity_valid = np.zeros_like(filtered_valid)
    velocity_valid[1:] = filtered_valid[1:] & filtered_valid[:-1]
    root_speed_valid = np.zeros_like(root_valid)
    root_speed_valid[1:] = root_valid[1:] & root_valid[:-1]
    groups = list(joints['groups'].values())
    mask = np.column_stack([root_speed_valid] + [
        root_speed_valid & velocity_valid[:, group].all(axis=1) for group in groups])
    magnitude = np.column_stack([np.linalg.norm(root_velocity, axis=-1)] + [
        np.linalg.norm(relative_velocity[:, group], axis=-1).mean(axis=1) for group in groups])
    result.update(B=np.where(mask, magnitude, 0).astype(np.float32), B_mask=mask)
    if radial:
        radar_root = transform_points(root, geometry)
        direction, los_valid = estimate_los(radar_root)
        radial_root_velocity = np.einsum('ti,ti->t', direction, transform_velocities(root_velocity, geometry))
        radial_relative_velocity = np.einsum('ti,tji->tj', direction, transform_velocities(relative_velocity, geometry))
        values = np.column_stack([radial_root_velocity] + [
            np.abs(radial_relative_velocity[:, group]).mean(axis=1) for group in groups])
        radial_mask = mask & los_valid[:, None]
        result.update(C=np.where(radial_mask, values, 0).astype(np.float32), C_mask=radial_mask)
    return result


def build_cd_labels(pose, frames, joints, dt, extrinsics_path, *, pose_unit, doppler_path=None):
    """Formal entry point. Every metadata gate is checked before computing labels."""
    geometry = load_extrinsics(extrinsics_path)
    spec = load_doppler(doppler_path) if doppler_path is not None else None
    if pose_unit != 'm' or joints.get('verified') is not True:
        raise ValueError('BLOCKED_DATA: verified joint map and meter pose coordinates required')
    if len(pose) != len(frames):
        raise ValueError('BLOCKED_DATA: skeleton/radar frame count mismatch')
    targets = compute_targets(pose, joints, dt, radial=True, geometry=geometry)
    observed = np.array([np.isfinite(p).all(axis=1).any() for p in frames])
    result = {'C':targets['C'], 'C_mask':targets['C_mask'] & observed[:,None],
              'dt_sec':dt, 'units':'m/s', 'los_mode':'shared_root'}
    if spec is not None:
        support = echo_support(frames,spec)
        result.update(echo_weight=support['weights'], echo_frame_mask=support['frame_mask'])
        result['D_mask'] = result['C_mask'] & support['frame_mask'][:,None]
    return result


def compute_echo_weight(frames, doppler_col, units):
    if units != 'm/s' or doppler_col not in (3, 4):
        raise ValueError('BLOCKED_DOPPLER: column and m/s units must be verified')
    weights = []
    for points in frames:
        speeds = points[:, doppler_col]
        speeds = speeds[np.isfinite(speeds)]
        ratio = np.mean(np.abs(speeds) > .1) if len(speeds) else 0
        weights.append(.25 + .75 * np.clip(ratio / .25, 0, 1))
    return np.asarray(weights, dtype=np.float32)
