"""Validated world-to-radar geometry, with lengths expressed in meters."""
import json
from pathlib import Path

import numpy as np


def read_config(path):
    text = Path(path).read_text(encoding='utf-8')
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError('YAML input requires PyYAML. JSON is also supported.') from exc
        try:
            return yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ValueError(f'Invalid YAML: {exc}') from exc


def _rigid(rotation, translation):
    return (rotation.shape == (3,3) and translation.shape == (3,)
            and np.isfinite(rotation).all() and np.isfinite(translation).all()
            and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5, rtol=0)
            and np.isclose(np.linalg.det(rotation), 1, atol=1e-5, rtol=0))


def geometry_verified(geometry):
    """Retain the original in-memory contract used by synthetic target callers."""
    try:
        rotation = np.asarray(geometry.get('R'), dtype=float)
        translation = np.asarray(geometry.get('t'), dtype=float)
        errors = np.asarray([geometry.get('mean_error_m'), geometry.get('p95_error_m')], dtype=float)
        fit = set(geometry.get('fit_recordings', []))
        validation = set(geometry.get('validation_recordings', []))
        return bool(geometry.get('verified') is True and _rigid(rotation, translation)
                    and geometry.get('source') and geometry.get('translation_unit') == 'm'
                    and validation and not fit.intersection(validation)
                    and np.isfinite(errors).all() and 0 <= errors[0] <= errors[1])
    except (TypeError, ValueError, AttributeError):
        return False


def load_extrinsics(path):
    try:
        value = read_config(path)
        if not isinstance(value, dict) or value.get('verified') is not True:
            raise ValueError('verified must be true after review')
        if value.get('from_frame') != 'world' or value.get('to_frame') != 'radar':
            raise ValueError('explicit world -> radar direction required')
        if value.get('world_unit') != 'm' or not value.get('verified_by'):
            raise ValueError('meter world coordinates and reviewer required')
        if not value.get('world_axes') or not value.get('radar_axes'):
            raise ValueError('world and radar axis definitions required')
        factor = {'m':1., 'cm':.01, 'mm':.001}.get(value.get('translation_unit'))
        if factor is None:
            raise ValueError('unsupported translation unit')
        if 'T_world_to_radar' in value:
            if 'R' in value or 't' in value:
                raise ValueError('provide R/t or T_world_to_radar, avoiding ambiguous duplicates')
            matrix = np.asarray(value['T_world_to_radar'], dtype=float)
            if matrix.shape != (4,4) or not np.allclose(matrix[3], [0,0,0,1], atol=1e-8, rtol=0):
                raise ValueError('invalid homogeneous matrix')
            rotation, translation = matrix[:3,:3], matrix[:3,3] * factor
        else:
            rotation = np.asarray(value['R'], dtype=float)
            translation = np.asarray(value['t'], dtype=float) * factor
        if not _rigid(rotation, translation):
            raise ValueError('rotation must be SO(3), with finite translation')
        origin = np.asarray(value['radar_origin_world_m'], dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all() or not np.allclose(
                origin, -rotation.T @ translation, atol=1e-5, rtol=0):
            raise ValueError('declared radar origin disagrees with R/t')
        result = {**value, 'R':rotation, 't':translation, 'translation_unit':'m'}
        if not geometry_verified(result):
            raise ValueError('source and independent validation error evidence required')
        return result
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise ValueError(f'BLOCKED_GEOMETRY: {exc}') from exc


def proxy_geometry_usable(geometry):
    try:
        rotation=np.asarray(geometry['R'],dtype=float)
        translation=np.asarray(geometry['t'],dtype=float)
        origin=np.asarray(geometry['radar_origin_world_m'],dtype=float)
        return bool(geometry.get('status')=='PROXY_GEOMETRY_ONLY'
                    and geometry.get('verified') is False and geometry.get('translation_unit')=='m'
                    and geometry.get('source') and _rigid(rotation,translation)
                    and origin.shape==(3,) and np.isfinite(origin).all()
                    and np.allclose(origin,-rotation.T@translation,atol=1e-6))
    except (ValueError,TypeError,KeyError):
        return False


def transform_points(points, geometry):
    return np.asarray(points, dtype=float) @ np.asarray(geometry['R']).T + np.asarray(geometry['t'])


def transform_velocities(velocity, geometry):
    # Translation acts on positions, while velocity only changes orientation.
    return np.asarray(velocity, dtype=float) @ np.asarray(geometry['R']).T


def estimate_los(radar_positions, epsilon=1e-6):
    positions = np.asarray(radar_positions, dtype=float)
    if positions.shape[-1] != 3 or epsilon <= 0:
        raise ValueError('Expected [...,3] positions and a positive distance floor')
    distance = np.linalg.norm(positions, axis=-1)
    valid = np.isfinite(positions).all(-1) & np.isfinite(distance) & (distance > epsilon)
    direction = np.zeros_like(positions)
    np.divide(positions, distance[...,None], out=direction, where=valid[...,None])
    return direction, valid


def project_radial(radar_positions, radar_velocity):
    direction, valid = estimate_los(radar_positions)
    velocity = np.asarray(radar_velocity, dtype=float)
    if velocity.shape != direction.shape:
        raise ValueError('Position and velocity shapes must match')
    valid &= np.isfinite(velocity).all(-1)
    speed = np.einsum('...i,...i->...', direction, np.where(valid[...,None],velocity,0))
    return np.where(valid, speed, 0), valid
