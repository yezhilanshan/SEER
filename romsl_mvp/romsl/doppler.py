"""Audited Doppler conversion and heuristic echo-support diagnostics."""
import numpy as np

from .geometry import read_config


def load_doppler(path):
    try:
        spec = read_config(path)
        if spec.get('verified') is not True or spec.get('sign_verified') is not True:
            raise ValueError('field and sign verification required')
        if type(spec.get('doppler_col')) is not int or spec['doppler_col'] not in (3,4):
            raise ValueError('doppler_col must be zero-based 3 or 4')
        if spec.get('units') not in ('m/s','cm/s','mm/s'):
            raise ValueError('unsupported velocity units')
        if spec.get('positive_direction') not in ('away_from_radar','toward_radar'):
            raise ValueError('explicit measured positive direction required')
        if not all(spec.get(key) for key in ('source','sign_verification_source','verified_by')):
            raise ValueError('field definition and sign verification provenance required')
        return spec
    except (ValueError, TypeError, AttributeError, OSError) as exc:
        raise ValueError(f'BLOCKED_DOPPLER: {exc}') from exc


def read_doppler(points, spec):
    if spec.get('verified') is not True or spec.get('sign_verified') is not True:
        raise ValueError('BLOCKED_DOPPLER: reviewed fields and sign required')
    data = np.asarray(points, dtype=float)
    if data.ndim != 2 or data.shape[1] != 5:
        raise ValueError('Expected unnormalized raw [N,5] points')
    factors = {'m/s':1., 'cm/s':.01, 'mm/s':.001}
    if spec.get('units') not in factors or spec.get('doppler_col') not in (3,4):
        raise ValueError('BLOCKED_DOPPLER: invalid unit or column')
    signs = {'away_from_radar':1., 'toward_radar':-1.}
    if spec.get('positive_direction') not in signs:
        raise ValueError('BLOCKED_DOPPLER: unknown sign')
    valid = np.isfinite(data).all(axis=1)
    velocity = data[:, spec['doppler_col']] * factors[spec['units']] * signs[spec['positive_direction']]
    return np.where(valid,velocity,0), valid


def verify_sign(converted_velocity, known_radial_velocity, *, deadzone=.1, minimum_agreement=.9):
    measured = np.asarray(converted_velocity,dtype=float)
    reference = np.asarray(known_radial_velocity,dtype=float)
    if measured.shape != reference.shape or deadzone < 0 or not 0 <= minimum_agreement <= 1:
        raise ValueError('Invalid paired known-motion observations')
    valid = np.isfinite(measured) & np.isfinite(reference) & (np.abs(measured)>deadzone) & (np.abs(reference)>deadzone)
    count = int(valid.sum())
    agreement = float((np.sign(measured[valid]) == np.sign(reference[valid])).mean()) if count else None
    both_directions = bool((reference[valid] > 0).any() and (reference[valid] < 0).any())
    return {'usable_pairs':count,'sign_agreement':agreement, 'both_directions_observed':both_directions,
            'passed':bool(both_directions and agreement >= minimum_agreement),
            'reference_requirement':'paired known-motion calibration observations'}


def echo_support(frames, spec):
    weights, ratios, valid_frames, counts = [], [], [], []
    for points in frames:
        speed, valid = read_doppler(points,spec)
        speed = speed[valid]
        ratio = float((np.abs(speed)>.1).mean()) if len(speed) else 0.
        ratios.append(ratio)
        weights.append(.25+.75*np.clip(ratio/.25,0,1))
        valid_frames.append(bool(len(speed)))
        counts.append(len(speed))
    return {'weights':np.asarray(weights,dtype=np.float32),'ratios':np.asarray(ratios),
            'frame_mask':np.asarray(valid_frames,dtype=bool),'valid_point_counts':np.asarray(counts)}


def weight_statistics(support):
    mask = support['frame_mask']
    weights = support['weights'][mask]
    result = {'valid_frames':int(mask.sum()),'empty_or_invalid_frames':int((~mask).sum()),
              'weight_formula':'.25 + .75 * clip(dynamic_ratio/.25,0,1)',
              'dynamic_threshold_mps':.1, 'diagnostic_std_floor':.001,
              'diagnostic_near_one_fraction_limit':.95}
    if not len(weights):
        return {**result,'status':'NO_VALID_FRAMES','fraction_near_one':None}
    fraction = float((weights>=.99).mean())
    std = float(weights.std())
    return {**result,'status':'LOW_DIAGNOSTIC_VARIATION' if std < .001 or fraction >= .95 else 'VARIABLE',
            'mean':float(weights.mean()),'std':std,'min':float(weights.min()),'max':float(weights.max()),
            'p05_p50_p95':np.quantile(weights,[.05,.5,.95]).tolist(),
            'fraction_near_one':fraction,
            'effective_weight_sum':float(weights.sum())}
