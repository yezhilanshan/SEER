"""Verified full-recording native-grid inputs and partial boundary supervision."""
import json
from pathlib import Path
import numpy as np
from ..data import locked_path, validate_records, _finite_array


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_records(settings):
    """Load only continuous_hidden; never replay, crop, or open oracle features.

    Gaussian targets have sigma .2 seconds and positive support out to 3 sigma.
    Unknown frames in that support are weak transition supervision, explicitly
    counted; unknown frames elsewhere have no evidence or class supervision.
    """
    settings = _json(settings) if isinstance(settings, (str, Path)) else settings
    source = settings.get('source_config')
    base = _json(source) if source else settings
    locks = base['locks']
    if source and 'locks' in settings and settings['locks'] != locks:
        raise ValueError('source lock conflict')
    paths = {name: locked_path(locks[name]) for name in
             ('feature_manifest', 'feature_verification', 'radar_checkpoint', 'radar_normalizers')}
    records = _json(paths['feature_manifest'])
    validate_records(records)  # role rejection before any signal value access
    verification = _json(paths['feature_verification'])
    if (verification.get('passed') is not True
            or verification.get('test_signal_values_read') is not False
            or verification.get('source_checkpoint_sha256') != locks['radar_checkpoint']['sha256']
            or verification.get('source_normalizers_sha256') != locks['radar_normalizers']['sha256']
            or verification.get('recordings', len(records)) != len(records)
            or verification.get('frames', sum(r['frames'] for r in records)) != sum(r['frames'] for r in records)):
        raise ValueError('feature/front-end provenance mismatch')
    period = settings.get('mvp', {}).get('frame_period_s',
               settings.get('data', {}).get('frame_period_s', settings.get('gate', {}).get('frame_period_s', 1/15)))
    if not np.isfinite(period) or period <= 0:
        raise ValueError('positive finite frame period required')
    pending = []
    for row in records:
        frames = row['frames']
        if not np.isclose(row.get('frame_period_s',period),period,rtol=0,atol=1e-12):
            raise ValueError('source time grid differs from configured frame period')
        for name in ('source_frames', 'frame_indices'):
            if name in row and not np.array_equal(row[name], np.arange(frames)):
                raise ValueError('native frame grid must be full 1:1')
        if row.get('frame_stride', 1) != 1 or row.get('frame_start', 0) != 0:
            raise ValueError('native frame grid must be full 1:1')
        for name, lock_name in [('source_checkpoint_sha256','radar_checkpoint'),
                                ('source_normalizers_sha256','radar_normalizers')]:
            if name in row and row[name] != locks[lock_name]['sha256']:
                raise ValueError('record provenance conflict')
        root = Path(row['artifact_dir'])
        if not root.is_absolute(): root = paths['feature_manifest'].parent/root
        path = (root/'continuous_hidden.npy').resolve()
        if 'continuous_path' in row:
            alternate = Path(row['continuous_path'])
            if not alternate.is_absolute(): alternate = paths['feature_manifest'].parent/alternate
            if alternate.resolve() != path: raise ValueError('continuous source path conflict')
        info = row['artifacts']['continuous_hidden.npy']
        lock = dict(path=str(path), sha256=info['sha256'])
        pending.append((row, lock))
    output = []
    for row, lock in pending:
        x = _finite_array(locked_path(lock), (row['frames'], 128))
        labels = np.full(row['frames'], -100, dtype=np.int64)
        edges = set()
        for instance in row['instances']:
            labels[instance['start']:instance['stop']] = instance['label']
            edges.update((instance['start'], instance['stop']))
        boundaries = sorted(b for b in edges if 0 < b < row['frames'])
        known = labels != -100
        distance = np.full(row['frames'], np.inf)
        for b in boundaries:
            distance = np.minimum(distance, np.abs(np.arange(row['frames'])-b)*period)
        target = np.exp(-.5*(distance/.2)**2).astype(np.float32)
        positive = distance <= .6 + 1e-12
        target[~positive] = 0
        weak = positive & ~known
        # Recording start/EOF are censored edges: never teach them as negatives.
        all_edge_distance = distance.copy()
        for b in edges:
            all_edge_distance = np.minimum(all_edge_distance, np.abs(np.arange(row['frames'])-b)*period)
        negative = known & (all_edge_distance > .6 + 1e-12)
        output.append(dict(row, x=x, labels=labels, known_frames=known,
                           source_frames=np.arange(row['frames']), boundaries=boundaries,
                           boundary_target=target, boundary_mask=negative | positive,
                           weak_transition_frames=weak, frame_period_s=float(period),
                           time_basis='nominal', time_uncertainty='Configured nominal frame grid; hardware timestamps unavailable',
                           supervision=dict(gaussian_sigma_s=.2, positive_radius_s=.6,
                                            weak_unknown_transition_frames=int(weak.sum()),
                                            known_negative_frames=int(negative.sum()),
                                            masked_unknown_frames=int((~known & ~positive).sum())),
                           provenance=dict(source_locks=locks, continuous_hidden=lock,
                                           feature_kind='continuous_hidden', native_grid='full_1_to_1',
                                           test_signal_values_read=False)))
    for _, lock in pending: locked_path(lock)
    for name in paths: locked_path(locks[name])
    return output


def split_fit_calibration(records):
    """Hold out lexicographic-last train recording per subject, never subject7."""
    validate_records(records)
    fit, calibration = [], []
    validation = sorted((r for r in records if r['role']=='validation'), key=lambda r:r['file'])
    train = [r for r in records if r['role']=='train']
    if not train or not validation: raise ValueError('train and subject7 validation required')
    for subject in sorted({r['subject'] for r in train}):
        rows = sorted((r for r in train if r['subject']==subject), key=lambda r:r['file'])
        if len(rows)<2: raise ValueError('at least two recordings per train subject required')
        fit.extend(rows[:-1]); calibration.append(rows[-1])
    return fit, calibration, validation
