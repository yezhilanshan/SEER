"""Source-locked HARrad preparation; reconstructed clips replay the frozen GRU."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def locked_path(lock):
    path = Path(lock['path'])
    if not path.is_file():
        raise FileNotFoundError(f'required asset not available: {path}')
    if sha256(path) != lock['sha256']:
        raise ValueError(f'SHA256 mismatch: {path}')
    return path


def validate_records(records):
    roles = {**{f'subject{i}': 'train' for i in range(1, 7)}, 'subject7': 'validation'}
    files, identities = set(), set()
    if not records:
        raise ValueError('empty feature manifest')
    for record in records:
        if record['role'] == 'test':
            raise ValueError('test signals are forbidden in this first protocol')
        if roles.get(record['subject']) != record['role']:
            raise ValueError('subject split mismatch')
        if record['file'] in files or type(record['frames']) is not int or record['frames'] < 1:
            raise ValueError('duplicate file or invalid frame count')
        files.add(record['file'])
        previous = 0
        for instance in record['instances']:
            s, e, label, iid = (instance[k] for k in ('start', 'stop', 'label', 'instance_id'))
            if (any(type(v) is not int for v in (s, e, label, iid))
                    or not previous <= s < e <= record['frames'] or not 0 <= label < 12
                    or (record['file'], iid) in identities
                    or instance.get('subject', record['subject']) != record['subject']
                    or instance.get('file', record['file']) != record['file']):
                raise ValueError('invalid instance interval, identity or class')
            identities.add((record['file'], iid)); previous = e


def sample_sequences(records, seed=42, stride=3, max_events=6, max_frames=900):
    validate_records(records)
    if min(stride, max_events, max_frames) < 1:
        raise ValueError('positive sequence sampling limits required')
    rng = np.random.default_rng(seed)
    samples = []
    for record in sorted(records, key=lambda r:r['file']):
        instances = record['instances']
        for start in range(0, len(instances), stride):
            available = min(max_events, len(instances)-start)
            sizes = [1]
            if available >= 2:
                sizes.append(int(rng.integers(2, available+1)))
            for size in sizes:
                chosen = instances[start:start+size]
                while len(chosen)>1 and sum(r['stop']-r['start'] for r in chosen)>max_frames:
                    chosen = chosen[:-1]
                if sum(r['stop']-r['start'] for r in chosen)>max_frames:
                    raise ValueError('an instance exceeds max_sequence_frames; refusing silent truncation')
                if size>1 and len(chosen)==1:
                    continue
                identity = record['file']+':'+','.join(str(r['instance_id']) for r in chosen)
                samples.append(dict(sample_id=hashlib.sha256(identity.encode()).hexdigest()[:24],
                                    file=record['file'], subject=record['subject'], role=record['role'],
                                    instances=chosen))
    return samples


@torch.no_grad()
def construct_sequence(frame_features, instances, temporal):
    indices = np.concatenate([np.arange(r['start'], r['stop']) for r in instances])
    joined = np.array(frame_features[indices], dtype=np.float32, copy=True)
    if joined.ndim != 2 or joined.shape[1] != 64 or not np.isfinite(joined).all():
        raise ValueError('finite per-frame CNN features [T,64] required')
    device = next(temporal.parameters()).device
    if temporal.training:
        raise ValueError('frozen temporal encoder must be in eval mode')
    output = temporal(torch.from_numpy(joined)[None].to(device)).frame_hidden[0].cpu()
    labels = np.concatenate([np.full(r['stop']-r['start'], r['label'], dtype=np.int64) for r in instances])
    return dict(features=output, source_frames=indices, frame_labels=labels,
                targets=[r['label'] for r in instances])


def _array_lock(record, name, path):
    info = record['artifacts'][name]
    return dict(path=str(path), sha256=info['sha256'])


def _finite_array(path, shape):
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.shape != shape or not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f'invalid shape/dtype: {path}')
    for start in range(0, len(array), 1024):
        if not np.isfinite(array[start:start+1024]).all():
            raise ValueError(f'nonfinite feature cache: {path}')
    return array


def prepare(settings, output, device='cpu'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    try:
        return _prepare(settings, output, device)
    except Exception as error:
        write_json(output/'failed.json', dict(error=type(error).__name__, message=str(error)))
        raise


def _prepare(settings, output, device):
    from continuous_mmwave.harrad.model import SegmentClassifier
    locks = settings['locks']
    manifest_path = locked_path(locks['feature_manifest'])
    records = json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    validate_records(records)  # reject role leakage before any signal array access
    if {r['role'] for r in records} != {'train', 'validation'}:
        raise ValueError('both train and validation recordings are required')
    checkpoint = locked_path(locks['radar_checkpoint'])
    locked_path(locks['radar_normalizers'])
    verification_path = locked_path(locks['feature_verification'])
    verification = json.loads(verification_path.read_text(encoding='utf-8-sig'))
    if (verification.get('passed') is not True
            or verification.get('source_checkpoint_sha256') != locks['radar_checkpoint']['sha256']
            or verification.get('source_normalizers_sha256') != locks['radar_normalizers']['sha256']
            or verification.get('test_signal_values_read') is not False):
        raise ValueError('feature/front-end provenance mismatch')
    # Legacy HARrad checkpoints contain NumPy normalizers; this trusted file was SHA-checked above.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if saved.get('arm') != 'rd':
        raise ValueError('HARrad RD checkpoint required')
    frontend = SegmentClassifier('rd').to(device).requires_grad_(False).eval()
    frontend.load_state_dict(saved['model'])
    before = {k:v.detach().cpu().clone() for k,v in frontend.state_dict().items()}
    sampling = settings.get('data', {})
    samples = sample_sequences(records, seed=int(sampling.get('seed', 42)),
                               stride=int(sampling.get('sequence_stride', 3)),
                               max_events=int(sampling.get('max_events', 6)),
                               max_frames=int(sampling.get('max_sequence_frames', 900)))
    sequences = output/'sequences'; sequences.mkdir()
    prepared, natural, verified = [], [], []
    for record in records:
        root = Path(record['artifact_dir'])
        if not root.is_absolute():
            root = manifest_path.parent/root
        vectors_path = locked_path(_array_lock(record, 'frame_features.npy', root/'frame_features.npy'))
        hidden_path = locked_path(_array_lock(record, 'continuous_hidden.npy', root/'continuous_hidden.npy'))
        vectors = _finite_array(vectors_path, (record['frames'], 64))
        _finite_array(hidden_path, (record['frames'], 128))
        verified.extend([_array_lock(record, 'frame_features.npy', vectors_path),
                         _array_lock(record, 'continuous_hidden.npy', hidden_path)])
        natural.append({k:record[k] for k in ('file', 'subject', 'role', 'frames', 'instances')})
        natural[-1].update(path=str(hidden_path.resolve()), sha256=sha256(hidden_path))
        for row in samples:
            if row['file'] != record['file']:
                continue
            result = construct_sequence(vectors, row['instances'], frontend.temporal)
            path = sequences/(row['sample_id']+'.npz')
            np.savez(path, features=result['features'].numpy(), source_frames=result['source_frames'],
                     frame_labels=result['frame_labels'], targets=np.array(result['targets'], dtype=np.int64))
            prepared.append({**row, 'path':str(path.relative_to(output)), 'sha256':sha256(path),
                             'targets':result['targets'], 'frames':len(result['features'])})
    if any(not torch.equal(before[k], v.cpu()) for k,v in frontend.state_dict().items()):
        raise RuntimeError('frozen source frontend changed during preparation')
    # Detect concurrent changes after reading, before advertising successful preparation.
    for lock in verified:
        locked_path(lock)
    for lock in locks.values():
        locked_path(lock)
    payload = dict(format_version=1, dataset='HARrad', input_kind='constructed_sequences_and_natural_streams',
                   time_basis='nominal', frame_period_s=1/15, feature_dim=128, num_classes=12,
                   settings=settings, source_locks=locks, verified_arrays=verified,
                   sequences=prepared, natural=natural, test_signal_values_read=False)
    write_json(output/'prepared.json', payload)
    return payload


def load_prepared(path, *, verify_natural=False):
    path = Path(path)
    payload = json.loads(path.read_text(encoding='utf-8-sig'))
    if payload.get('format_version') != 1 or payload.get('test_signal_values_read') is not False:
        raise ValueError('invalid prepared manifest')
    validate_records(payload['natural'])
    subjects = {role:{r['subject'] for r in payload['natural'] if r['role']==role} for role in ('train', 'validation')}
    if not all(subjects.values()) or subjects['train'] & subjects['validation']:
        raise ValueError('prepared dataset lacks disjoint train/validation')
    by_file = {r['file']:r for r in payload['natural']}
    ids = set()
    for row in payload['sequences']:
        if (row['file'] not in by_file or row['role'] != by_file[row['file']]['role']
                or row['subject'] != by_file[row['file']]['subject'] or row['sample_id'] in ids):
            raise ValueError('sequence provenance or identity mismatch')
        ids.add(row['sample_id'])
        p = (path.parent/row['path']).resolve()
        if not p.is_relative_to(path.parent.resolve()):
            raise ValueError('sequence path escapes prepared directory')
        locked_path(dict(path=str(p), sha256=row['sha256']))
    if verify_natural:
        for row in payload['natural']:
            locked_path(dict(path=row['path'], sha256=row['sha256']))
    return payload


def read_sequence(root, row, device='cpu'):
    with np.load(Path(root)/row['path'], allow_pickle=False) as values:
        features = np.array(values['features'], dtype=np.float32, copy=True)
        targets = values['targets'].tolist()
        labels = np.array(values['frame_labels'], copy=True)
    if targets != row['targets'] or len(features) != row['frames'] or len(labels) != len(features):
        raise ValueError('prepared sequence contents disagree with manifest')
    return torch.from_numpy(features).to(device), targets, torch.from_numpy(labels).long().to(device)
