"""Immutable continuous feature contract and durable boundary baseline entrypoint."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np

from .boundary import continuous_inputs, boundary_targets
from .boundary_training import fit_boundary
from .preprocess import sha256, write_json


def prepare_records(manifest, output):
    if not manifest or any(r['role'] not in ('train', 'validation') for r in manifest):
        raise ValueError('forbidden or empty role')
    if not any(r['role'] == 'train' for r in manifest):
        raise ValueError('train role required')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    cache = output / 'features'; cache.mkdir()
    raw = []; total = np.zeros(256, dtype=np.float64); square = total.copy(); count = 0
    for row in manifest:
        path = Path(row['continuous_path'])
        if sha256(path) != row['artifacts']['continuous_hidden.npy']['sha256']:
            raise ValueError('continuous cache hash mismatch')
        hidden = np.load(path, mmap_mode='r')
        if hidden.shape != (row['frames'], 128):
            raise ValueError('invalid continuous hidden shape')
        x, cutoffs = continuous_inputs(hidden)
        targets = boundary_targets(row['frames'], row['instances'])
        if not np.array_equal(cutoffs, targets['cutoffs']):
            raise RuntimeError('feature/target Update mismatch')
        if not np.all(targets['targets'].sum(0) == len(row['instances'])):
            raise RuntimeError('boundary target coverage mismatch')
        record = {**targets, 'x': x, 'file': row['file'], 'subject': row['subject'],
                  'role': row['role'], 'frames': row['frames']}
        raw.append(record)
        if row['role'] == 'train':
            values = x.astype(np.float64)
            total += values.sum(0); square += (values * values).sum(0); count += len(values)
    mean = total / count
    scale = np.sqrt(np.maximum(square / count - mean * mean, 1e-6))
    np.savez(output / 'normalizers.npz', mean=mean.astype('float32'), scale=scale.astype('float32'), train_updates=count)
    manifest_out = []
    for record in raw:
        record['x'] = np.clip((record['x'] - mean.astype('float32')) / scale.astype('float32'), -8, 8).astype('float32')
        directory = cache / Path(record['file']).stem; directory.mkdir(exist_ok=False)
        np.save(directory / 'inputs.npy', record['x'])
        np.savez_compressed(directory / 'targets.npz', targets=record['targets'], mask=record['mask'], cutoffs=record['cutoffs'],
                            known_frames=record['known_frames'], truth_start=record['truth']['start'], truth_end=record['truth']['end'])
        meta = {key: record[key] for key in ('file', 'subject', 'role', 'frames')}
        meta.update(updates=len(record['x']), instances=len(record['truth']['start']),
                    unknown_frames=int((~record['known_frames']).sum()),
                    positive=record['targets'].sum(0).astype(int).tolist(),
                    valid_negative=((1-record['targets']) * record['mask']).sum(0).astype(int).tolist(),
                    cache=str(directory), artifacts={p.name: {'bytes': p.stat().st_size, 'sha256': sha256(p)} for p in directory.iterdir()})
        manifest_out.append(meta)
    write_json(output / 'manifest.json', manifest_out)
    return raw


def run(config, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False); started = time.time()
    try:
        write_json(output / 'config.json', config)
        locks = {name: sha256(value['path']) for name, value in config['locks'].items()}
        if any(locks[name] != value['sha256'] for name, value in config['locks'].items()):
            raise ValueError('locked source changed')
        manifest = json.loads(Path(config['locks']['continuous_manifest']['path']).read_text())
        canonical = json.loads(Path(config['locks']['canonical_manifest']['path']).read_text())
        if len(manifest) != 35 or len(canonical) != 35:
            raise ValueError('unexpected source recording count')
        source = {r['file']: r for r in canonical}
        for row in manifest:
            expected = source[row['file']]
            if any(row[k] != expected[k] for k in ('role', 'subject', 'frames', 'instances')):
                raise ValueError('annotation metadata differs from canonical contract')
            if row['role'] == 'train' and row['subject'] not in {f'subject{i}' for i in range(1, 7)}:
                raise ValueError('unexpected train subject')
            if row['role'] == 'validation' and row['subject'] != 'subject7':
                raise ValueError('unexpected validation subject')
            if row['role'] not in ('train', 'validation'):
                raise ValueError('test signal forbidden')
        for role, count, frames, instances in [('train', 31, 190238, 3018), ('validation', 4, 25244, 511)]:
            rows = [r for r in manifest if r['role'] == role]
            if len(rows) != count or sum(r['frames'] for r in rows) != frames or sum(len(r['instances']) for r in rows) != instances:
                raise ValueError('incomplete role coverage')
        write_json(output / 'input-lock.json', {'locks': config['locks'], 'test_signal_values_read': False,
                   'prediction_input': 'continuous GRU hidden and past difference; no Oracle state or boundary reset'})
        write_json(output / 'status.json', {'phase': 'continuous_data_preparation', 'updated_unix': time.time()})
        records = prepare_records(manifest, output / 'data')
        write_json(output / 'status.json', {'phase': 'boundary_training', 'updated_unix': time.time()})
        result = fit_boundary(records, config['training'], output / 'training', 'cuda')
        if any(sha256(value['path']) != locks[name] for name, value in config['locks'].items()):
            raise RuntimeError('source weight or metadata changed')
        report = {'phase': 'complete', 'training': result, 'source_weights_unchanged': True,
                  'test_signal_values_read': False, 'seconds': time.time()-started, 'completed_unix': time.time()}
        write_json(output / 'complete.json', report)
        write_json(output / 'status.json', {'phase': 'complete', 'updated_unix': time.time()})
    except BaseException as exc:
        failure = {'phase': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc(), 'updated_unix': time.time()}
        write_json(output / 'failed.json', failure); write_json(output / 'status.json', failure)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); run(json.loads(Path(args.config).read_text()), args.output)
