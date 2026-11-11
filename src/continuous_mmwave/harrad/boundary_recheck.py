"""Recheck existing completed epochs after fixing the independent match tie rule."""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from .boundary import BoundaryHead
from .boundary_training import predict, audit_predictions, evaluate_tolerances, baseline_scores, _save_probabilities
from .preprocess import sha256, write_json
from .training import seed_everything


def run(original, output, expected_best_sha):
    original, output = Path(original), Path(output)
    output.mkdir(parents=True, exist_ok=False); started = time.time()
    try:
        config = json.loads((original / 'config.json').read_text())
        history = json.loads((original / 'training/history.json').read_text())
        if len(history) != 20 or history[-1]['successful_steps'] != 620:
            raise ValueError('expected original 20 epochs / 620 updates')
        best = original / 'training/best.pt'
        if sha256(best) != expected_best_sha:
            raise ValueError('trained best changed')
        locks = config['locks']
        if any(sha256(row['path']) != row['sha256'] for row in locks.values()):
            raise ValueError('source lock changed')
        metadata = json.loads((original / 'data/manifest.json').read_text())
        if len(metadata) != 35 or any(r['role'] not in ('train', 'validation') for r in metadata):
            raise ValueError('unexpected roles or coverage')
        records = []
        for row in metadata:
            path = Path(row['cache'])
            for name, artifact in row['artifacts'].items():
                if sha256(path / name) != artifact['sha256']:
                    raise ValueError('prepared cache changed')
            x = np.array(np.load(path / 'inputs.npy', mmap_mode='r'))
            with np.load(path / 'targets.npz', allow_pickle=False) as a:
                record = {**row, 'x': x, 'cutoffs': a['cutoffs'], 'targets': a['targets'], 'mask': a['mask'],
                          'known_frames': a['known_frames'], 'truth': {'start': a['truth_start'], 'end': a['truth_end']}}
            records.append(record)
        train = [r for r in records if r['role'] == 'train']; val = [r for r in records if r['role'] == 'validation']
        if (len(train), len(val), sum(r['frames'] for r in train), sum(r['frames'] for r in val)) != (31, 4, 190238, 25244):
            raise ValueError('coverage changed')
        seed_everything(42, 4)
        saved = torch.load(best, map_location='cuda', weights_only=False)
        model = BoundaryHead().cuda(); model.load_state_dict(saved['model']); model.requires_grad_(False).eval()
        probabilities = predict(model, val, 'cuda'); gap = 0.
        for record, actual in zip(val, probabilities):
            with np.load(original / 'training/validation-best' / (Path(record['file']).stem + '.npz')) as prior:
                gap = max(gap, float(np.max(np.abs(actual - prior['probability']))))
        if gap != 0:
            raise RuntimeError('saved probability replay mismatch')
        _save_probabilities(output / 'validation-best', val, probabilities)
        thresholds = saved['thresholds']; tolerances = config['training']['tolerances_frames']
        write_json(output / 'status.json', {'phase': 'independent_match_audit', 'updated_unix': time.time()})
        audit_predictions(val, probabilities, thresholds, tolerances, output / 'event-matches.json')
        metrics = evaluate_tolerances(val, probabilities, thresholds, tolerances)
        for name in ('start', 'end'):
            expected = history[saved['epoch'] - 1]['validation'][name]
            actual = metrics['4'][name]
            if any(actual[k] != expected[k] for k in ('tp', 'fp', 'fn', 'unscored_unknown', 'f1')):
                raise RuntimeError('primary evaluation changed unexpectedly')
        baselines = baseline_scores(train, val, config['training'], output)
        if sha256(best) != expected_best_sha or any(sha256(row['path']) != row['sha256'] for row in locks.values()):
            raise RuntimeError('source changed during recheck')
        report = {'phase': 'complete', 'original_training': str(original), 'epochs_completed': 20,
                  'original_successful_steps': 620, 'new_optimizer_steps': 0, 'best_epoch': saved['epoch'],
                  'trained_best_sha256': expected_best_sha, 'thresholds': thresholds, 'metrics': metrics,
                  'baselines': baselines, 'replay_max_abs': gap, 'independent_verified': True,
                  'trainable_parameters': 33154, 'train_frame_exposures': history[-1]['train_frame_exposures'],
                  'source_weights_unchanged': True, 'test_signal_values_read': False,
                  'seconds': time.time() - started, 'completed_unix': time.time(),
                  'metric_scope': 'partial labelled boundary evaluation; unmatched unknown predictions unscored',
                  'original_failed_audit_preserved': True}
        write_json(output / 'complete.json', report)
        write_json(output / 'status.json', {'phase': 'complete', 'updated_unix': time.time()})
        print(json.dumps({'phase': 'complete', 'best_epoch': saved['epoch'], 'replay_max_abs': gap}), flush=True)
    except BaseException as exc:
        failure = {'phase': 'failed', 'error': repr(exc), 'traceback': traceback.format_exc()}
        write_json(output / 'failed.json', failure); write_json(output / 'status.json', failure)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--original', required=True); p.add_argument('--output', required=True)
    p.add_argument('--expected-best-sha', required=True); args = p.parse_args()
    run(args.original, args.output, args.expected_best_sha)
