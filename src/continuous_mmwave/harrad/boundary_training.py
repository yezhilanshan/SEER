"""Complete-recording supervised boundary training with explicit partial-label scores."""
from pathlib import Path
import time
import numpy as np
import torch

from .boundary import (BoundaryHead, masked_boundary_loss, decode_boundaries,
                       score_events, combine_scores, independent_event_counts)
from .training import seed_everything
from .preprocess import write_json, sha256


@torch.no_grad()
def predict(model, records, device):
    model.eval()
    return [model(torch.from_numpy(r['x']).to(device)).sigmoid().cpu().numpy() for r in records]


def evaluate_head(records, probabilities, head, threshold, tolerance):
    name = ('start', 'end')[head]
    rows = []
    for record, scores in zip(records, probabilities):
        emitted = decode_boundaries(scores[:, head], record['cutoffs'], threshold)
        rows.append(score_events(emitted, record['truth'][name], record['known_frames'], tolerance))
    return combine_scores(rows)


def choose_thresholds(records, probabilities, candidates, tolerance):
    thresholds, metrics, search = {}, {}, {}
    for head, name in enumerate(('start', 'end')):
        values = candidates[name] if isinstance(candidates, dict) else candidates
        choices = [(float(t), evaluate_head(records, probabilities, head, float(t), tolerance)) for t in values]
        selected, result = max(choices, key=lambda r: (r[1]['f1'], r[0]))
        thresholds[name] = selected; metrics[name] = result
        search[name] = [{'threshold': t, 'f1': m['f1'], 'precision': m['precision'], 'recall': m['recall']} for t, m in choices]
    return thresholds, metrics, search


def evaluate_tolerances(records, probabilities, thresholds, tolerances):
    return {str(tol): {name: evaluate_head(records, probabilities, head, thresholds[name], tol)
                      for head, name in enumerate(('start', 'end'))} for tol in tolerances}


def _save_probabilities(directory, records, probabilities):
    directory.mkdir(exist_ok=True)
    for row, probability in zip(records, probabilities):
        np.savez_compressed(directory / (Path(row['file']).stem + '.npz'), probability=probability,
                            cutoffs=row['cutoffs'], known_frames=row['known_frames'],
                            truth_start=row['truth']['start'], truth_end=row['truth']['end'],
                            targets=row['targets'], mask=row['mask'])


def audit_predictions(records, probabilities, thresholds, tolerances, output):
    details = {}
    for head, name in enumerate(('start', 'end')):
        details[name] = {}
        for tol in tolerances:
            per_record = []
            for record, scores in zip(records, probabilities):
                emitted = decode_boundaries(scores[:, head], record['cutoffs'], thresholds[name])
                metrics = score_events(emitted, record['truth'][name], record['known_frames'], tol)
                counts = independent_event_counts(emitted, record['truth'][name], record['known_frames'], tol)
                if counts != tuple(metrics[key] for key in ('tp', 'fp', 'fn', 'unscored_unknown')):
                    raise RuntimeError('independent event match counts disagree')
                per_record.append({'file': record['file'], 'emitted': emitted.tolist(), **metrics})
            details[name][str(tol)] = per_record
    write_json(output, {'independent_verified': True, 'details': details})


def baseline_scores(train, val, cfg, output):
    # Change-energy thresholds come from train only; selection uses validation.
    energy_train = np.concatenate([np.abs(r['x'][:, 128:]).mean(1) for r in train])
    quantiles = [.5, .75, .9, .95, .975, .99]
    candidates = np.unique(np.quantile(energy_train, quantiles)).tolist()
    energies = [np.repeat(np.abs(r['x'][:, 128:]).mean(1)[:, None], 2, axis=1) for r in val]
    thresholds, metrics, search = choose_thresholds(val, energies, candidates, cfg['main_tolerance_frames'])
    result = {'hidden_change': {'thresholds': thresholds, 'train_quantiles': quantiles,
                               'main': metrics, 'search': search,
                               'tolerances': evaluate_tolerances(val, energies, thresholds, cfg['tolerances_frames'])}}
    _save_probabilities(output / 'baseline-hidden-change', val, energies)
    clocks = {}; clock_parameters = {}
    for name in ('start', 'end'):
        first = [r['truth'][name][0] for r in train if len(r['truth'][name])]
        intervals = np.concatenate([np.diff(r['truth'][name]) for r in train if len(r['truth'][name]) > 1])
        offset, period = float(np.median(first)), float(np.median(intervals))
        per_record = {}
        for record in val:
            desired = np.arange(max(1, offset), record['frames'] + 1, period)
            indexes = np.searchsorted(record['cutoffs'], desired)
            emitted = np.unique(record['cutoffs'][indexes[indexes < len(record['cutoffs'])]])
            per_record[record['file']] = emitted
        clock_parameters[name] = {'offset_frames': offset, 'period_frames': period}
        clocks[name] = {str(tol): combine_scores([
            score_events(per_record[r['file']], r['truth'][name], r['known_frames'], tol) for r in val])
            for tol in cfg['tolerances_frames']}
    result['periodic_clock'] = {'train_parameters': clock_parameters, 'metrics': clocks}
    write_json(output / 'baselines.json', result)
    return result


def fit_boundary(records, config, output, device='cuda'):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    train = [r for r in records if r['role'] == 'train']
    val = [r for r in records if r['role'] == 'validation']
    if not train or not val or any(r['role'] not in ('train', 'validation') for r in records):
        raise ValueError('train/validation only')
    if {r['subject'] for r in train} & {r['subject'] for r in val}:
        raise ValueError('subject leakage')
    positive = sum((r['targets'] * r['mask']).sum(0) for r in train)
    negative = sum(((1 - r['targets']) * r['mask']).sum(0) for r in train)
    if np.any(positive <= 0) or np.any(negative <= 0):
        raise ValueError('both boundary classes required')
    seed_everything(config['seed'], 4)
    model = BoundaryHead().to(device)
    weights = torch.tensor(negative / positive, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    torch.save({'model': model.state_dict(), 'epoch': 0}, output / 'initial.pt')
    write_json(output / 'protocol.json', {'config': config, 'positive': positive.tolist(), 'negative': negative.tolist(),
               'pos_weight': weights.cpu().tolist(), 'train_recordings': len(train), 'validation_recordings': len(val),
               'scope': 'partial-label boundary detection; no annotation-driven reset or inference mask'})
    history = []; best = -1.; steps = frames_seen = 0; started = time.time()
    for epoch in range(1, config['epochs'] + 1):
        losses = []
        for index in np.random.default_rng(config['seed'] * 10000 + epoch).permutation(len(train)):
            row = train[int(index)]; model.train(); optimizer.zero_grad(set_to_none=True)
            x = torch.from_numpy(row['x']).to(device)
            target = torch.from_numpy(row['targets']).to(device)
            mask = torch.from_numpy(row['mask']).to(device)
            loss = masked_boundary_loss(model(x), target, mask, weights)
            if not torch.isfinite(loss):
                raise RuntimeError('nonfinite boundary loss')
            loss.backward(); norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not torch.isfinite(norm):
                raise RuntimeError('nonfinite boundary gradient')
            optimizer.step(); steps += 1; frames_seen += row['frames']; losses.append(float(loss.detach()))
        probabilities = predict(model, val, device)
        thresholds, metrics, search = choose_thresholds(val, probabilities, config['thresholds'], config['main_tolerance_frames'])
        score = sum(m['f1'] for m in metrics.values()) / 2
        row = {'epoch': epoch, 'loss': float(np.mean(losses)), 'thresholds': thresholds, 'validation': metrics,
               'macro_boundary_f1': score, 'successful_steps': steps, 'train_frame_exposures': frames_seen,
               'seconds': time.time() - started}
        history.append(row)
        checkpoint = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch,
                      'thresholds': thresholds, 'config': config}
        torch.save(checkpoint, output / 'last.pt')
        if score > best:
            best, best_epoch = score, epoch
            torch.save(checkpoint, output / 'best.pt')
            _save_probabilities(output / 'validation-best', val, probabilities)
            write_json(output / 'best-threshold-search.json', search)
        write_json(output / 'history.json', history)
        write_json(output / 'status.json', {'phase': 'training', 'epoch': epoch, 'total_epochs': config['epochs'],
                   'successful_steps': steps, 'best_epoch': best_epoch, 'best_macro_boundary_f1': best,
                   'validation': metrics, 'updated_unix': time.time()})
        print(f"boundary epoch {epoch}/{config['epochs']}: startF1={metrics['start']['f1']:.4f} endF1={metrics['end']['f1']:.4f}", flush=True)
    saved = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(saved['model'])
    replay = predict(model, val, device); gap = 0.
    for record, actual in zip(val, replay):
        with np.load(output / 'validation-best' / (Path(record['file']).stem + '.npz')) as old:
            gap = max(gap, float(np.max(np.abs(actual - old['probability']))))
    if gap != 0:
        raise RuntimeError(f'best probability replay mismatch: {gap}')
    tolerances = evaluate_tolerances(val, replay, saved['thresholds'], config['tolerances_frames'])
    audit_predictions(val, replay, saved['thresholds'], config['tolerances_frames'], output / 'event-matches.json')
    baselines = baseline_scores(train, val, config, output)
    report = {'epochs_completed': config['epochs'], 'successful_steps': steps, 'train_frame_exposures': frames_seen,
              'training_recordings_per_epoch': len(train), 'validation_recordings': len(val), 'best_epoch': best_epoch,
              'thresholds': saved['thresholds'], 'metrics': tolerances, 'baselines': baselines,
              'replay_max_abs': gap, 'independent_verified': True,
              'trainable_parameters': sum(p.numel() for p in model.parameters()),
              'best_sha256': sha256(output / 'best.pt'), 'seconds': time.time() - started,
              'test_signal_values_read': False, 'metric_scope': 'partial labelled boundaries; unknown unmatched emissions unscored'}
    write_json(output / 'complete.json', report)
    write_json(output / 'status.json', {'phase': 'complete', 'updated_unix': time.time()})
    return report
