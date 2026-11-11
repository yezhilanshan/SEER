import csv
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .mmfi_io import load_recording
from .model import RadarEncoder, RegressionHead, masked_huber
from .pipeline import Windows, digest
from .preprocess import make_windows, sample_points
from .targets import compute_targets


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_checkpoint(path, value):
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def classification_metrics(truth, prediction, classes):
    matrix = np.zeros((classes, classes), dtype=int)
    np.add.at(matrix, (truth, prediction), 1)
    true_positive = matrix.diagonal()
    precision = np.divide(true_positive, matrix.sum(0), out=np.zeros(classes), where=matrix.sum(0) != 0)
    recall = np.divide(true_positive, matrix.sum(1), out=np.zeros(classes), where=matrix.sum(1) != 0)
    f1 = np.divide(2*precision*recall, precision+recall, out=np.zeros(classes), where=precision+recall != 0)
    return {'accuracy': float(true_positive.sum() / max(matrix.sum(), 1)), 'macro_f1': float(f1.mean()),
            'per_class_precision': precision.tolist(), 'per_class_recall': recall.tolist(),
            'per_class_f1': f1.tolist(), 'confusion_matrix': matrix.tolist(), 'recordings': int(matrix.sum())}


def clip_classification(logits, recordings, truth, classes):
    recordings = np.asarray(recordings)
    truth = np.asarray(truth)
    ids = sorted(set(recordings))
    labels, predictions = [], []
    for recording in ids:
        selected = recordings == recording
        if len(set(truth[selected])) != 1:
            raise ValueError(f'Conflicting labels: {recording}')
        labels.append(int(truth[selected][0]))
        predictions.append(int(logits[selected].mean(0).argmax()))
    result = classification_metrics(labels, predictions, classes)
    result['per_subject'] = {}
    subjects = [identifier.split('/')[-2] for identifier in ids]
    for subject in sorted(set(subjects)):
        indices = [i for i, item in enumerate(subjects) if item == subject]
        result['per_subject'][subject] = classification_metrics(
            [labels[i] for i in indices], [predictions[i] for i in indices], classes)
    result['predictions'] = [{'recording_id': identifier, 'truth': y, 'prediction': p}
                             for identifier, y, p in zip(ids, labels, predictions)]
    return result


def recording_mae(errors, valid, recordings):
    recordings = np.asarray(recordings)
    per_recording = {}
    for identifier in sorted(set(recordings)):
        selected = recordings == identifier
        count = valid[selected].sum(axis=0)
        if not (count > 0).all():
            raise ValueError(f'No valid motion targets in {identifier}')
        per_recording[identifier] = ((errors[selected] * valid[selected]).sum(axis=0) / count).tolist()
    per_subject = {}
    for subject in sorted({key.split('/')[-2] for key in per_recording}):
        per_subject[subject] = np.mean([v for k,v in per_recording.items() if k.split('/')[-2] == subject], axis=0).tolist()
    aggregate = np.mean(list(per_subject.values()), axis=0)
    return {'per_component': aggregate.tolist(), 'mean_mae': float(aggregate.mean()),
            'per_recording': per_recording, 'per_subject': per_subject}


def reserve_test(job, lock):
    with (job / 'test_access_reserved.json').open('x') as handle:
        json.dump(lock, handle, indent=2)


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def loader(dataset, batch_size, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed))


def verify_lock(job):
    lock = json.loads((job / 'experiment_lock.json').read_text())
    for filename, expected in lock['files'].items():
        if digest(job / filename) != expected:
            raise ValueError(f'Locked file changed: {filename}')
    return digest(job / 'experiment_lock.json')


def pretrain(job, exp, device, settings, dataset_class=Windows):
    seed = settings.get('seed', 42)
    seed_all(seed)
    encoder, head = RadarEncoder().to(device), RegressionHead(51 if exp == 'A' else 5).to(device)
    run = job / 'runs' / exp
    run.mkdir(parents=True, exist_ok=False)
    lock_sha = verify_lock(job)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=.001, weight_decay=.0001)
    datasets = {split: dataset_class(job, split, exp) for split in ('train', 'val')}
    loaders = {split: loader(data, 32, split == 'train', seed) for split, data in datasets.items()}
    best, stale = float('inf'), 0
    history = []
    started = time.monotonic()
    for epoch in range(1, settings['pretrain_epochs'] + 1):
        row = {'epoch': epoch}
        for split, batches in loaders.items():
            encoder.train(split == 'train')
            head.train(split == 'train')
            total, denominator, gradients = 0., 0, []
            with torch.set_grad_enabled(split == 'train'):
                for batch in batches:
                    points, point_mask, target, target_mask = [item.to(device) for item in batch[:4]]
                    weight = batch[4].to(device) if len(batch)==5 else None
                    frame_mask = point_mask.any(-1) & target_mask.any(-1)
                    count = int((frame_mask[..., None] & target_mask).sum())
                    prediction = head(encoder(points, point_mask))
                    loss = masked_huber(prediction, target, target_mask, frame_mask, weight)
                    if not torch.isfinite(loss):
                        raise ValueError(f'{exp} {split} nonfinite loss')
                    if split == 'train':
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        norm = torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *head.parameters()], 1)
                        if not torch.isfinite(norm):
                            raise ValueError('Nonfinite gradient')
                        optimizer.step()
                        gradients.append(float(norm))
                    loss_denominator = count if weight is None else float(
                        ((frame_mask[...,None] & target_mask)*weight[...,None]).sum())
                    total += float(loss.detach()) * loss_denominator
                    denominator += loss_denominator
            row[split + '_loss'] = total / max(denominator, 1)
            row[split + '_effective_targets'] = denominator
            if gradients:
                row['grad_norm'] = float(np.mean(gradients))
        row.update(lr=.001, elapsed_sec=time.monotonic() - started)
        history.append(row)
        state = {'encoder': encoder.state_dict(), 'head': head.state_dict(),
                 'optimizer': optimizer.state_dict(), 'epoch': epoch, 'seed': seed,
                 'experiment_lock_sha': lock_sha, 'selection_metric': row['val_loss']}
        save_checkpoint(run / 'last.pt', state)
        if row['val_loss'] < best:
            best, stale = row['val_loss'], 0
            save_checkpoint(run / 'best.pt', state)
        else:
            stale += 1
        with (run / 'history.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        print('PRETRAIN', exp, json.dumps(row), flush=True)
        if stale >= 5:
            break
    write_json(run / 'completed.json', {'epochs': len(history), 'best_val_loss': best,
                                       'best_sha': digest(run / 'best.pt'), 'lock_sha': lock_sha})


def extract_features(job, exp, split, device):
    checkpoint = torch.load(job / 'runs' / exp / 'best.pt', map_location='cpu', weights_only=True)
    if checkpoint['experiment_lock_sha'] != digest(job / 'experiment_lock.json'):
        raise ValueError('Encoder and experiment lock differ')
    encoder = RadarEncoder().to(device)
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()
    encoder.requires_grad_(False)
    dataset = Windows(job, split, 'B')
    features, targets, masks, frame_masks = [], [], [], []
    with torch.inference_mode():
        for points, point_mask, target, target_mask in loader(dataset, 64):
            frame_mask = point_mask.any(-1) & target_mask.any(-1)
            hidden = encoder(points.to(device), point_mask.to(device)).cpu()
            features.append(hidden)
            targets.append(target)
            masks.append(target_mask & frame_mask[..., None])
            frame_masks.append(frame_mask)
    return {'features': torch.cat(features), 'target': torch.cat(targets), 'mask': torch.cat(masks),
            'frame_mask': torch.cat(frame_masks), 'rows': dataset.rows, 'std': dataset.std}


def pooled_features(data):
    mask = data['frame_mask'][..., None]
    return (data['features'] * mask).sum(1) / mask.sum(1).clamp_min(1)


def attribute_scores(head, data, device):
    predictions = []
    with torch.inference_mode():
        for (features,) in loader(TensorDataset(data['features']), 64):
            predictions.append(head(features.to(device)).cpu())
    error = (torch.cat(predictions) - data['target']).abs().numpy() * data['std']
    ids = [row['recording_id'] for row in data['rows'] for _ in range(16)]
    return recording_mae(error.reshape(-1, 5), data['mask'].numpy().reshape(-1, 5), ids)


def classification_data(data, actions):
    indices = [i for i,row in enumerate(data['rows']) if row['action_id'] in actions]
    labels = torch.tensor([actions.index(data['rows'][i]['action_id']) for i in indices])
    ids = [data['rows'][i]['recording_id'] for i in indices]
    return pooled_features(data)[indices], labels, ids


def classification_scores(head, data, actions, device):
    features, labels, ids = classification_data(data, actions)
    with torch.inference_mode():
        logits = head(features.to(device)).cpu().numpy()
    return clip_classification(logits, ids, labels.numpy(), len(actions))


def fit_probes(job, exp, device, config, settings):
    seed = settings.get('seed', 42)
    seed_all(seed)
    data = {split: extract_features(job, exp, split, device) for split in ('train', 'val')}
    actions = config['data']['probe_actions']
    run = job / 'runs' / exp
    for task in ('classification', 'motion'):
        seed_all(seed)
        head = nn.Linear(128, 8 if task == 'classification' else 5).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
        if task == 'classification':
            features, labels, _ = classification_data(data['train'], actions)
            batches = loader(TensorDataset(features, labels), 64, True, seed)
        else:
            batches = loader(TensorDataset(data['train']['features'], data['train']['target'],
                                          data['train']['mask'], data['train']['frame_mask']), 64, True, seed)
        best, stale, history = -float('inf'), 0, []
        for epoch in range(1, settings['probe_epochs'] + 1):
            head.train()
            for batch in batches:
                batch = [item.to(device) for item in batch]
                prediction = head(batch[0])
                loss = (nn.functional.cross_entropy(prediction, batch[1]) if task == 'classification'
                        else masked_huber(prediction, batch[1], batch[2], batch[3]))
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite probe loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            head.eval()
            scores = (classification_scores(head, data['val'], actions, device) if task == 'classification'
                      else attribute_scores(head, data['val'], device))
            metric = scores['macro_f1'] if task == 'classification' else -scores['mean_mae']
            history.append({'epoch': epoch, 'val_metric': metric})
            if metric > best:
                best, stale = metric, 0
                save_checkpoint(run / f'probe_{task}_best.pt',
                                {'head': head.state_dict(), 'epoch': epoch, 'metric': metric,
                                 'encoder_sha': digest(run / 'best.pt'), 'seed': seed})
            else:
                stale += 1
            print('PROBE', exp, task, epoch, metric, flush=True)
            if stale >= 5:
                break
        write_json(run / f'probe_{task}_history.json', history)
    write_json(run / 'probes_completed.json', {
        task: digest(run / f'probe_{task}_best.pt') for task in ('classification', 'motion')})


def prepare_test(root, job, config):
    joints = json.loads((job / 'source/configs/joints.yaml').read_text())
    with np.load(job / 'artifacts/input_stats.npz') as stats:
        mean, std = stats['mean'], stats['std']
    rows = []
    for subject in config['data']['splits']['test']:
        for action in range(1, 28):
            rec = load_recording(root / 'E01' / subject / f'A{action:02}')
            targets = compute_targets(rec.pose, joints, .1)
            points, masks = [], []
            for frame, cloud in enumerate(rec.points_per_frame):
                seed = int(hashlib.sha256(f'42/{rec.recording_id}/{frame}'.encode()).hexdigest()[:8], 16)
                p, m = sample_points(cloud, mean, std, seed=seed)
                points.append(p)
                masks.append(m)
            points, masks = np.stack(points), np.stack(masks)
            filename = rec.recording_id.replace('/', '_') + '.npz'
            np.savez(job / 'cache' / filename, points=points, point_mask=masks, **targets)
            for start, end in make_windows(masks.any(-1), targets['A_mask'].any(-1) & targets['B_mask'].any(-1), stride=16):
                rows.append({'recording_id': rec.recording_id, 'action_id': rec.action_id,
                             'split': 'test', 'start': start, 'end': end, 'cache': filename})
    with (job / 'artifacts/manifest_test.jsonl').open('x') as handle:
        handle.write(''.join(json.dumps(row) + '\n' for row in rows))


def evaluate_locked(root, job, config, device, reuse_test=False):
    verify_lock(job)
    lock = {'experiment_lock_sha': digest(job / 'experiment_lock.json'), 'models': {}}
    for exp in ('A', 'B'):
        lock['models'][exp] = {name: digest(job / 'runs' / exp / name) for name in
                               ('best.pt', 'probe_classification_best.pt', 'probe_motion_best.pt')}
    reserve_test(job, lock)
    if not reuse_test:
        prepare_test(root, job, config)
    results = {}
    for exp in ('A', 'B'):
        data = extract_features(job, exp, 'test', device)
        heads = {}
        for task, dimensions in (('classification', 8), ('motion', 5)):
            head = nn.Linear(128, dimensions).to(device)
            checkpoint = torch.load(job / 'runs' / exp / f'probe_{task}_best.pt', map_location='cpu', weights_only=True)
            if checkpoint['encoder_sha'] != digest(job / 'runs' / exp / 'best.pt'):
                raise ValueError('Probe encoder mismatch')
            head.load_state_dict(checkpoint['head'])
            heads[task] = head.eval()
        results[exp] = {
            'classification': classification_scores(heads['classification'], data, config['data']['probe_actions'], device),
            'motion': attribute_scores(heads['motion'], data, device)}
    results['C'] = results['D'] = {'status': 'BLOCKED_GEOMETRY'}
    results['test_lock'] = lock
    results['evaluation_role'] = 'supplementary stability analysis on previously read S09/S10' if reuse_test else 'first locked evaluation'
    results['motion_units'] = 'nominal m/s; skeleton meter scale inferred from anthropometry'
    write_json(job / 'test_results.json', results)
    print('TEST_RESULTS', json.dumps(results), flush=True)


def run_all(job, root, device, settings):
    torch.set_num_threads(4)
    verify_lock(job)
    config = json.loads((job / 'source/configs/mvp.yaml').read_text())
    write_json(job / 'environment.json', {'python': platform.python_version(), 'torch': torch.__version__,
                                         'device': str(device), 'cuda': torch.version.cuda,
                                         'settings': settings, 'data_root': str(root)})
    for exp in ('A', 'B'):
        pretrain(job, exp, device, settings)
        fit_probes(job, exp, device, config, settings)
    evaluate_locked(root, job, config, device, settings.get('reuse_test', False))
    write_json(job / 'completed.json', {'test_results_sha': digest(job / 'test_results.json')})
