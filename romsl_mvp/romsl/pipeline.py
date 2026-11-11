import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .mmfi_io import load_recording
from .model import RadarEncoder, RegressionHead, masked_huber
from .preprocess import make_windows, sample_points
from .targets import compute_targets


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root, job, config, joints):
    audit_path = job / 'reports/audit/data_audit.json'
    audit = json.loads(audit_path.read_text())
    if audit['gates']['A'] != 'READY' or audit['gates']['B'] != 'READY':
        raise ValueError('BLOCKED_DATA: A/B audit must pass before preparation')
    artifacts = job / 'artifacts'
    cache = job / 'cache'
    cache.mkdir(exist_ok=False)
    records = []
    point_sum, point_square, point_count = np.zeros(5), np.zeros(5), 0
    for split in ('train', 'val'):
        for subject in config['data']['splits'][split]:
            for action in range(1, 28):
                rec = load_recording(root / config['data']['scene'] / subject / f'A{action:02}')
                targets = compute_targets(rec.pose, joints, 1 / config['data']['fps'])
                valid_points = [p[np.isfinite(p).all(axis=1)] for p in rec.points_per_frame]
                radar_valid = np.array([len(p) > 0 for p in valid_points])
                target_valid = targets['A_mask'].any(-1) & targets['B_mask'].any(-1)
                windows = make_windows(radar_valid, target_valid, stride=8 if split == 'train' else 16)
                records.append((split, rec, valid_points, targets, windows))
                if split == 'train':
                    points = np.concatenate(valid_points)
                    point_sum += points.sum(axis=0)
                    point_square += (points ** 2).sum(axis=0)
                    point_count += len(points)
    if not point_count:
        raise ValueError('No finite training radar points')
    mean = point_sum / point_count
    std = np.sqrt(np.maximum(point_square / point_count - mean ** 2, 1e-12))
    np.savez(artifacts / 'input_stats.npz', mean=mean, std=std, source_subjects=config['data']['splits']['train'])
    manifests = {'train': [], 'val': []}
    target_stats = {}
    for exp in ('A', 'B'):
        values = np.concatenate([targets[exp] for split, _, _, targets, _ in records if split == 'train'])
        valid = np.concatenate([targets[exp + '_mask'] for split, _, _, targets, _ in records if split == 'train'])
        count = valid.sum(0)
        if not (count > 0).all():
            raise ValueError(f'No training labels for some {exp} components')
        target_mean = (values * valid).sum(0, dtype=np.float64) / count
        target_var = (((values - target_mean) ** 2) * valid).sum(0) / count
        target_stats[exp + '_mean'] = target_mean
        target_stats[exp + '_std'] = np.sqrt(np.maximum(target_var, 1e-12))
    np.savez(artifacts / 'target_stats.npz', **target_stats)
    for split, rec, points, targets, windows in records:
        sampled, masks = [], []
        for frame, cloud in enumerate(points):
            seed = int(hashlib.sha256(f'42/{rec.recording_id}/{frame}'.encode()).hexdigest()[:8], 16)
            sampled_frame, mask = sample_points(cloud, mean, std, seed=seed)
            sampled.append(sampled_frame)
            masks.append(mask)
        filename = rec.recording_id.replace('/', '_') + '.npz'
        np.savez(cache / filename, points=np.stack(sampled), point_mask=np.stack(masks), **targets)
        for start, end in windows:
            manifests[split].append({
                'recording_id': rec.recording_id, 'action_id': rec.action_id,
                'split': split, 'start': start, 'end': end, 'cache': filename,
            })
    for split, rows in manifests.items():
        if not rows:
            raise ValueError(f'No common windows for {split}')
        (artifacts / f'manifest_{split}.jsonl').write_text(
            ''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    lock_files = [audit_path, artifacts / 'input_stats.npz', artifacts / 'target_stats.npz',
                  artifacts / 'manifest_train.jsonl', artifacts / 'manifest_val.jsonl']
    lock = {str(path.relative_to(job)): digest(path) for path in lock_files}
    (artifacts / 'preparation_lock.json').write_text(json.dumps(lock, indent=2) + '\n')
    return {key: len(rows) for key, rows in manifests.items()}


class Windows(torch.utils.data.Dataset):
    def __init__(self, job, split, exp):
        self.job, self.exp = job, exp
        self.rows = [json.loads(line) for line in (job / f'artifacts/manifest_{split}.jsonl').read_text().splitlines()]
        with np.load(job / 'artifacts/target_stats.npz') as stats:
            self.mean = stats[exp + '_mean'].astype(np.float32)
            self.std = stats[exp + '_std'].astype(np.float32)
        self.loaded = {}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        filename = row['cache']
        if filename not in self.loaded:
            with np.load(self.job / 'cache' / filename) as data:
                self.loaded[filename] = {key: data[key] for key in ('points', 'point_mask', self.exp, self.exp + '_mask')}
        data = self.loaded[filename]
        sl = slice(row['start'], row['end'])
        return (torch.from_numpy(data['points'][sl]), torch.from_numpy(data['point_mask'][sl]),
                torch.from_numpy((data[self.exp][sl] - self.mean) / self.std),
                torch.from_numpy(data[self.exp + '_mask'][sl]))


def smoke(job, exp, steps=250):
    for filename, expected in json.loads((job / 'artifacts/preparation_lock.json').read_text()).items():
        if digest(job / filename) != expected:
            raise ValueError(f'Artifact changed: {filename}')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    encoder = RadarEncoder()
    head = RegressionHead(51 if exp == 'A' else 5)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=.001, weight_decay=.0001)
    dataset = Windows(job, 'train', exp)
    selected = np.random.default_rng(42).choice(len(dataset), min(32, len(dataset)), replace=False)
    batch = torch.utils.data.default_collate([dataset[int(index)] for index in selected])
    points, point_mask, targets, target_mask = batch
    frame_mask = point_mask.any(-1) & target_mask.any(-1)
    history = []
    directory = job / 'runs' / f'{exp}_smoke'
    directory.mkdir(parents=True, exist_ok=False)
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = head(encoder(points, point_mask))
        loss = masked_huber(prediction, targets, target_mask, frame_mask)
        if not torch.isfinite(loss):
            raise ValueError(f'Nonfinite loss at step {step}')
        loss.backward()
        gradient = torch.nn.utils.clip_grad_norm_([*encoder.parameters(), *head.parameters()], 1)
        if not torch.isfinite(gradient):
            raise ValueError('Nonfinite gradient')
        optimizer.step()
        history.append({'step': step, 'loss': float(loss.detach()), 'gradient_norm': float(gradient)})
        if step % 50 == 0:
            print(exp, history[-1], flush=True)
    initial = float(np.mean([row['loss'] for row in history[:10]]))
    final = float(np.mean([row['loss'] for row in history[-10:]]))
    result = {'experiment': exp, 'steps': steps, 'seed': 42, 'windows': len(selected),
              'selected_indices': selected.tolist(), 'initial_mean_loss': initial,
              'final_mean_loss': final, 'loss_ratio': final / initial,
              'passed': final < .5 * initial, 'device': 'cpu', 'stage': 'smoke_only'}
    (directory / 'history.json').write_text(json.dumps(history, indent=2) + '\n')
    (directory / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    torch.save({'encoder': encoder.state_dict(), 'head': head.state_dict(), 'result': result}, directory / 'smoke.pt')
    if not result['passed']:
        raise ValueError(f'Smoke did not reduce loss sufficiently: {result}')
    return result
