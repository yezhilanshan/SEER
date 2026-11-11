"""Cache-only controlled linear-head experiments; no radar/LLM models are loaded."""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .diagnostics import BED_CLASSES, classification_metrics, confusion


def arm_specs(panel):
    if panel not in ('original_optimizer', 'probe_replay'):
        raise ValueError('unknown experiment panel')
    common = dict(panel=panel, seed=17, batch_size=4096, weight_decay=.01,
                  lr=.001 if panel == 'original_optimizer' else .01,
                  grad_clip=1. if panel == 'original_optimizer' else None,
                  initialization='checkpoint' if panel == 'original_optimizer' else 'random')
    factors = [(3, False, False), (20, False, False), (3, True, False),
               (3, False, True), (20, True, True)]
    return [dict(common, arm=f'H{i}', epochs=e, weighted=w, standardized=s)
            for i, (e, w, s) in enumerate(factors)]


def training_statistics(x, labels, classes=14):
    valid = (labels >= 0) & (labels < classes)
    if not valid.any() or not np.isfinite(x[valid]).all():
        raise ValueError('training features must be finite and have valid labels')
    mean, std = x[valid].mean(0), x[valid].std(0).clip(1e-5)
    counts = np.bincount(labels[valid], minlength=classes).astype('float32')
    weights = (counts.sum() / counts.clip(1)).astype('float32') / classes
    return mean, std, weights


def initialize_head(spec, reference, mean, std, device):
    torch.manual_seed(spec['seed'])
    width, classes = reference['weight'].shape[1], reference['weight'].shape[0]
    head = nn.Linear(width, classes).to(device)
    if spec['initialization'] == 'checkpoint':
        state = {key: value.detach().clone() for key, value in reference.items()}
        if spec['standardized']:
            # Preserve the raw checkpoint's logits when changing coordinates.
            center = torch.as_tensor(mean, device=state['weight'].device)
            scale = torch.as_tensor(std, device=state['weight'].device)
            state['bias'] = state['bias'] + state['weight'] @ center
            state['weight'] = state['weight'] * scale[None]
        head.load_state_dict(state)
    elif spec['initialization'] != 'random':
        raise ValueError('unknown initialization')
    return head


def assert_reference(actual, expected, predictions):
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all()
    assert np.array_equal(actual.argmax(-1), predictions), 'original head predictions changed'
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-6)
    return float(np.max(np.abs(actual-expected)))


def fit_head(arrays, spec, reference, output, *, device='cuda', classes=14):
    """Fit using train/validation only; never inspect test during selection."""
    train_x, train_y = arrays['train'][:2]
    val_x, val_y = arrays['validation'][:2]
    mean, std, weights = training_statistics(train_x, train_y, classes)
    if not spec['standardized']:
        mean, std = np.zeros_like(mean), np.ones_like(std)
    x = {name: torch.from_numpy((data[0]-mean)/std).to(device)
         for name, data in (('train', arrays['train']), ('validation', arrays['validation']))}
    y = {name: torch.from_numpy(data[1]).long().to(device)
         for name, data in (('train', arrays['train']), ('validation', arrays['validation']))}
    head = initialize_head(spec, reference, mean, std, device)
    weight = torch.from_numpy(weights).to(device) if spec['weighted'] else None
    optimizer = torch.optim.AdamW(head.parameters(), lr=spec['lr'], weight_decay=spec['weight_decay'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    best, history = -1., []

    def payload(epoch):
        return dict(head={k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
                    mean=torch.from_numpy(mean.copy()), std=torch.from_numpy(std.copy()),
                    epoch=epoch, spec=spec)

    for epoch in range(1, spec['epochs']+1):
        ids = torch.randperm(len(train_x), device=device)
        head.train()
        loss_total, loss_denominator = 0., 0.
        for part in ids.split(spec['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            target = y['train'][part]
            loss = F.cross_entropy(head(x['train'][part]), target, weight=weight)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite head loss')
            loss.backward()
            if spec['grad_clip'] is not None:
                nn.utils.clip_grad_norm_(head.parameters(), spec['grad_clip'], error_if_nonfinite=True)
            optimizer.step()
            denominator = float(weight[target].sum()) if weight is not None else len(part)
            loss_total += float(loss.detach()) * denominator
            loss_denominator += denominator
        head.eval()
        with torch.no_grad():
            logits = head(x['validation'])
            pred = logits.argmax(-1).cpu().numpy()
            nll = float(F.cross_entropy(logits, y['validation']))
        metric = classification_metrics(confusion(val_y, pred, classes=classes))
        history.append(dict(epoch=epoch, train_objective=loss_total/loss_denominator,
                            validation_macro_f1=metric['macro_f1'],
                            validation_accuracy=metric['frame_accuracy'], validation_nll=nll))
        if metric['macro_f1'] > best:
            best = metric['macro_f1']
            torch.save(payload(epoch), output)
        print(json.dumps(dict(panel=spec['panel'], arm=spec['arm'], **history[-1])), flush=True)
    torch.save(payload(spec['epochs']), output.with_name(output.stem+'-last.pt'))
    saved = torch.load(output, map_location='cpu', weights_only=True)
    return dict(spec=spec, history=history, selected_epoch=saved['epoch'],
                trainable_parameters=sum(p.numel() for p in head.parameters()),
                selection='validation macro-F1; ties earliest; test never used',
                updates_per_epoch=(len(train_x)+spec['batch_size']-1)//spec['batch_size'])


@torch.no_grad()
def evaluate_head(saved, arrays, *, device='cuda'):
    classes, width = saved['head']['weight'].shape
    head = nn.Linear(width, classes).to(device).eval()
    head.load_state_dict(saved['head'])
    mean, std = saved['mean'].numpy(), saved['std'].numpy()
    outputs, predictions = {}, {}
    for split, (features, labels, short, stable) in arrays.items():
        logits = head(torch.from_numpy((features-mean)/std).to(device))
        probabilities = logits.softmax(-1)
        pred = logits.argmax(-1).cpu().numpy()
        metric = classification_metrics(confusion(labels, pred, classes=classes))
        metric['steady'] = classification_metrics(confusion(labels, pred, stable, classes=classes))
        metric['bed_macro_f1'] = float(np.mean([metric['per_class'][i]['f1'] for i in BED_CLASSES if i < classes]))
        short_metric = classification_metrics(confusion(labels, pred, short, classes=classes))
        present = [row for row in short_metric['per_class'] if row['support']]
        metric['short_frame_support'] = short_metric['frames']
        metric['short_macro_recall_present_classes'] = float(np.mean([r['recall'] for r in present])) if present else None
        target = torch.from_numpy(labels).long().to(device)
        metric['nll'] = float(F.cross_entropy(logits, target))
        metric['brier'] = float(((probabilities-F.one_hot(target, classes)).square().sum(-1)).mean())
        confidence = probabilities.max(-1).values.cpu().numpy()
        correct = pred == labels
        bins = np.minimum((confidence*15).astype(int), 14)
        metric['ece_15_equal_width'] = float(sum(
            np.mean(bins==i)*abs(float(correct[bins==i].mean())-float(confidence[bins==i].mean()))
            for i in range(15) if (bins==i).any()))
        outputs[split], predictions[split] = metric, pred
    return outputs, predictions


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cache(root):
    root = Path(root)
    index = json.loads((root/'feature-index.json').read_text())
    hashes = json.loads((root/'feature-cache-manifest.json').read_text())
    arrays, raw = {}, {}
    subjects = {split: sorted({row['subject_id'] for row in rows}) for split, rows in index.items()}
    assert not (set(subjects['train']) & set(subjects['validation']))
    assert not (set(subjects['test']) & (set(subjects['train'])|set(subjects['validation'])))
    for split in ('train', 'validation', 'test'):
        parts, original = [], []
        for row in index[split]:
            path = root/'features'/row['cache']
            entry = hashes[row['cache']]
            assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'], str(path)
            with np.load(path, allow_pickle=False) as data:
                mask = (data['labels']>=0)&(data['labels']<14)
                parts.append(tuple(data[key][mask] for key in ('stage1_gru','labels','short','steady')))
                original.append((data['logits'][mask], data['predictions'][mask]))
        arrays[split] = tuple(np.concatenate([r[i] for r in parts]) for i in range(4))
        raw[split] = tuple(np.concatenate([r[i] for r in original]) for i in range(2))
    return arrays, raw, subjects, hashes
