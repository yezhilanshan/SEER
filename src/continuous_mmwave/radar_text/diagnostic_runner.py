"""Frozen full-recording diagnosis and equal-budget linear feature probes.

Run subcommands separately so a failed stage can be resumed without refitting others.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .config import RadarTextConfig
from .data import collate_windows, caption_from_actions
from .diagnostics import (BED_CLASSES, action_instances, attribute_instances, classification_metrics,
                          confusion, period, runs, smooth_predictions, steady_mask)
from .diagnostic_models import frozen_features
from .metrics import score_caption
from .model import RadarEncoder, RadarPretrainer
from .runner import datasets, load_weights, read_checkpoint, read_settings


FEATURES = ('stage1_cnn', 'stage1_gru', 'align_cnn', 'align_gru')
METRICS = ('action_recall', 'action_precision', 'sequence_exact', 'order_lcs', 'template_exact', 'unmatched')


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def means(rows):
    return {key: float(np.mean([row[key] for row in rows])) if rows else None for key in METRICS}


def render(actions, phrases):
    return caption_from_actions(actions, phrases) if actions else 'No activity was predicted.'


def init(args):
    torch.set_num_threads(4)
    torch.manual_seed(17)
    np.random.seed(17)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    settings = read_settings(Path(args.pilot) / 'align.yaml')
    return root, settings


@torch.no_grad()
def extract(args):
    root, settings = init(args)
    sets, phrases = datasets(settings, ('train', 'validation', 'test'))
    saved = read_checkpoint(Path(args.pilot) / 'stage1/best.pt')
    stage1 = RadarPretrainer(RadarTextConfig(**saved['config']), 14).cuda().eval()
    load_weights(Path(args.pilot) / 'stage1/best.pt', stage1)
    align = RadarEncoder(stage1.config).cuda().eval()
    align.load_state_dict(read_checkpoint(Path(args.pilot) / 'align/best.pt')['encoder'])
    cache = root / 'features'
    cache.mkdir(exist_ok=True)
    reports, index = {}, {}
    for split, dataset in sets.items():
        all_matrix, steady_matrix = np.zeros((14, 14), dtype=np.int64), np.zeros((14, 14), dtype=np.int64)
        records = []
        for rid, meta in enumerate(dataset.recordings.rows):
            sample = dataset.recordings[rid]
            times, labels = sample.timestamps.numpy(), sample.labels.numpy()
            dt = period(times)
            stable = steady_mask(labels, times)
            short = np.zeros(len(labels), dtype=bool)
            for a, b, _ in runs(labels):
                if times[b - 1] - times[a] + dt < 2. - 1e-8:
                    short[a:b] = True
            chunks = []
            first = 0
            while first < len(labels):
                stop = max(first + 1, int(np.searchsorted(times, times[first] + 12 - 1e-7)))
                chunks.append((first, stop))
                first = stop
            values = {key: [] for key in FEATURES}
            predictions, logits = [], []
            for a, b in chunks:
                rd, md = sample.rd[a:b][None].cuda(), sample.md[a:b][None].cuda()
                length = torch.tensor([b - a], device='cuda')
                cnn, gru = frozen_features(stage1.encoder, rd, md, length)
                values['stage1_cnn'].append(cnn[0].cpu().numpy())
                values['stage1_gru'].append(gru[0].cpu().numpy())
                output = stage1.classifier(gru)[0]
                logits.append(output.cpu().numpy())
                predictions.append(output.argmax(-1).cpu().numpy())
                cnn, gru = frozen_features(align, rd, md, length)
                values['align_cnn'].append(cnn[0].cpu().numpy())
                values['align_gru'].append(gru[0].cpu().numpy())
            pred = np.concatenate(predictions)
            assert len(pred) == len(labels)
            filename = f'{split}-{rid:03}.npz'
            np.savez(cache / filename, labels=labels, timestamps=times, steady=stable, short=short,
                     predictions=pred, logits=np.concatenate(logits),
                     **{key: np.concatenate(value) for key, value in values.items()})
            all_matrix += confusion(labels, pred)
            steady_matrix += confusion(labels, pred, stable)
            records.append(dict(meta, cache=filename, frames=len(labels), period=dt))
            print(json.dumps(dict(stage='extract', split=split, recording=rid + 1,
                                  recordings=len(dataset.recordings), frames=len(labels))), flush=True)
        reports[split] = dict(all_valid=classification_metrics(all_matrix),
                              steady=classification_metrics(steady_matrix),
                              context='nonoverlapping 12s chunks, final short chunk included; each original frame counted once',
                              transition_radius_seconds=.3, recordings=records)
        index[split] = records
    dump(root / 'feature-index.json', index)
    dump(root / 'frame-classification.json', reports)


def probes(args):
    root, _ = init(args)
    index = json.loads((root / 'feature-index.json').read_text())
    result = {}
    for key in FEATURES:
        arrays = {}
        for split in ('train', 'validation', 'test'):
            parts = []
            for row in index[split]:
                with np.load(root / 'features' / row['cache'], allow_pickle=False) as data:
                    mask = data['labels'] >= 0
                    parts.append((data[key][mask], data['labels'][mask], data['short'][mask], data['steady'][mask]))
            arrays[split] = [np.concatenate([row[i] for row in parts]) for i in range(4)]
        mean, std = arrays['train'][0].mean(0), arrays['train'][0].std(0).clip(1e-5)
        x = {split: torch.from_numpy((data[0] - mean) / std).cuda() for split, data in arrays.items()}
        y = {split: torch.from_numpy(data[1]).long().cuda() for split, data in arrays.items()}
        torch.manual_seed(17)
        head = nn.Linear(256, 14).cuda()
        counts = torch.bincount(y['train'], minlength=14).float()
        weight = (counts.sum() / counts.clamp_min(1)).div(14)
        optimizer = torch.optim.AdamW(head.parameters(), lr=.01, weight_decay=.01)
        best, history = -1., []
        for epoch in range(1, 21):
            permutation = torch.randperm(len(x['train']), device='cuda')
            head.train()
            for ids in permutation.split(4096):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.cross_entropy(head(x['train'][ids]), y['train'][ids], weight=weight)
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                pred = head(x['validation']).argmax(-1).cpu().numpy()
            metric = classification_metrics(confusion(arrays['validation'][1], pred))
            history.append(dict(epoch=epoch, validation_macro_f1=metric['macro_f1']))
            if metric['macro_f1'] > best:
                best = metric['macro_f1']
                torch.save(dict(head={k: v.cpu().clone() for k, v in head.state_dict().items()},
                                mean=torch.from_numpy(mean), std=torch.from_numpy(std), epoch=epoch,
                                feature=key, training_subjects=['subject_02','subject_03','subject_04'],
                                selection_subjects=['subject_05']), root / (key + '-probe.pt'))
        saved = torch.load(root / (key + '-probe.pt'), map_location='cpu', weights_only=True)
        head.load_state_dict(saved['head'])
        split_results = {}
        for split in ('validation', 'test'):
            with torch.no_grad():
                pred = head(x[split]).argmax(-1).cpu().numpy()
            labels, short, stable = arrays[split][1:]
            metric = classification_metrics(confusion(labels, pred))
            metric['bed_macro_f1'] = float(np.mean([metric['per_class'][i]['f1'] for i in BED_CLASSES]))
            short_metric = classification_metrics(confusion(labels, pred, short))
            supported = [row for row in short_metric['per_class'] if row['support']]
            metric['short_macro_recall_present_classes'] = float(np.mean([row['recall'] for row in supported])) if supported else None
            metric['short_frame_support'] = short_metric['frames']
            metric['short_per_class'] = short_metric['per_class']
            metric['steady'] = classification_metrics(confusion(labels, pred, stable))
            split_results[split] = metric
            np.savez(root / f'{key}-probe-{split}-predictions.npz', labels=labels, predictions=pred, short=short, steady=stable)
        result[key] = dict(selected_epoch=saved['epoch'], history=history, results=split_results,
                           train_frames=len(x['train']), standardization='train only',
                           bed_classes=BED_CLASSES, short_definition='full GT labeled run <2 seconds')
        dump(root / 'linear-probes.json', result)
        print(json.dumps(dict(stage='probe', feature=key, validation_f1=best,
                              test_f1=split_results['test']['macro_f1'])), flush=True)


@torch.no_grad()
def smoothing(args):
    root, settings = init(args)
    sets, phrases = datasets(settings, ('validation', 'test'))
    saved = read_checkpoint(Path(args.pilot) / 'stage1/best.pt')
    model = RadarPretrainer(RadarTextConfig(**saved['config']), 14).cuda().eval()
    load_weights(Path(args.pilot) / 'stage1/best.pt', model)
    windows, outputs = {}, {}
    for split, dataset in sets.items():
        rows, predictions, timestamps, labels_all = [], [], [], []
        for i in range(len(dataset)):
            sample = dataset[i]
            rid, start, stop, _, actions = dataset.windows[i]
            full = dataset._cached_recording
            meta = dataset.recordings.rows[rid]
            batch = collate_windows([sample])
            logits = model(batch['rd'].cuda(), batch['md'].cuda(), batch['lengths'].cuda())[0].cpu().numpy()
            times = sample['timestamps'].numpy()
            pred = logits.argmax(-1)
            rows.append(dict(sample_id=sample['sample_id'], subject=meta['subject_id'],
                             sensor=meta['sensor_id'], environment=meta['environment_id'], recording=rid,
                             session=meta['session_id'], start=start, stop=stop,
                             duration=float(times[-1] - times[0] + period(times)),
                             period=period(times), actions=actions,
                             instances=action_instances(full.labels.numpy(), full.timestamps.numpy(), start, stop)))
            predictions.append(logits)
            timestamps.append(times)
            labels_all.append(sample['labels'].numpy())
            if (i + 1) % 100 == 0:
                print(json.dumps(dict(stage='caption-logits', split=split, windows=i + 1)), flush=True)
        offsets = np.r_[0, np.cumsum([len(x) for x in predictions])]
        np.savez(root / f'caption-logits-{split}.npz', logits=np.concatenate(predictions),
                 timestamps=np.concatenate(timestamps), labels=np.concatenate(labels_all), offsets=offsets)
        dump(root / f'caption-windows-{split}.json', rows)
        windows[split] = rows
        outputs[split] = [(x.argmax(-1), t) for x, t in zip(predictions, timestamps)]
    candidates = []
    for median in (0., .3, .5):
        for minimum in (0., .3, .6):
            scores = [score_caption(render(smooth_predictions(pred, time, median, minimum), phrases), row['actions'], phrases)
                      for row, (pred, time) in zip(windows['validation'], outputs['validation'])]
            candidates.append(dict(median_seconds=median, minimum_seconds=minimum, validation=means(scores)))
    chosen = max(candidates, key=lambda row: (row['validation']['sequence_exact'], row['validation']['order_lcs'],
                                              row['validation']['action_recall'], -row['median_seconds'], -row['minimum_seconds']))
    result = dict(candidates=candidates, selected=chosen, selection_split='validation',
                  note='minimum-duration removes predicted runs, then merges equal neighbors; no GT-count constraint')
    for split in ('validation','test'):
        scores = []
        for row, (pred, time) in zip(windows[split], outputs[split]):
            text = render(smooth_predictions(pred, time, chosen['median_seconds'], chosen['minimum_seconds']), phrases)
            scores.append(dict(sample_id=row['sample_id'], generated=text, **score_caption(text, row['actions'], phrases)))
        result[split] = dict(metrics=means(scores), results=scores)
    dump(root / 'smoothed-baseline.json', result)


def misses(args):
    root, _ = init(args)
    output = {}
    for split in ('validation', 'test'):
        original = json.loads((Path(args.pilot) / f'{split}-evaluation.json').read_text())['results']
        by_id = {row['sample_id']: row for row in original}
        windows = json.loads((root / f'caption-windows-{split}.json').read_text())
        instances = []
        for row in windows:
            prediction = by_id[row['sample_id']]
            for instance in attribute_instances(row['instances'], prediction['predicted_actions']):
                duration = instance['duration']
                center = (instance['relative_start'] + instance['relative_end']) / 2
                instances.append(dict(instance, sample_id=row['sample_id'], subject=row['subject'],
                                      sensor=row['sensor'], environment=row['environment'],
                                      action_count=len(row['actions']),
                                      duration_bin='<1' if duration < 1 else '1-2' if duration < 2 else '2-4' if duration < 4 else '>=4',
                                      position_bin='begin' if center < 1/3 else 'end' if center >= 2/3 else 'middle'))
        grouped = {}
        for key in ('duration_bin','position_bin','truncation','action_count','class_id','subject','sensor','environment','at_window_edge'):
            groups = {}
            for value in sorted({str(row[key]) for row in instances}):
                group = [row for row in instances if str(row[key]) == value]
                missing = sum(row['missed_weight'] for row in group)
                groups[value] = dict(instances=len(group), missed_weight=missing, miss_rate=missing / len(group),
                                     identity_ambiguous=sum(row['identity_ambiguous'] for row in group))
            grouped[key] = groups
        # Include every category and every requested bucket, even when support is zero.
        for key, keys in [('class_id',map(str,range(14))), ('duration_bin',('<1','1-2','2-4','>=4')),
                          ('position_bin',('begin','middle','end')), ('truncation',('complete','left','right','both')),
                          ('action_count',('1','2','3'))]:
            for value in keys:
                grouped[key].setdefault(value,dict(instances=0,missed_weight=0,miss_rate=None,identity_ambiguous=0))
        deficit = sum(max(0, len(row['reference_actions']) - len(row['predicted_actions'])) for row in original)
        queue = []
        wrong = [row for row in original if not row['sequence_exact']]
        for i in np.linspace(0,len(wrong)-1,min(40,len(wrong)),dtype=int):
            row = wrong[int(i)]
            queue.append(dict(sample_id=row['sample_id'], reference=row['reference'], generated=row['generated'],
                              reference_actions=row['reference_actions'], predicted_actions=row['predicted_actions'],
                              temporal_substitution='unresolved without timestamp-aligned output',
                              review_status='pending'))
        dump(root / f'miss-instances-{split}.json', instances)
        output[split] = dict(instances=len(instances), missed_weight=sum(row['missed_weight'] for row in instances),
                             count_deficit_only=deficit, grouped=grouped, manual_review_queue=queue,
                             attribution='class-occurrence absence; repeated-class miss weight uniformly shared, no temporal substitution claim')
    dump(root / 'miss-attribution.json', output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['extract','probes','smoothing','misses'])
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    globals()[args.stage](args)


if __name__ == '__main__':
    main()
