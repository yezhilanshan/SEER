"""Locked-checkpoint paired evaluation; original set003 calibration is unchanged."""
from __future__ import annotations

import argparse
from itertools import product
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from continuous_mmwave import parrad_stage3_complementarity as probe
from continuous_mmwave.parrad_projection_data import load_rehearsal, prepare_rehearsal
from continuous_mmwave.parrad_projection_pilot import retention_gate
from continuous_mmwave.parrad_projection_runner import (
    TRAIN_SEEDS, PROJECTION_SEEDS, _old_projection, semantic_evaluation, semantic_complete, write_json,
)
from continuous_mmwave.parrad_stage3_linear_probe import SUBJECTS
from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source, sha256
from continuous_mmwave.parrad_stage3_sequence_features import iter_history_blocks, features_from_pack

GROUPS = ('natural-fixed_age-4', 'natural-fixed_age-5', 'natural-fixed_age-6',
          'natural-fixed_age-7', 'natural-terminal-min3')
PRIMARY = GROUPS[2:4]
METRICS = ('auroc', 'average_precision', 'recall@0.05', 'false_positive_rate@0.05',
           'recall@0.1', 'false_positive_rate@0.1')


def summarize_decision(pairs, semantic):
    expected = set(product(GROUPS, TRAIN_SEEDS, PROJECTION_SEEDS, SUBJECTS))
    actual = {(r['group'], r['training_seed'], r['seed'], r['held_subject']) for r in pairs}
    semantic_expected = set(product(SUBJECTS, TRAIN_SEEDS))
    semantic_actual = {(r['held_subject'], r['training_seed']) for r in semantic}
    result = dict(decision='inconclusive', production_promotion_allowed=False,
                  interpretation='TRAIN-only adaptation-held subjects; upstream Stage2 saw these subjects',
                  primary={}, reasons=[])
    if (actual != expected or len(pairs) != len(expected) or semantic_actual != semantic_expected
            or len(semantic) != len(semantic_expected)
            or any(r['delta'].get(k) is None or not np.isfinite(r['delta'][k])
                   for r in pairs if r['group'] in PRIMARY for k in METRICS)
            or any(r.get('nonconverged', False) for r in pairs)
            or any(r.get('complete', True) is False for r in semantic)):
        result['reasons'].append('incomplete or undefined paired/semantic matrix')
        return result
    result['undefined_supporting_metrics'] = [
        dict(group=r['group'], training_seed=r['training_seed'], seed=r['seed'], subject=r['held_subject'], metric=k)
        for r in pairs if r['group'] not in PRIMARY for k in METRICS if r['delta'].get(k) is None]
    by_training = {}
    for training_seed in TRAIN_SEEDS:
        by_training[training_seed] = {}
        for metric in METRICS:
            by_training[training_seed][metric] = statistics.median(
                statistics.mean(r['delta'][metric] for r in pairs
                                if r['training_seed'] == training_seed and r['seed'] == projection_seed
                                and r['group'] in PRIMARY) for projection_seed in PROJECTION_SEEDS)
    result['primary'] = {metric: statistics.median(by_training[t][metric] for t in TRAIN_SEEDS)
                         for metric in METRICS}
    result['per_training_seed'] = by_training
    if result['primary']['auroc'] < .02 or result['primary']['average_precision'] < .01:
        result['reasons'].append('insufficient primary AUROC/AP gain')
    passing_seeds = sum(by_training[t]['auroc'] >= .02 and by_training[t]['average_precision'] >= .01 for t in TRAIN_SEEDS)
    if passing_seeds < 2:
        result['reasons'].append('fewer than two training seeds improve reliably')
    if result['primary']['recall@0.05'] < .03 or result['primary']['recall@0.1'] < 0:
        result['reasons'].append('insufficient calibrated recall gain')
    result['per_subject_auroc_delta'] = {}
    for subject in SUBJECTS:
        delta = statistics.median(statistics.median(
            statistics.mean(r['delta']['auroc'] for r in pairs if r['held_subject'] == subject
                            and r['training_seed'] == t and r['seed'] == p and r['group'] in PRIMARY)
            for p in PROJECTION_SEEDS) for t in TRAIN_SEEDS)
        result['per_subject_auroc_delta'][subject] = delta
        if delta <= 0:
            result['reasons'].append(f'{subject} has no primary AUROC gain')
    if any(r['delta']['false_positive_rate@0.05'] > .0100000001 or
           r['delta']['false_positive_rate@0.1'] > .0200000001 for r in pairs if r['group'] in PRIMARY):
        result['reasons'].append('a subject/seed exceeds allowed held FPR deterioration')
    if any(r['delta']['auroc'] < -.0100000001 for r in pairs
           if r['group'] not in PRIMARY and r['delta'].get('auroc') is not None):
        result['reasons'].append('a supporting endpoint regresses')
    if not all(r.get('passed') is True for r in semantic):
        result['reasons'].append('semantic or compatibility retention fails')
    result['decision'] = 'no_go' if result['reasons'] else 'go_research'
    return result


def _merge_rehearsal(first, second):
    a, b = first['tensors'], second['tensors']
    if a['text'].shape[1:] != b['text'].shape[1:]:
        raise ValueError('semantic text padding differs between fit/dev caches')
    tensors = {name: torch.cat((a[name], b[name])) for name in ('stream', 'commit', 'labels', 'text', 'mask')}
    tensors['text_index'] = torch.cat((a['text_index'], b['text_index'] + len(a['text'])))
    return dict(tensors=tensors, rows=first['rows']+second['rows'])


def _new_packs(source, endpoints, projection, event_tokens, llm_dim):
    packs = {}
    for indices, current, reference in iter_history_blocks(source, endpoints, ('F', 'E', 'z'), 1):
        transformed = []
        for values in (current, reference):
            states = torch.from_numpy(values['E'].reshape(-1, event_tokens, llm_dim))
            with torch.no_grad():
                z = projection(states).numpy().reshape(len(indices), 1, 128)
            transformed.append(dict(F=values['F'], z=z))
        for sources in (('z',), ('F', 'z')):
            c, a = [{s: values[s] for s in sources} for values in transformed]
            for seed in PROJECTION_SEEDS:
                block = probe.pack_block(c, a, sources, seed)
                key = sources, seed
                if key not in packs:
                    packs[key] = dict(sources=sources, history=1, blocks={
                        s: {field: np.empty((len(endpoints), *v.shape[1:]), dtype=v.dtype)
                            for field, v in block['blocks'][s].items()} for s in sources})
                for s in sources:
                    for field, values in block['blocks'][s].items():
                        packs[key]['blocks'][s][field][indices] = values
    return packs


def evaluate(source, training, output, *, cache_device='cuda:0'):
    source, training, output = Path(source), Path(training), Path(output)
    if output.exists():
        raise FileExistsError(output)
    trained = json.loads((training/'manifest.json').read_text())
    if trained['status'] != 'training_complete' or trained['smoke']:
        raise ValueError('final evaluation requires the complete fixed-budget real training matrix')
    checked = load_sequence_source(source)
    if checked['manifest_sha256'] != trained['provenance']['source_manifest_sha256']:
        raise ValueError('training/source fingerprint mismatch')
    code = {p.name: sha256(p) for p in Path(__file__).parent.glob('parrad_projection_*.py')}
    if code != trained['provenance']['code_sha256']:
        raise ValueError('runtime code changed after training')
    upstream = checked['manifest']
    import yaml
    config = Path(upstream['config']); raw = yaml.safe_load(config.read_text())
    old = _old_projection(Path(upstream['stage2_checkpoint']), raw['model']['llm_dim'])
    output.mkdir(parents=True); (output/'fits').mkdir(); (output/'cohorts').mkdir()
    manifest = dict(status='evaluating', training_manifest_sha256=sha256(training/'manifest.json'),
                    source_manifest_sha256=checked['manifest_sha256'], source_split='train',
                    validation_test_read=False, completed=[])
    write_json(output/'manifest.json', manifest)
    records, pairs, fusion_pairs, semantic = [], [], [], []
    identifier = 0
    try:
        for subject in SUBJECTS:
            fold = training/subject
            locks = []
            for training_seed in TRAIN_SEEDS:
                path = fold/f'seed-{training_seed}'/'selection-lock.json'
                lock = json.loads(path.read_text())
                if lock['status'] != 'locked' or lock['synthetic'] or lock['held_subject'] != subject:
                    raise ValueError('invalid selected checkpoint fold')
                if sha256(Path(lock['selected_checkpoint'])) != lock['selected_sha256']:
                    raise ValueError('selected checkpoint changed')
                locks.append((training_seed, path, lock))
            # Held inputs are opened only after ALL checkpoints for this fold are locked.
            held_cache = fold/'semantic-held'
            if not held_cache.exists():
                prepare_rehearsal(raw['data']['train_manifest'], held_cache, subject, role='held', config_path=config,
                                  stage2_checkpoint=Path(upstream['stage2_checkpoint']), device=torch.device(cache_device),
                                  selection_lock=locks[0][1])
            cache = load_rehearsal(held_cache, subject, 'held', selection_lock=locks[0][1])
            fit_cache = _merge_rehearsal(load_rehearsal(fold/'semantic-fit', subject, 'fit'),
                                        load_rehearsal(fold/'semantic-dev', subject, 'dev'))
            old_semantic = semantic_evaluation(old, old, fit_cache, cache)
            for training_seed, _, lock in locks:
                new = _old_projection(Path(upstream['stage2_checkpoint']), raw['model']['llm_dim'])
                selected = torch.load(lock['selected_checkpoint'], map_location='cpu', weights_only=False)
                new.load_state_dict(selected['projection_state'], strict=True)
                new_semantic = semantic_evaluation(new, old, fit_cache, cache)
                semantic.append(dict(held_subject=subject, training_seed=training_seed,
                                     old=old_semantic, new=new_semantic,
                                     complete=semantic_complete(old_semantic) and semantic_complete(new_semantic),
                                     passed=retention_gate(old_semantic, new_semantic)))
                for group in (g for g in probe.experiment_grid() if g['name'] in GROUPS):
                    endpoints, exclusions = probe.select_group(checked, group)
                    cohort_rows = [e.metadata for e in endpoints]
                    targets = np.array([r['update_target'] for r in cohort_rows], bool)
                    cohort_name = f"{subject}-{training_seed}-{group['name']}"
                    write_json(output/'cohorts'/f'{cohort_name}.json', dict(
                        exclusions=exclusions, endpoints=[dict(metadata=e.metadata, history_rows=e.history_rows,
                                                              endpoint_row=e.endpoint_row) for e in endpoints]))
                    old_packs = probe.prepare(source, endpoints, 1)
                    new_packs = _new_packs(source, endpoints, new, raw['model']['event_tokens'], raw['model']['llm_dim'])
                    lookup = {}
                    for family, sources, packs in (
                        ('F', ('F',), old_packs), ('E', ('E',), old_packs),
                        ('z_old', ('z',), old_packs), ('z_new', ('z',), new_packs),
                        ('F+z_old', ('F', 'z'), old_packs), ('F+z_new', ('F', 'z'), new_packs)):
                        for seed in PROJECTION_SEEDS:
                            pack = packs.get((sources, seed))
                            fit, _, _ = probe.fold_masks(cohort_rows, subject)
                            scales = probe.source_scales(pack, fit)
                            features = features_from_pack(pack, scales) if scales else np.empty((len(cohort_rows), 130))
                            identifier += 1
                            descriptor = dict(group=group['name'], source=family, history=1, seed=seed,
                                              held_subject=subject, training_seed=training_seed)
                            record = probe.fit_record(output, f'{identifier:06d}', descriptor, cohort_rows, targets, features, scales)
                            if record['converged'] is not True:
                                record['metrics'] = {**record['metrics'], 'auroc': None, 'average_precision': None}
                            lookup[family, seed] = record; records.append(record)
                    for seed in PROJECTION_SEEDS:
                        descriptor = dict(group=group['name'], seed=seed, held_subject=subject, training_seed=training_seed)
                        pairs.append({**descriptor, 'delta': probe.paired_delta(lookup['z_old', seed], lookup['z_new', seed]),
                                      'nonconverged': any(lookup[family, seed]['converged'] is False
                                                          for family in ('z_old', 'z_new')),
                                      'counts': lookup['z_new', seed]['role_counts']})
                        fusion_pairs.append({**descriptor, 'comparison': 'F+z_new-F',
                                             'delta': probe.paired_delta(lookup['F', seed], lookup['F+z_new', seed])})
                        fusion_pairs.append({**descriptor, 'comparison': 'F+z_new-F+z_old',
                                             'delta': probe.paired_delta(lookup['F+z_old', seed], lookup['F+z_new', seed])})
                    manifest['completed'].append(dict(held_subject=subject, training_seed=training_seed, group=group['name']))
                    write_json(output/'manifest.json', manifest)
                    write_json(output/'results.json', records); write_json(output/'paired.json', pairs)
                    write_json(output/'semantic.json', semantic); write_json(output/'fusion-paired.json', fusion_pairs)
                    print(json.dumps(manifest['completed'][-1]), flush=True)
        summary = summarize_decision(pairs, semantic)
        write_json(output/'decision.json', summary)
        final = load_sequence_source(source)
        if final['manifest_sha256'] != checked['manifest_sha256']:
            raise RuntimeError('source changed during final evaluation')
        manifest.update(status='complete', completed_unix=time.time(), record_count=len(records),
                        decision=summary['decision'], decision_sha256=sha256(output/'decision.json'))
        write_json(output/'manifest.json', manifest)
        return summary
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'manifest.json', manifest); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--training-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-device', default='cuda:0')
    args = parser.parse_args()
    print(json.dumps(evaluate(args.source_dir, args.training_dir, args.output_dir, cache_device=args.cache_device)))


if __name__ == '__main__':
    main()
