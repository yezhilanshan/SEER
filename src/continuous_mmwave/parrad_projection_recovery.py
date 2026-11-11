"""Isolated fit/dev-only replay and bounded recovery experiments, never held evaluation."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from continuous_mmwave.parrad_projection_pilot import ProjectionAdaptation, balanced_candidate_weights, candidate_identity, role_masks
from continuous_mmwave.parrad_projection_data import load_transition_rows, load_rehearsal
from continuous_mmwave.parrad_projection_runner import (
    PROJECTION_SEEDS, TRAIN_SETTINGS, _fingerprint, _old_projection, _train_epoch,
    dev_probes, project, save_pt, semantic_evaluation, semantic_complete, train_fold, write_json,
)
from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source, select_endpoints, sha256


def can_expand(result, *, semantic_complete=True):
    return semantic_complete and result['selected_epoch'] > 0 and result['improved_dev'] is True


def replay_next_epoch(old, data, caches, parent_checkpoint, output):
    """Reconstruct the failed optimizer epoch with its exact saved state, without altering v1."""
    torch.set_num_threads(1)
    saved = torch.load(parent_checkpoint, map_location='cpu', weights_only=False)
    identity = saved['identity']; held = identity['held_subject']
    if (identity['row_fingerprint'] != _fingerprint(data['rows']) or
        any(identity['semantic_manifest_fingerprints'][role] != _fingerprint(caches[role]['manifest'])
            or identity[role+'_semantic_fingerprint'] != _fingerprint(caches[role]['rows']) for role in ('fit', 'dev'))):
        raise ValueError('replay data/cache provenance mismatch')
    masks = role_masks(data['rows'], held)
    if masks['held'].any() or masks['calibration'].any():
        raise ValueError('replay must only receive fit/dev inputs')
    fit_indices = np.flatnonzero(masks['fit'])
    fit_rows = [data['rows'][i] for i in fit_indices]
    weights, _ = balanced_candidate_weights(fit_rows)
    groups = defaultdict(list)
    for i, row in enumerate(fit_rows):
        if weights[i] > 0:
            groups[candidate_identity(row)].append(i)
    c, a = project(old, data['current']), project(old, data['reference'])
    indices = fit_indices[weights > 0]
    scale = max(float(np.sqrt(np.average((c[indices].double()-a[indices].double()).square().sum(1),
                                        weights=weights[weights > 0]))), 1e-6)
    model = ProjectionAdaptation(old, PROJECTION_SEEDS, [scale]*3,
                                 saved['model_state']['means'], saved['model_state']['stds'])
    model.load_state_dict(saved['model_state'])
    if not all(torch.equal(v, model.teacher.state_dict()[k]) for k, v in old.state_dict().items()):
        raise ValueError('replay teacher does not match old Projection')
    cfg = identity['settings']
    optimizer = torch.optim.AdamW([dict(params=model.projection.parameters(), lr=cfg['projection_lr']),
                                  dict(params=model.heads.parameters(), lr=cfg['head_lr'])],
                                 weight_decay=cfg['weight_decay'])
    optimizer.load_state_dict(saved['optimizer_state'])
    rng = np.random.default_rng(); rng.bit_generator.state = saved['rng_state']['numpy_generator']
    torch.set_rng_state(saved['rng_state']['torch'])
    output = Path(output); output.mkdir(parents=True)
    summary = _train_epoch(model, optimizer, data, fit_indices, weights, groups, caches['fit'], rng, cfg)
    epoch = saved['epoch']+1
    save_pt(output/'replayed-epoch.pt', dict(epoch=epoch, model_state=model.state_dict(),
                                          parent_checkpoint_sha256=sha256(Path(parent_checkpoint))))
    diagnostics = []
    retention = semantic_evaluation(model.projection.eval(), model.teacher, caches['fit'], caches['dev'], diagnostics=diagnostics)
    report = dict(epoch=epoch, parent_checkpoint=str(parent_checkpoint), parent_sha256=sha256(Path(parent_checkpoint)),
                  summary=summary, diagnostics=diagnostics, retention=retention,
                  semantic_complete=semantic_complete(retention), scientific_result=False)
    write_json(output/'replay-report.json', report)
    print(json.dumps(dict(phase='replayed_failed_epoch', epoch=epoch,
                          nonconverged_probes=sum(not d['converged'] for d in diagnostics))), flush=True)
    return report


def run_recovery(source, parent, output):
    import yaml
    torch.set_num_threads(1)
    source, parent, output = map(Path, (source, parent, output))
    if output.exists():
        raise FileExistsError(output)
    checked = load_sequence_source(source)
    upstream = checked['manifest']; checkpoint = Path(upstream['stage2_checkpoint'])
    raw = yaml.safe_load(Path(upstream['config']).read_text())
    held, seed = 'subject_02', 1701
    old = _old_projection(checkpoint, raw['model']['llm_dim'], raw['model']['semantic_dim'])
    code = {p.name: sha256(p) for p in Path(__file__).parent.glob('parrad_projection_*.py')}
    spec = Path('docs/stage3-transition-aware-projection-recovery-2026-10-03.md')
    original = json.loads((parent/'manifest.json').read_text())
    if (original['provenance']['source_manifest_sha256'] != checked['manifest_sha256'] or
        original['provenance']['stage2_checkpoint_sha256'] != sha256(checkpoint)):
        raise ValueError('parent source/checkpoint mismatch')
    provenance = dict(source_manifest_sha256=checked['manifest_sha256'], stage2_checkpoint_sha256=sha256(checkpoint),
                      parent_manifest_sha256=sha256(parent/'manifest.json'), code_sha256=code,
                      spec_sha256=sha256(spec), source_split='train', diagnostic_only=True)
    manifest = dict(status='preparing', provenance=provenance, held_subject=held, seed=seed,
                    validation_test_read=False, held_calibration_numeric_read=False, production_model_changed=False,
                    conditions=[], budget=dict(rehearsal_only_epochs=1, conservative_epochs=3, anchor_epochs_if_needed=3))
    output.mkdir(parents=True)
    write_json(output/'manifest.json', manifest)
    try:
        caches = {role: load_rehearsal(parent/held/('semantic-'+role), held, role) for role in ('fit', 'dev')}
        if any(c['manifest']['checkpoint_sha256'] != provenance['stage2_checkpoint_sha256'] for c in caches.values()):
            raise ValueError('recovery rehearsal checkpoint mismatch')
        data = load_transition_rows(source, checked, old, held, event_tokens=raw['model']['event_tokens'],
                                    llm_dim=raw['model']['llm_dim'])
        manifest['epoch0_key_error'] = data['max_old_key_error']
        eligible = set()
        for key, rows in checked['rows_by_record'].items():
            for age in (6, 7):
                endpoints, _ = select_endpoints(rows, checked['candidates_by_record'][key], cohort='natural',
                                               endpoint_kind='fixed_age', age=age, history_lengths=(1, 3), task='update')
                eligible.update((e.recording_key, e.endpoint_row) for e in endpoints)
        for row in data['rows']:
            row['probe_eligible'] = (row['recording_key'], row['source_row']) in eligible
        manifest['status'] = 'replaying_failed_epoch'; write_json(output/'manifest.json', manifest)
        replay = replay_next_epoch(old, data, caches, parent/held/f'seed-{seed}'/'last.pt', output/'failed-epoch-replay')
        manifest['replay_nonconverged'] = [d for d in replay['diagnostics'] if not d['converged']]
        conditions = [
            ('rehearsal-only', dict(epochs=1, lambda_t=0., projection_lr=1e-4)),
            ('joint-lr1e-5', dict(epochs=3, projection_lr=1e-5)),
            ('joint-lr1e-5-anchor10', dict(epochs=3, projection_lr=1e-5, lambda_a=10.)),
        ]
        conservative_passed = False
        for name, settings in conditions:
            if name.endswith('anchor10') and conservative_passed:
                break
            manifest.update(status='diagnostic_training', active_condition=name)
            write_json(output/'manifest.json', manifest)
            result = train_fold(old, data, caches, output/name, held_subject=held, seed=seed,
                                epochs=settings['epochs'], provenance={**provenance, 'condition': name}, settings=settings)
            history = json.loads((output/name/'history.json').read_text())
            selected_complete = semantic_complete(history[result['selected_epoch']]['retention'])
            if name != 'rehearsal-only':
                conservative_passed = can_expand(result, semantic_complete=selected_complete)
            manifest['conditions'].append(dict(name=name, settings=settings, selected_epoch=result['selected_epoch'],
                                               improved_dev=result['improved_dev'], eligible_for_expansion=conservative_passed
                                               if name != 'rehearsal-only' else False))
            write_json(output/'manifest.json', manifest)
        final = load_sequence_source(source)
        if final['manifest_sha256'] != checked['manifest_sha256']:
            raise ValueError('source changed during recovery')
        manifest.update(status='diagnostics_complete', completed_unix=time.time(),
                        next_action='review bounded improvement before full pilot' if conservative_passed else 'retain epoch0; no expansion',
                        full_pilot_launched=False, held_evaluation_performed=False)
        write_json(output/'manifest.json', manifest)
        print(json.dumps(dict(status=manifest['status'], next_action=manifest['next_action'])), flush=True)
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'manifest.json', manifest); raise
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--parent-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    run_recovery(args.source_dir, args.parent_dir, args.output_dir)


if __name__ == '__main__':
    main()
