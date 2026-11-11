"""Bounded fit-only teacher-relation experiment; original pilot remains unchanged."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_mmwave.parrad_projection_pilot import (
    ProjectionAdaptation, balanced_candidate_weights, candidate_identity, choose_checkpoint,
    relational_features, retention_gate, role_masks,
)
from continuous_mmwave.parrad_projection_data import load_rehearsal, load_transition_rows
from continuous_mmwave.parrad_projection_runner import (
    PROJECTION_SEEDS, _fingerprint, _old_projection, dev_probes, project,
    save_pt, semantic_complete, semantic_evaluation, write_json,
)
from continuous_mmwave.parrad_stage3_linear_probe import binary_metrics
from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source, select_endpoints, sha256


SETTINGS = dict(held_subject='subject_02', seed=1701, warmup_epochs=1, joint_epochs=3,
                projection_lr=1e-5, head_lr=1e-3, projection_decay=0., head_decay=1e-4,
                candidate_batch=16, gradient_clip=1., lambda_anchor=1., consistency_weight=.1,
                conditions=[['head-only', 0.], ['joint-relation1', 1.], ['joint-relation10', 10.]])


def relation_terms(new, old, *, temperature):
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError('positive finite temperature required')
    old = {k: v.detach() for k, v in old.items()}
    errors = []
    for branch in ('stream', 'commit'):
        target = old[branch] @ old['text'].T / temperature
        for event, text in ((new[branch], new['text']), (new[branch], old['text']),
                            (old[branch], new['text'])):
            errors.append((event @ text.T / temperature - target).square().mean())
    consistency = ((new['stream'] * new['commit']).sum(-1)
                   - (old['stream'] * old['commit']).sum(-1)).square().mean()
    return dict(relation=torch.stack(errors).mean(), consistency=consistency)


def project_bank(projection, bank):
    result = {}
    for name in ('stream', 'commit', 'text'):
        hidden = bank['hidden'][name]
        result[name] = torch.cat([
            projection(hidden[i:i+128], bank['mask'][i:i+128] if name == 'text' else None)
            for i in range(0, len(hidden), 128)])
    return result


def make_bank(teacher, cache, held):
    rows = cache['rows']
    if not rows or not role_masks(rows, held)['fit'].all():
        raise ValueError('relation bank must contain fit rows only')
    tensors = cache['tensors']
    bank = dict(hidden={k: tensors[k].detach() for k in ('stream', 'commit', 'text')},
                mask=tensors['mask'].detach(), temperature=float(cache['manifest']['temperature']))
    with torch.no_grad():
        bank['teacher_keys'] = project_bank(teacher, bank)
    bank['identity'] = _fingerprint(dict(rows=rows, manifest=cache['manifest']))
    return bank


def transition_loss(model, current, reference, target, weights):
    if weights.shape != target.shape or not (weights >= 0).all() or weights.sum() <= 0:
        raise ValueError('positive Candidate weights required')
    c, a = model.projection(current), model.projection(reference)
    losses = []
    for i, head in enumerate(model.heads):
        x = relational_features(c, a, seed=model.seeds[i], scale=model.scales[i])
        logits = head((x-model.means[i])/model.stds[i]).flatten()
        losses.append((F.binary_cross_entropy_with_logits(logits, target, reduction='none')*weights).sum()
                      / weights.sum())
    return torch.stack(losses).mean()


class RelationalAdaptation(ProjectionAdaptation):
    def relation_loss(self, current, reference, target, weights, bank, *, lambda_relation):
        transition = transition_loss(self, current, reference, target, weights)
        new = project_bank(self.projection, bank)
        old = bank['teacher_keys']
        terms = relation_terms(new, old, temperature=bank['temperature'])
        rehearsal_anchor = torch.stack([.5*(new[k]-old[k]).square().sum(-1).mean()
                                        for k in new]).mean()
        with torch.no_grad():
            old_c, old_a = self.teacher(current), self.teacher(reference)
        pair_anchor = .25*((self.projection(current)-old_c).square().sum(-1).mean()
                           + (self.projection(reference)-old_a).square().sum(-1).mean())
        anchor = (rehearsal_anchor+pair_anchor)/2
        parts = dict(transition=transition, **terms, anchor=anchor)
        loss = transition + lambda_relation*(terms['relation']+.1*terms['consistency']) + anchor
        return loss, parts


def make_optimizer(model, *, joint):
    groups = [dict(params=model.heads.parameters(), lr=SETTINGS['head_lr'],
                   weight_decay=SETTINGS['head_decay'])]
    if joint:
        groups.append(dict(params=model.projection.parameters(), lr=SETTINGS['projection_lr'], weight_decay=0.))
    return torch.optim.AdamW(groups)


def _context(old, data, held, seed):
    masks = role_masks(data['rows'], held)
    if masks['held'].any() or masks['calibration'].any():
        raise ValueError('experiment requires fit/dev-only numeric inputs')
    fit_indices = np.flatnonzero(masks['fit'])
    fit_rows = [data['rows'][i] for i in fit_indices]
    weights, report = balanced_candidate_weights(fit_rows)
    groups = defaultdict(list)
    for i, row in enumerate(fit_rows):
        if weights[i] > 0:
            groups[candidate_identity(row)].append(i)
    if not groups:
        raise ValueError('no fitting Candidates')
    c, a = project(old, data['current']), project(old, data['reference'])
    indices, w = fit_indices[weights > 0], weights[weights > 0]
    scale = max(float(np.sqrt(np.average((c[indices].double()-a[indices].double()).square().sum(1), weights=w))), 1e-6)
    means, stds = [], []
    for s in PROJECTION_SEEDS:
        x = relational_features(c, a, seed=s, scale=scale).numpy()[indices]
        mean = np.average(x, axis=0, weights=w)
        means.append(mean)
        stds.append(np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=w)), 1e-6))
    torch.manual_seed(seed)
    model = RelationalAdaptation(old, PROJECTION_SEEDS, [scale]*len(PROJECTION_SEEDS),
                                torch.tensor(np.array(means), dtype=torch.float32),
                                torch.tensor(np.array(stds), dtype=torch.float32))
    return model, dict(fit_indices=fit_indices, weights=weights, groups=groups,
                       targets=torch.tensor([r['update_target'] for r in data['rows']], dtype=torch.float32),
                       weight_report=report)


def _gradient_stats(parts, parameters):
    vectors = {}
    for name, value in parts.items():
        gradients = torch.autograd.grad(value, parameters, retain_graph=True, allow_unused=True)
        vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                   for p, g in zip(parameters, gradients)])
    return {k: float(v.norm()) for k, v in vectors.items()}


def audit_warmup(model, old, data, ctx):
    for branch in (model.projection, model.teacher):
        if not all(torch.equal(v, branch.state_dict()[k]) for k, v in old.state_dict().items()):
            raise ValueError('warmup changed frozen weights')
    parameters = list(model.projection.parameters())
    previous = [p.requires_grad for p in parameters]
    local = np.flatnonzero(ctx['weights'] > 0)[:128]
    ids = ctx['fit_indices'][local]
    try:
        model.projection.requires_grad_(True)
        loss = transition_loss(model, data['current'][ids], data['reference'][ids], ctx['targets'][ids],
                               torch.tensor(ctx['weights'][local], dtype=torch.float32))
        gradients = torch.autograd.grad(loss, parameters)
        norm = float(torch.sqrt(sum(g.square().sum() for g in gradients)))
    finally:
        for parameter, flag in zip(parameters, previous):
            parameter.requires_grad_(flag)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError('warmup did not produce finite nonzero Projection task gradient')
    return dict(projection_unchanged=True, teacher_unchanged=True, fit_task_gradient_norm=norm,
                checked_fit_rows=len(ids))


def train_epoch(model, optimizer, data, ctx, bank, rng, *, joint, lambda_relation):
    names = list(ctx['groups'])
    order = rng.permutation(len(names))
    stats, first = [], None
    model.train(); model.teacher.eval()
    for start in range(0, len(order), SETTINGS['candidate_batch']):
        local = [j for k in order[start:start+SETTINGS['candidate_batch']] for j in ctx['groups'][names[k]]]
        ids = ctx['fit_indices'][local]
        current, reference = data['current'][ids], data['reference'][ids]
        target, weights = ctx['targets'][ids], torch.tensor(ctx['weights'][local], dtype=torch.float32)
        if joint:
            loss, parts = model.relation_loss(current, reference, target, weights, bank,
                                               lambda_relation=lambda_relation)
            if first is None:
                first = _gradient_stats(parts, list(model.projection.parameters()))
        else:
            loss = transition_loss(model, current, reference, target, weights)
            parts = dict(transition=loss)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite relation objective')
        optimizer.zero_grad(); loss.backward()
        active = [p for p in model.parameters() if p.requires_grad]
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in active):
            raise RuntimeError('nonfinite relation gradient')
        nn.utils.clip_grad_norm_(active, SETTINGS['gradient_clip']); optimizer.step()
        stats.append({k: float(v.detach()) for k, v in parts.items()})
    return dict(components={k: statistics.mean(s[k] for s in stats) for k in stats[0]},
                optimizer_steps=len(stats), first_projection_gradient_norms=first)


@torch.no_grad()
def head_metrics(model, data, held):
    masks = role_masks(data['rows'], held)
    selected = masks['dev'] & np.isin([r['age_updates'] for r in data['rows']], [6, 7])
    result = []
    ids = np.flatnonzero(selected)
    c, a = project(model.projection, data['current'][ids]), project(model.projection, data['reference'][ids])
    targets = [data['rows'][i]['update_target'] for i in ids]
    for i, head in enumerate(model.heads):
        x = relational_features(c, a, seed=model.seeds[i], scale=model.scales[i])
        scores = head((x-model.means[i])/model.stds[i]).sigmoid().flatten().tolist()
        result.append(dict(seed=model.seeds[i], metrics=binary_metrics(targets, scores)))
    return dict(role='diagnostic pooled dev only; not checkpoint selection', records=result)


def _evaluate(model, data, caches, held, baseline, output, epoch):
    model.eval()
    dev = dev_probes(model.projection, data, held)
    diagnostics = []
    retention = semantic_evaluation(model.projection, model.teacher, caches['fit'], caches['dev'], diagnostics=diagnostics)
    write_json(output/f'probe-diagnostics-{epoch:02d}.json', diagnostics)
    return dict(epoch=epoch, score=dev['score'], dev=dev, retention=retention,
                semantic_complete=semantic_complete(retention),
                eligible=dev['score'] is not None and retention_gate(baseline, retention),
                temporary_head_dev=head_metrics(model, data, held))


def run_condition(warm, ctx, bank, data, caches, root, *, held, name, lambda_relation,
                  joint_epochs, baseline_history, identity, rng_state, resume=False):
    joint = name != 'head-only'
    root = Path(root)
    if root.exists() and not resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=resume)
    model = copy.deepcopy(warm)
    model.projection.requires_grad_(joint)
    optimizer = make_optimizer(model, joint=joint)
    rng = np.random.default_rng(); rng.bit_generator.state = copy.deepcopy(rng_state)
    history = copy.deepcopy(baseline_history)
    signature = dict(identity=identity, name=name, lambda_relation=lambda_relation, joint_epochs=joint_epochs)
    first, pending = 2, None
    if resume and (root/'last.pt').exists():
        saved = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('condition resume provenance mismatch')
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        rng.bit_generator.state = saved['rng_state']['numpy']; torch.set_rng_state(saved['rng_state']['torch'])
        history = saved['history']
        pending = saved['pending_evaluation'] if not saved['evaluation_complete'] else None
        first = saved['epoch'] if pending else saved['epoch']+1
    else:
        save_pt(root/'fork.pt', dict(model_state=model.state_dict(), signature=signature, rng_state=rng_state))
        save_pt(root/'epoch-00.pt', dict(projection_state=warm.teacher.state_dict(), epoch=0, identity=identity))
    for epoch in range(first, joint_epochs+2):
        started = time.monotonic()
        summary = pending or train_epoch(model, optimizer, data, ctx, bank, rng, joint=joint,
                                        lambda_relation=lambda_relation)
        pending = None
        def persist(complete):
            save_pt(root/'last.pt', dict(signature=signature, epoch=epoch, model_state=model.state_dict(),
                                        optimizer_state=optimizer.state_dict(), history=history,
                                        evaluation_complete=complete, pending_evaluation=None if complete else summary,
                                        rng_state=dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state())))
        save_pt(root/f'epoch-{epoch:02d}.pt', dict(projection_state=model.projection.state_dict(),
                                                 epoch=epoch, identity=identity))
        persist(False)
        record = _evaluate(model, data, caches, held, baseline_history[0]['retention'], root, epoch)
        change = max(float((v-model.teacher.state_dict()[k]).abs().max()) for k, v in model.projection.state_dict().items())
        record.update(**summary, projection_max_parameter_change=change, seconds=time.monotonic()-started)
        history.append(record)
        persist(True); write_json(root/'history.json', history)
        print(json.dumps(dict(phase='relational_train', condition=name, epoch=epoch,
                              score=record['score'], eligible=record['eligible'], seconds=record['seconds'])), flush=True)
    teacher_unchanged = all(torch.equal(v, model.teacher.state_dict()[k]) for k, v in warm.teacher.state_dict().items())
    projection_unchanged = all(torch.equal(v, model.projection.state_dict()[k]) for k, v in warm.teacher.state_dict().items())
    if not teacher_unchanged or (not joint and not projection_unchanged):
        raise RuntimeError('frozen weights changed')
    selected = choose_checkpoint(history)
    result = dict(name=name, status='locked', selected_epoch=selected,
                  selected_score=history[selected]['score'], lambda_relation=lambda_relation,
                  improved_dev=selected > 1 and history[selected]['score'] > history[0]['score'],
                  selected_checkpoint=str((root/f'epoch-{selected:02d}.pt').resolve()),
                  selected_sha256=sha256(root/f'epoch-{selected:02d}.pt'),
                  teacher_unchanged=teacher_unchanged, projection_unchanged=projection_unchanged,
                  signature=signature, total_epochs=1+joint_epochs)
    write_json(root/'selection-lock.json', result)
    return result


def train_small_experiment(old, data, caches, output, *, held, seed, joint_epochs=3, provenance, resume=False):
    torch.set_num_threads(1)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and not (output/'common-warmup.pt').exists():
        raise ValueError('incomplete warmup cannot resume; preserve evidence and use a new output')
    output.mkdir(parents=True, exist_ok=resume)
    model, ctx = _context(old, data, held, seed)
    bank = make_bank(model.teacher, caches['fit'], held)
    identity = dict(provenance=provenance, settings=SETTINGS, held_subject=held, seed=seed,
                    joint_epochs=joint_epochs, row_fingerprint=_fingerprint(data['rows']), bank_identity=bank['identity'],
                    dev_cache_identity=_fingerprint(dict(rows=caches['dev']['rows'], manifest=caches['dev']['manifest'])))
    rng = np.random.default_rng(seed)
    if resume and (output/'common-warmup.pt').exists():
        warm = torch.load(output/'common-warmup.pt', map_location='cpu', weights_only=False)
        if warm['identity'] != identity:
            raise ValueError('warmup resume provenance mismatch')
        model.load_state_dict(warm['model_state']); rng.bit_generator.state = warm['rng_state']['numpy']
        torch.set_rng_state(warm['rng_state']['torch']); history = warm['history']
    else:
        terms = relation_terms(project_bank(model.projection, bank), bank['teacher_keys'], temperature=bank['temperature'])
        gradients = torch.autograd.grad(sum(terms.values()), list(model.projection.parameters()))
        preflight = dict(losses={k: float(v.detach()) for k, v in terms.items()},
                         max_gradient=max(float(g.abs().max()) for g in gradients))
        write_json(output/'preflight.json', preflight)
        if max(preflight['losses'].values()) > 1e-10 or preflight['max_gradient'] > 1e-7:
            raise ValueError('teacher-equality relation objective is not stationary')
        baseline = _evaluate(model, data, caches, held, {}, output, 0)
        if not semantic_complete(baseline['retention']) or baseline['score'] is None:
            raise ValueError('baseline required dev/semantic metrics undefined')
        baseline['eligible'] = True
        history = [baseline]
        model.projection.requires_grad_(False)
        opt = make_optimizer(model, joint=False)
        summary = train_epoch(model, opt, data, ctx, bank, rng, joint=False, lambda_relation=0.)
        warmup_checks = audit_warmup(model, old, data, ctx)
        write_json(output/'warmup-checks.json', warmup_checks)
        warm_record = copy.deepcopy(baseline)
        warm_record.update(epoch=1, **summary, temporary_head_dev=head_metrics(model, data, held))
        history.append(warm_record)
        save_pt(output/'common-warmup.pt', dict(identity=identity, model_state=model.state_dict(),
                                              optimizer_state=opt.state_dict(), history=history,
                                              rng_state=dict(numpy=rng.bit_generator.state, torch=torch.get_rng_state())))
        write_json(output/'common-warmup-history.json', history)
    results = []
    for name, coefficient in SETTINGS['conditions']:
        results.append(run_condition(model, ctx, bank, data, caches, output/name, held=held, name=name,
                                     lambda_relation=coefficient, joint_epochs=joint_epochs, baseline_history=history,
                                     identity=identity, rng_state=rng.bit_generator.state,
                                     resume=resume and (output/name).exists()))
        write_json(output/'conditions.json', results)
    return results


def run(source, parent, output, *, resume=False):
    import yaml
    source, parent, output = map(Path, (source, parent, output))
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and not (output/'experiment'/'common-warmup.pt').exists():
        raise ValueError('incomplete warmup cannot resume; preserve evidence and use a new output')
    checked = load_sequence_source(source)
    upstream = checked['manifest']; checkpoint = Path(upstream['stage2_checkpoint'])
    config = Path(upstream['config']); raw = yaml.safe_load(config.read_text())
    held, seed = SETTINGS['held_subject'], SETTINGS['seed']
    original = json.loads((parent/'manifest.json').read_text())
    if (original['provenance']['source_manifest_sha256'] != checked['manifest_sha256'] or
        original['provenance']['stage2_checkpoint_sha256'] != sha256(checkpoint)):
        raise ValueError('parent source/checkpoint mismatch')
    spec = Path('docs/stage3-transition-aware-relational-pilot-2026-10-03.md')
    provenance = dict(source_manifest_sha256=checked['manifest_sha256'], stage2_checkpoint_sha256=sha256(checkpoint),
                      parent_manifest_sha256=sha256(parent/'manifest.json'), config_sha256=sha256(config),
                      code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('parrad_projection_*.py')},
                      spec_sha256=sha256(spec), source_split='train', diagnostic_only=True)
    manifest = dict(status='preparing', provenance=provenance, settings=SETTINGS,
                    validation_test_read=False, held_calibration_numeric_read=False,
                    production_model_changed=False, full_pilot_launched=False)
    if resume:
        previous = json.loads((output/'manifest.json').read_text())
        if previous['provenance'] != provenance:
            raise ValueError('run resume provenance mismatch')
    output.mkdir(parents=True, exist_ok=resume); write_json(output/'manifest.json', manifest)
    try:
        old = _old_projection(checkpoint, raw['model']['llm_dim'], raw['model']['semantic_dim'])
        caches = {role: load_rehearsal(parent/held/('semantic-'+role), held, role) for role in ('fit', 'dev')}
        if any(c['manifest']['checkpoint_sha256'] != provenance['stage2_checkpoint_sha256'] for c in caches.values()):
            raise ValueError('relation cache checkpoint mismatch')
        data = load_transition_rows(source, checked, old, held, event_tokens=raw['model']['event_tokens'],
                                    llm_dim=raw['model']['llm_dim'])
        eligible = set()
        for key, rows in checked['rows_by_record'].items():
            for age in (6, 7):
                endpoints, _ = select_endpoints(rows, checked['candidates_by_record'][key], cohort='natural',
                                               endpoint_kind='fixed_age', age=age, history_lengths=(1, 3), task='update')
                eligible.update((e.recording_key, e.endpoint_row) for e in endpoints)
        for row in data['rows']:
            row['probe_eligible'] = (row['recording_key'], row['source_row']) in eligible
        manifest.update(status='training', epoch0_key_error=data['max_old_key_error'],
                        fit_windows=len(caches['fit']['rows']), dev_windows=len(caches['dev']['rows']),
                        text_templates=len(caches['fit']['tensors']['text']))
        write_json(output/'manifest.json', manifest)
        results = train_small_experiment(old, data, caches, output/'experiment', held=held, seed=seed,
                                         joint_epochs=SETTINGS['joint_epochs'], provenance=provenance,
                                         resume=resume and (output/'experiment').exists())
        if load_sequence_source(source)['manifest_sha256'] != checked['manifest_sha256']:
            raise RuntimeError('source changed during experiment')
        if sha256(checkpoint) != provenance['stage2_checkpoint_sha256']:
            raise RuntimeError('Stage2 checkpoint changed during experiment')
        qualified = [r for r in results if r['improved_dev']]
        recommended = max(qualified, key=lambda r: (r['selected_score'], -r['lambda_relation'])) if qualified else None
        manifest.update(status='diagnostics_complete', conditions=results, completed_unix=time.time(),
                        recommended_condition=recommended['name'] if recommended else None,
                        next_action='review before further runs' if any(r['improved_dev'] for r in results)
                        else 'retain epoch0; no expansion')
        write_json(output/'manifest.json', manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'manifest.json', manifest); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', required=True, type=Path)
    parser.add_argument('--parent-dir', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run(args.source_dir, args.parent_dir, args.output_dir, resume=args.resume)
    print(json.dumps(dict(status=result['status'], next_action=result['next_action'])), flush=True)


if __name__ == '__main__':
    main()
