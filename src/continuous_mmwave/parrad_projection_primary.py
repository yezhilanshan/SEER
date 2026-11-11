"""Bounded primary-age alternating head fit / teacher-relation Projection pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_mmwave.parrad_projection_pilot import candidate_identity, relational_features, retention_gate, role_masks
from continuous_mmwave.parrad_projection_relational import _context, make_bank, project_bank, relation_terms
from continuous_mmwave.parrad_projection_data import load_rehearsal, load_transition_rows
from continuous_mmwave.parrad_projection_runner import (
    _fingerprint, _old_projection, dev_probes, project, save_pt, semantic_complete, semantic_evaluation, write_json,
)
from continuous_mmwave.parrad_stage3_linear_probe import binary_metrics
from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source, select_endpoints, sha256


SETTINGS = dict(held_subject='subject_02', seed=1701, steps=20, milestones=[5, 10, 20],
                ridge=.001, head_max_iter=400, tolerance_grad=1e-7, tolerance_change=1e-9,
                projection_lr=1e-5, projection_decay=0., gradient_clip=1.,
                lambda_anchor=1., consistency_weight=.1,
                conditions=[['frozen-control', 0.], ['joint-relation1', 1.], ['joint-relation10', 10.]])


class HeadFitNumericalError(RuntimeError):
    """Determinate solver failure, distinct from a recoverable interruption."""


def validate_budget(steps, milestones, budget_limit):
    """Extended budgets require explicit opt-in; reject them before data access."""
    if type(budget_limit) is not int or budget_limit not in (20, 100):
        raise ValueError('budget_limit must be explicitly 20 or 100')
    if type(steps) is not int or not 1 <= steps <= budget_limit:
        raise ValueError('integer Projection steps must stay within the explicit budget')
    milestones = (5, 10, 20) if milestones is None and steps == 20 else ((steps,) if milestones is None else tuple(milestones))
    if (not milestones or any(type(s) is not int or not 1 <= s <= steps for s in milestones)
            or sorted(set(milestones)) != list(milestones) or milestones[-1] != steps):
        raise ValueError('positive sorted unique milestones must end at the fixed steps budget')
    return milestones


def validate_weighting(weighting):
    if weighting not in ('age-class', 'subject-age-class'):
        raise ValueError('weighting must be age-class or subject-age-class')


def actual_settings(held, seed, steps, milestones, budget_limit, weighting='age-class'):
    validate_weighting(weighting)
    settings = dict(SETTINGS, held_subject=held, seed=seed, steps=steps, milestones=list(milestones))
    if budget_limit != 20:
        settings['budget_limit'] = budget_limit
    # Preserve legacy control signatures; the changed objective must be explicit.
    if weighting != 'age-class':
        settings['weighting'] = weighting
    return settings


def primary_cohort(rows, held, *, weighting='age-class'):
    validate_weighting(weighting)
    masks = role_masks(rows, held)
    ids = np.array([i for i, r in enumerate(rows) if masks['fit'][i] and
                    r.get('cohort') == 'natural' and r.get('probe_eligible') is True and
                    r['age_updates'] in (6, 7)], dtype=np.int64)
    seen = set()
    for i in ids:
        identity = (*candidate_identity(rows[i]), rows[i]['age_updates'])
        if identity in seen:
            raise ValueError('duplicate primary Candidate/age endpoint')
        seen.add(identity)
        if not isinstance(rows[i]['update_target'], (bool, np.bool_)):
            raise ValueError('primary Boolean target required')
    weights = np.zeros(len(ids), dtype=np.float64)
    buckets = []
    subject_buckets = []
    subjects = [s for s in ('subject_02', 'subject_03', 'subject_04') if s != held]
    for age in (6, 7):
        for target in (False, True):
            select = np.array([rows[i]['age_updates'] == age and rows[i]['update_target'] == target for i in ids])
            count = int(select.sum())
            if not count:
                raise ValueError('both classes required within each primary age')
            if weighting == 'age-class':
                weights[select] = .25 / count
            else:
                for subject in subjects:
                    cell = select & np.array([rows[i]['subject_id'] == subject for i in ids], dtype=bool)
                    n = int(cell.sum())
                    if not n:
                        raise ValueError(f'missing subject/age/class bucket: {subject}/{age}/{target}')
                    weights[cell] = .125 / n
                    subject_buckets.append(dict(subject=subject, age=age, target=target,
                                                count=n, mass=float(weights[cell].sum())))
            buckets.append(dict(age=age, target=target, count=count, mass=float(weights[select].sum())))
    mass = defaultdict(float)
    for i, w in zip(ids, weights):
        mass[candidate_identity(rows[i])] += float(w)
    report = dict(rows=len(ids), candidates=len(mass), buckets=buckets,
                              candidate_mass_min=min(mass.values()), candidate_mass_max=max(mass.values()),
                              total_mass=float(weights.sum()), normalization='age/class buckets; no Candidate renormalization')
    if weighting == 'subject-age-class':
        report.update(weighting=weighting, subject_buckets=subject_buckets,
                      normalization='subject/age/class buckets; no Candidate renormalization')
    return ids, weights, report


def weighted_bce(logits, target, weights):
    """Weights carry global mass: summing arbitrary chunks equals the full loss."""
    return (F.binary_cross_entropy_with_logits(logits, target.to(logits), reduction='none') * weights.to(logits)).sum()


def fit_linear_head(head, x, y, weights, *, max_iter=400):
    x, y, weights = x.detach().double(), y.detach().double(), weights.detach().double()
    if (x.ndim != 2 or len(x) != len(y) or y.shape != weights.shape or
        not all(torch.isfinite(v).all() for v in (x, y, weights)) or
        not (weights > 0).all() or not torch.isclose(weights.sum(), torch.tensor(1., dtype=torch.float64)) or
        set(y.tolist()) != {0., 1.}):
        raise ValueError('finite full primary fit with unit positive weights and both classes required')
    head.double().requires_grad_(True)
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=max_iter, line_search_fn='strong_wolfe',
                                 tolerance_grad=SETTINGS['tolerance_grad'], tolerance_change=SETTINGS['tolerance_change'])
    def closure():
        optimizer.zero_grad()
        loss = weighted_bce(head(x).flatten(), y, weights) + .5*SETTINGS['ridge']*head.weight.square().sum()
        if not torch.isfinite(loss):
            raise HeadFitNumericalError('nonfinite primary head objective')
        loss.backward()
        return loss
    optimizer.step(closure)
    loss = closure()
    state = optimizer.state[head.weight]
    iterations = int(state.get('n_iter', 0))
    gradient = max(float(p.grad.abs().max()) for p in head.parameters())
    previous = state.get('prev_loss')
    change = abs(float(loss.detach()) - float(previous)) if previous is not None else None
    if not torch.isfinite(loss) or not all(torch.isfinite(p).all() for p in head.parameters()) or not np.isfinite(gradient):
        raise HeadFitNumericalError('nonfinite primary fitted head')
    return dict(iterations=iterations, gradient_inf=gradient, loss_change=change, objective=float(loss.detach()),
                converged=gradient <= SETTINGS['tolerance_grad'] or
                (iterations < max_iter and change is not None and change <= SETTINGS['tolerance_change']), max_iter=max_iter)


def context(old, data, held, seed, *, weighting='age-class'):
    # Validate primary metadata before projecting any numeric input.
    ids, weights, report = primary_cohort(data['rows'], held, weighting=weighting)
    model, previous = _context(old, data, held, seed)
    model.heads.double()
    with torch.no_grad():
        teacher_c = project(model.teacher, data['current'][ids])
        teacher_a = project(model.teacher, data['reference'][ids])
    return model, dict(ids=ids, weights=torch.tensor(weights), targets=previous['targets'][ids].double(),
                       weight_report=report, teacher_c=teacher_c, teacher_a=teacher_a)


def pair_keys(model, data, ctx):
    return tuple(torch.cat([model.projection(data[key][ctx['ids'][start:start+128]])
                            for start in range(0, len(ctx['ids']), 128)]) for key in ('current', 'reference'))


def head_features(model, c, a, i):
    x = relational_features(c, a, seed=model.seeds[i], scale=model.scales[i])
    return ((x-model.means[i])/model.stds[i]).double()


def fit_heads(model, data, ctx):
    with torch.no_grad():
        c, a = pair_keys(model, data, ctx)
    reports = []
    for i, head in enumerate(model.heads):
        try:
            report = fit_linear_head(head, head_features(model, c, a, i), ctx['targets'], ctx['weights'])
            reports.append(dict(seed=model.seeds[i], **report))
        except HeadFitNumericalError as exc:
            reports.append(dict(seed=model.seeds[i], converged=False, objective=None,
                                error=f'{type(exc).__name__}: {exc}'))
            break
        finally:
            head.requires_grad_(False)
    return dict(converged=all(r['converged'] for r in reports),
                objective=float(np.mean([r['objective'] for r in reports])) if all(r['objective'] is not None for r in reports) else None,
                heads=reports,
                constant_balanced_bce=math.log(2), fit_only=True)


def task_loss(model, data, ctx, keys=None):
    c, a = pair_keys(model, data, ctx) if keys is None else keys
    return torch.stack([weighted_bce(head(head_features(model, c, a, i)).flatten(),
                                     ctx['targets'], ctx['weights']) + .5*SETTINGS['ridge']*head.weight.square().sum()
                        for i, head in enumerate(model.heads)]).mean()


def projection_step(model, optimizer, data, ctx, bank, coefficient):
    model.projection.requires_grad_(True); model.heads.requires_grad_(False)
    c, a = pair_keys(model, data, ctx)
    task = task_loss(model, data, ctx, (c, a))
    new, old = project_bank(model.projection, bank), bank['teacher_keys']
    terms = relation_terms(new, old, temperature=bank['temperature'])
    bank_anchor = torch.stack([.5*(new[k]-old[k]).square().sum(-1).mean() for k in new]).mean()
    pair_anchor = .25*((c-ctx['teacher_c']).square().sum(-1).mean() + (a-ctx['teacher_a']).square().sum(-1).mean())
    anchor = (bank_anchor+pair_anchor)/2
    loss = task + coefficient*(terms['relation']+.1*terms['consistency']) + anchor
    if not torch.isfinite(loss):
        raise RuntimeError('nonfinite primary Projection objective')
    optimizer.zero_grad(); loss.backward()
    active = list(model.projection.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in active):
        raise RuntimeError('nonfinite/missing primary Projection gradient')
    norm = float(nn.utils.clip_grad_norm_(active, SETTINGS['gradient_clip']))
    optimizer.step()
    return dict(loss=float(loss.detach()), transition=float(task.detach()),
                relation=float(terms['relation'].detach()), consistency=float(terms['consistency'].detach()),
                anchor=float(anchor.detach()), projection_gradient_norm=norm)


@torch.no_grad()
def head_metrics(model, data, held):
    masks = role_masks(data['rows'], held)
    ids = np.array([i for i, r in enumerate(data['rows']) if masks['dev'][i] and
                    r.get('probe_eligible') is True and r['age_updates'] in (6, 7)])
    c, a = project(model.projection, data['current'][ids]), project(model.projection, data['reference'][ids])
    targets = [data['rows'][i]['update_target'] for i in ids]
    records = [dict(seed=model.seeds[i], metrics=binary_metrics(targets,
               head(head_features(model, c, a, i)).sigmoid().flatten().tolist())) for i, head in enumerate(model.heads)]
    return dict(role='pooled dev diagnostic; not checkpoint selection', records=records)


def evaluate(model, data, caches, held, baseline, output, step):
    model.eval()
    dev = dev_probes(model.projection, data, held)
    diagnostics = []
    metrics = semantic_evaluation(model.projection, model.teacher, caches['fit'], caches['dev'], diagnostics=diagnostics)
    write_json(output/f'probe-diagnostics-{step:02d}.json', diagnostics)
    return dict(step=step, score=dev['score'], dev=dev, retention=metrics, semantic_complete=semantic_complete(metrics),
                eligible=dev['score'] is not None and retention_gate(baseline, metrics),
                temporary_head_dev=head_metrics(model, data, held))


def run_condition(warm, ctx, bank, data, caches, root, *, name, coefficient, held, steps, milestones,
                  baseline, identity, resume=False):
    root = Path(root)
    if root.exists() and not resume:
        raise FileExistsError(root)
    root.mkdir(parents=True, exist_ok=resume)
    model = copy.deepcopy(warm)
    optimizer = torch.optim.AdamW(model.projection.parameters(), lr=SETTINGS['projection_lr'], weight_decay=0.)
    signature = dict(identity=identity, name=name, coefficient=coefficient, steps=steps, milestones=list(milestones))
    step, phase, history, summary, head_report = 0, 'ready', [copy.deepcopy(baseline)], {}, None
    def persist():
        save_pt(root/'last.pt', dict(signature=signature, step=step, phase=phase, model_state=model.state_dict(),
                                    optimizer_state=optimizer.state_dict(), history=history, summary=summary,
                                    head_report=head_report, torch_rng=torch.get_rng_state()))
    if resume and (root/'last.pt').exists():
        saved = torch.load(root/'last.pt', map_location='cpu', weights_only=False)
        if saved['signature'] != signature:
            raise ValueError('primary condition resume provenance mismatch')
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        torch.set_rng_state(saved['torch_rng'])
        step, phase, history, summary, head_report = [saved[k] for k in ('step', 'phase', 'history', 'summary', 'head_report')]
    else:
        save_pt(root/'fork.pt', dict(signature=signature, model_state=model.state_dict()))
        save_pt(root/'step-00.pt', dict(projection_state=model.projection.state_dict(), step=0, identity=identity))
        persist()
    while name != 'frozen-control' and phase != 'stopped':
        if phase == 'ready':
            if step == steps:
                break
            summary = projection_step(model, optimizer, data, ctx, bank, coefficient)
            step += 1; phase = 'head_fit_pending'; persist()
        if phase == 'head_fit_pending':
            head_report = fit_heads(model, data, ctx)
            if not head_report['converged']:
                phase = 'stopped'; persist()
                write_json(root/f'head-fit-{step:02d}.json', head_report)
                print(json.dumps(dict(phase='head_fit_failed', condition=name, step=step)), flush=True)
                break
            phase = 'evaluation_pending' if step in milestones else 'ready'; persist()
            write_json(root/f'head-fit-{step:02d}.json', head_report)
            print(json.dumps(dict(phase='primary_step', condition=name, step=step,
                                  components=summary, head_objective=head_report['objective'])), flush=True)
        if phase == 'evaluation_pending':
            save_pt(root/f'step-{step:02d}.pt', dict(projection_state=model.projection.state_dict(), step=step, identity=identity))
            record = evaluate(model, data, caches, held, baseline['retention'], root, step)
            record.update(components=summary, head_fit=head_report)
            history.append(record); phase = 'ready'; persist()
            print(json.dumps(dict(phase='primary_evaluation', condition=name, step=step,
                                  score=record['score'], eligible=record['eligible'])), flush=True)
        write_json(root/'history.json', history)
    teacher_ok = all(torch.equal(v, model.teacher.state_dict()[k]) for k, v in warm.teacher.state_dict().items())
    p_unchanged = all(torch.equal(v, model.projection.state_dict()[k]) for k, v in warm.teacher.state_dict().items())
    if not teacher_ok or (name == 'frozen-control' and not p_unchanged):
        raise RuntimeError('frozen primary weights changed')
    valid = [r for r in history if r['eligible'] and r['score'] is not None]
    selected = baseline if phase == 'stopped' else max(valid, key=lambda r: (r['score'], -r['step']))
    checkpoint = root/f"step-{selected['step']:02d}.pt"
    result = dict(name=name, status='head_fit_failed' if phase == 'stopped' else 'locked',
                  completed_projection_steps=step, selected_step=selected['step'], selected_score=selected['score'],
                  improved_dev=selected['step'] > 0 and selected['score'] > baseline['score'],
                  selected_checkpoint=str(checkpoint.resolve()), selected_sha256=sha256(checkpoint),
                  lambda_relation=coefficient, teacher_unchanged=teacher_ok, projection_unchanged=p_unchanged,
                  signature=signature)
    write_json(root/'history.json', history); write_json(root/'selection-lock.json', result)
    return result


def train_small_experiment(old, data, caches, output, *, held, seed, steps=20, milestones=(5, 10, 20), provenance,
                           resume=False, budget_limit=20, weighting='age-class'):
    validate_weighting(weighting)
    milestones = validate_budget(steps, milestones, budget_limit)
    torch.set_num_threads(1)
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and not (output/'common-warmup.pt').exists():
        raise ValueError('incomplete warmup cannot resume; preserve evidence and use a new output')
    output.mkdir(parents=True, exist_ok=resume)
    model, ctx = context(old, data, held, seed, weighting=weighting)
    bank = make_bank(model.teacher, caches['fit'], held)
    identity = dict(provenance=provenance, settings=actual_settings(held, seed, steps, milestones, budget_limit, weighting),
                    held=held, seed=seed, steps=steps, milestones=list(milestones),
                    rows=_fingerprint(data['rows']), bank=bank['identity'],
                    dev_cache=_fingerprint(dict(rows=caches['dev']['rows'], manifest=caches['dev']['manifest'])))
    if resume:
        warm = torch.load(output/'common-warmup.pt', map_location='cpu', weights_only=False)
        if warm['identity'] != identity:
            raise ValueError('primary warmup resume provenance mismatch')
        model.load_state_dict(warm['model_state']); torch.set_rng_state(warm['torch_rng']); baseline = warm['baseline']
        model.heads.requires_grad_(False)
    else:
        write_json(output/'weights.json', ctx['weight_report'])
        terms = relation_terms(project_bank(model.projection, bank), bank['teacher_keys'], temperature=bank['temperature'])
        gradients = torch.autograd.grad(sum(terms.values()), list(model.projection.parameters()))
        preflight = dict(losses={k: float(v.detach()) for k, v in terms.items()},
                         max_gradient=max(float(g.abs().max()) for g in gradients))
        write_json(output/'preflight.json', preflight)
        if max(preflight['losses'].values()) > 1e-10 or preflight['max_gradient'] > 1e-7:
            raise ValueError('primary teacher equality not stationary')
        model.projection.requires_grad_(False)
        head_report = fit_heads(model, data, ctx)
        write_json(output/'warmup-heads.json', head_report)
        if not head_report['converged'] or head_report['objective'] >= math.log(2):
            raise ValueError('primary warmup heads unconverged or no improvement over constant baseline')
        for branch in (model.projection, model.teacher):
            if not all(torch.equal(v, branch.state_dict()[k]) for k, v in old.state_dict().items()):
                raise ValueError('primary warmup changed frozen weights')
        model.projection.requires_grad_(True)
        grads = torch.autograd.grad(task_loss(model, data, ctx), list(model.projection.parameters()))
        norm = float(torch.sqrt(sum(g.square().sum() for g in grads)))
        model.projection.requires_grad_(False)
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError('primary warmup missing task gradient')
        write_json(output/'warmup-checks.json', dict(projection_unchanged=True, teacher_unchanged=True, fit_task_gradient_norm=norm))
        baseline = evaluate(model, data, caches, held, {}, output, 0)
        if not semantic_complete(baseline['retention']) or baseline['score'] is None:
            raise ValueError('primary baseline required metrics undefined')
        baseline.update(eligible=True, head_fit=head_report)
        save_pt(output/'common-warmup.pt', dict(identity=identity, model_state=model.state_dict(), baseline=baseline,
                                              torch_rng=torch.get_rng_state()))
        write_json(output/'baseline.json', baseline)
    results = []
    for name, coefficient in SETTINGS['conditions']:
        results.append(run_condition(model, ctx, bank, data, caches, output/name, name=name, coefficient=coefficient,
                                     held=held, steps=steps, milestones=milestones, baseline=baseline, identity=identity,
                                     resume=resume and (output/name).exists()))
        write_json(output/'conditions.json', results)
    return results


def run(source, parent, output, *, resume=False, held='subject_02', seed=1701,
        spec=Path('docs/stage3-transition-aware-primary-pilot-2026-10-03.md'), cache_parent=None,
        steps=20, milestones=None, budget_limit=20, weighting='age-class'):
    import yaml
    validate_weighting(weighting)
    milestones = validate_budget(steps, milestones, budget_limit)
    if held not in ('subject_02', 'subject_03', 'subject_04'):
        raise ValueError('unsupported held subject')
    source, parent, output = map(Path, (source, parent, output))
    cache_parent = parent if cache_parent is None else Path(cache_parent)
    if output.exists() and not resume:
        raise FileExistsError(output)
    if resume and not (output/'experiment'/'common-warmup.pt').exists():
        raise ValueError('incomplete warmup cannot resume; preserve evidence and use a new output')
    checked = load_sequence_source(source)
    upstream = checked['manifest']; checkpoint = Path(upstream['stage2_checkpoint']); config = Path(upstream['config'])
    raw = yaml.safe_load(config.read_text())
    original = json.loads((parent/'manifest.json').read_text())
    if (original['provenance']['source_manifest_sha256'] != checked['manifest_sha256'] or
        original['provenance']['stage2_checkpoint_sha256'] != sha256(checkpoint)):
        raise ValueError('primary parent source/checkpoint mismatch')
    spec = Path(spec)
    provenance = dict(source_manifest_sha256=checked['manifest_sha256'], stage2_checkpoint_sha256=sha256(checkpoint),
                      parent_manifest_sha256=sha256(parent/'manifest.json'), config_sha256=sha256(config),
                      code_sha256={p.name: sha256(p) for p in Path(__file__).parent.glob('parrad_projection_*.py')},
                      spec_sha256=sha256(spec), source_split='train', diagnostic_only=True)
    provenance['cache_parent'] = str(cache_parent.resolve())
    manifest = dict(status='preparing', provenance=provenance,
                    settings=actual_settings(held, seed, steps, milestones, budget_limit, weighting), validation_test_read=False,
                    held_calibration_numeric_read=False, production_model_changed=False, full_pilot_launched=False)
    if resume:
        previous = json.loads((output/'manifest.json').read_text())
        if previous['provenance'] != provenance:
            raise ValueError('primary run resume provenance mismatch')
        if previous['settings'] != manifest['settings']:
            raise ValueError('primary run resume settings mismatch')
    output.mkdir(parents=True, exist_ok=resume); write_json(output/'manifest.json', manifest)
    try:
        old = _old_projection(checkpoint, raw['model']['llm_dim'], raw['model']['semantic_dim'])
        caches = {role: load_rehearsal(cache_parent/held/('semantic-'+role), held, role) for role in ('fit', 'dev')}
        if any(c['manifest']['checkpoint_sha256'] != provenance['stage2_checkpoint_sha256'] for c in caches.values()):
            raise ValueError('primary cache checkpoint mismatch')
        data = load_transition_rows(source, checked, old, held, event_tokens=raw['model']['event_tokens'], llm_dim=raw['model']['llm_dim'])
        eligible = set()
        for key, rows in checked['rows_by_record'].items():
            for age in (6, 7):
                endpoints, _ = select_endpoints(rows, checked['candidates_by_record'][key], cohort='natural',
                    endpoint_kind='fixed_age', age=age, history_lengths=(1, 3), task='update')
                eligible.update((e.recording_key, e.endpoint_row) for e in endpoints)
        for row in data['rows']:
            row['probe_eligible'] = (row['recording_key'], row['source_row']) in eligible
        manifest.update(status='training', epoch0_key_error=data['max_old_key_error'], fit_windows=len(caches['fit']['rows']),
                        dev_windows=len(caches['dev']['rows']), text_templates=len(caches['fit']['tensors']['text']))
        write_json(output/'manifest.json', manifest)
        results = train_small_experiment(old, data, caches, output/'experiment', held=held, seed=seed, provenance=provenance,
                                        resume=resume and (output/'experiment').exists(),
                                        steps=steps, milestones=milestones, budget_limit=budget_limit, weighting=weighting)
        if load_sequence_source(source)['manifest_sha256'] != checked['manifest_sha256'] or sha256(checkpoint) != provenance['stage2_checkpoint_sha256']:
            raise RuntimeError('primary source/Stage2 checkpoint changed')
        qualified = [r for r in results if r['improved_dev']]
        best = max(qualified, key=lambda r: (r['selected_score'], -r['lambda_relation'])) if qualified else None
        manifest.update(status='diagnostics_complete', conditions=results, completed_unix=time.time(),
                        recommended_condition=best['name'] if best else None,
                        next_action='review small-experiment dev results' if best else 'retain step0; no expansion')
        write_json(output/'manifest.json', manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'manifest.json', manifest); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'parent', 'output'):
        parser.add_argument('--'+name+'-dir', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--held-subject', choices=('subject_02', 'subject_03', 'subject_04'), default='subject_02')
    parser.add_argument('--seed', type=int, default=1701)
    parser.add_argument('--spec', type=Path, default=Path('docs/stage3-transition-aware-primary-pilot-2026-10-03.md'))
    parser.add_argument('--cache-parent-dir', type=Path)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--milestones', type=int, nargs='+')
    parser.add_argument('--budget-limit', type=int, choices=(20, 100), default=20)
    parser.add_argument('--weighting', choices=('age-class', 'subject-age-class'), default='age-class')
    args = parser.parse_args()
    result = run(args.source_dir, args.parent_dir, args.output_dir, resume=args.resume,
                 held=args.held_subject, seed=args.seed, spec=args.spec, cache_parent=args.cache_parent_dir,
                 steps=args.steps, milestones=args.milestones, budget_limit=args.budget_limit, weighting=args.weighting)
    print(json.dumps(dict(status=result['status'], next_action=result['next_action'])), flush=True)


if __name__ == '__main__':
    main()
