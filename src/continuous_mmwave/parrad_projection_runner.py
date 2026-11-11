"""Fixed-budget Projection-only pilot; selection uses TRAIN-inner-dev, never calibration."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_mmwave.models.semantic import SemanticProjection
from continuous_mmwave.parrad_projection_pilot import (
    ProjectionAdaptation, balanced_candidate_weights, candidate_identity, choose_checkpoint,
    relational_features, retention_gate, role_masks, semantic_metrics,
)
from continuous_mmwave.parrad_projection_data import (
    load_rehearsal, load_transition_rows, prepare_rehearsal, rehearsal_batch,
)
from continuous_mmwave.parrad_stage3_complementarity import SETTINGS as PROBE
from continuous_mmwave.parrad_stage3_linear_probe import SUBJECTS, binary_metrics
from continuous_mmwave.parrad_stage3_radar_readout import fit_readout
from continuous_mmwave.parrad_stage3_sequence_data import load_sequence_source, select_endpoints, sha256

TRAIN_SEEDS = (1701, 1702, 1703)
PROJECTION_SEEDS = tuple(PROBE['projection_seeds'])
TRAIN_SETTINGS = dict(epochs=20, projection_lr=1e-4, head_lr=1e-3, weight_decay=1e-4,
                      gradient_clip=1., candidate_batch=16, rehearsal_batch=32,
                      lambda_s=1., lambda_a=1., device='cpu', training_seeds=TRAIN_SEEDS,
                      projection_seeds=PROJECTION_SEEDS, semantic_per_class_record=4)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def save_pt(path, payload):
    temporary = Path(path).with_name(Path(path).name + '.tmp')
    torch.save(payload, temporary); temporary.replace(path)


def _fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@torch.no_grad()
def project(projection, values):
    result = [projection(values[i:i+128].to(next(projection.parameters()).device)).float().cpu()
              for i in range(0, len(values), 128)]
    if not result:
        raise ValueError('empty projection cohort')
    return torch.cat(result)


def _solver():
    return dict(cpu_threads=1, ridge_lambda=PROBE['ridge_lambda'], tolerance_grad=PROBE['tolerance_grad'],
                tolerance_change=PROBE['tolerance_change'], logistic_max_iter=PROBE['max_iter'], mlp_hidden=0)


def _action_probe(fit_keys, fit_labels, keys, labels, *, max_iter=400, diagnostics=None, context=None):
    """Equal-capacity class-balanced action Logistic, independent of transition heads."""
    classes = sorted(fit_labels.unique().tolist())
    if set(labels.unique().tolist()) != set(classes) or classes != list(range(len(classes))):
        raise ValueError('action probe requires complete matching class coverage')
    x, z = fit_keys.double(), keys.double()
    mean, std = x.mean(0), x.std(0, unbiased=False).clamp_min(1e-6)
    x, z = (x-mean)/std, (z-mean)/std
    model = nn.Linear(x.shape[1], len(classes), dtype=torch.float64)
    with torch.no_grad():
        model.weight.zero_(); model.bias.zero_()
    counts = torch.bincount(fit_labels, minlength=len(classes)).double()
    weights = len(fit_labels)/(len(classes)*counts)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=max_iter, tolerance_grad=1e-7,
                                  tolerance_change=1e-9, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), fit_labels, weight=weights) + .0005*model.weight.square().sum()
        loss.backward(); return loss
    optimizer.step(closure); objective = closure()
    gradient = max(float(p.grad.abs().max()) for p in model.parameters())
    state = optimizer.state[next(model.parameters())]
    change = abs(float(objective.detach())-float(state['prev_loss']))
    finite = bool(torch.isfinite(objective) and torch.isfinite(x).all() and torch.isfinite(z).all()
                  and all(torch.isfinite(p.grad).all() for p in model.parameters()))
    converged = finite and (gradient <= 1e-7 or (state['n_iter'] < max_iter and change <= 1e-9))
    report = dict(context=context, converged=converged, finite=finite,
                  iterations=state['n_iter'], function_evaluations=state['func_evals'], max_iter=max_iter,
                  gradient_max=gradient, objective=float(objective.detach()), objective_change=change,
                  fit_std_min=float(std.min()), fit_std_max=float(std.max()),
                  fit_classes=counts.int().tolist(),
                  fit_fingerprint=hashlib.sha256(fit_keys.numpy().tobytes()+fit_labels.numpy().tobytes()).hexdigest())
    if diagnostics is not None:
        diagnostics.append(report)
    if not finite:
        raise RuntimeError('nonfinite action probe')
    if not converged:
        return dict(macro_f1=None)
    with torch.no_grad():
        predictions = model(z).argmax(1)
    f1 = []
    for label in classes:
        tp = int(((predictions == label) & (labels == label)).sum())
        fp = int(((predictions == label) & (labels != label)).sum())
        fn = int(((predictions != label) & (labels == label)).sum())
        f1.append(2*tp / (2*tp+fp+fn) if 2*tp+fp+fn else 0.)
    return dict(macro_f1=statistics.mean(f1))


def semantic_complete(metrics):
    return bool(metrics) and all(v is not None and np.isfinite(v) for values in metrics.values() for v in values.values())


def semantic_evaluation(projection, teacher, fit_cache, evaluation_cache, *, diagnostics=None):
    fit, evaluation = fit_cache['tensors'], evaluation_cache['tensors']
    text_ids = evaluation['text_index']
    text = project(projection, evaluation['text']) if evaluation['mask'].all() else None
    with torch.no_grad():
        if text is None:
            text = torch.cat([projection(evaluation['text'][i:i+128], evaluation['mask'][i:i+128]).cpu()
                              for i in range(0, len(evaluation['text']), 128)])
        old_text = torch.cat([teacher(evaluation['text'][i:i+128], evaluation['mask'][i:i+128]).cpu()
                              for i in range(0, len(evaluation['text']), 128)])
    text, old_text = text[text_ids], old_text[text_ids]
    result = {}
    subjects = sorted({r['subject_id'] for r in evaluation_cache['rows']})
    event_keys = {}
    for branch in ('stream', 'commit'):
        event_keys[branch] = project(projection, evaluation[branch])
        old_events = project(teacher, evaluation[branch])
        fit_keys = project(projection, fit[branch])
        for subject in subjects:
            selected = torch.tensor([r['subject_id'] == subject for r in evaluation_cache['rows']])
            labels = evaluation['labels'][selected]
            prefix = subject + ':'
            result[prefix+branch] = semantic_metrics(event_keys[branch][selected], text[selected], labels)
            result[prefix+'newtext_oldevent_'+branch] = semantic_metrics(old_events[selected], text[selected], labels)
            result[prefix+'newevent_oldtext_'+branch] = semantic_metrics(event_keys[branch][selected], old_text[selected], labels)
            result[prefix+'action_'+branch] = _action_probe(
                fit_keys, fit['labels'], event_keys[branch][selected], labels,
                diagnostics=diagnostics, context=dict(subject=subject, branch=branch))
    for subject in subjects:
        selected = torch.tensor([r['subject_id'] == subject for r in evaluation_cache['rows']])
        result[subject+':stream_commit'] = dict(consistency_loss=float(
            (1-F.cosine_similarity(event_keys['stream'][selected], event_keys['commit'][selected])).mean()))
    return result


def dev_probes(projection, data, held_subject):
    rows = data['rows']; masks = role_masks(rows, held_subject)
    c, a = project(projection, data['current']), project(projection, data['reference'])
    targets = np.array([r['update_target'] for r in rows], bool)
    ages = np.array([r['age_updates'] for r in rows])
    endpoint = np.array([r.get('probe_eligible', True) for r in rows], bool)
    records, by_seed = [], []
    for seed in PROJECTION_SEEDS:
        age_scores = []
        for age in (6, 7):
            fit = masks['fit'] & (ages == age) & endpoint
            dev = masks['dev'] & (ages == age) & endpoint
            if len(np.unique(targets[fit])) != 2 or not dev.any():
                return dict(score=None, records=records, reason='missing primary fit/dev classes')
            scale = max(float(torch.linalg.vector_norm(c[fit].double()-a[fit].double(), dim=1).square().mean().sqrt()), 1e-6)
            features = relational_features(c, a, seed=seed, scale=scale).numpy()
            fitted = fit_readout(features[fit], targets[fit], np.ones(int(fit.sum())), features,
                                 kind='logistic', seed=seed, settings=_solver())
            if not fitted['converged']:
                return dict(score=None, records=records, reason='dev probe did not converge')
            subject_scores = []
            subjects = sorted({r['subject_id'] for r, active in zip(rows, dev) if active})
            if len(subjects) != 2:
                return dict(score=None, records=records, reason='both dev subjects required')
            for subject in subjects:
                selected = dev & np.array([r['subject_id'] == subject for r in rows])
                metrics = binary_metrics(targets[selected].tolist(), fitted['scores'][selected].tolist())
                if metrics['auroc'] is None:
                    return dict(score=None, records=records, reason='dev subject lacks both classes')
                subject_scores.append(metrics['auroc'])
                records.append(dict(seed=seed, age=age, subject=subject, metrics=metrics,
                                    endpoint_ids=[f"{r['recording_key']}|{r['candidate_id']}|{r['age_updates']}"
                                                  for r, active in zip(rows, selected) if active],
                                    targets=targets[selected].tolist(), scores=fitted['scores'][selected].tolist(),
                                    fit_count=int(fit.sum()), scale=scale,
                                    model_state={k: v.tolist() for k, v in fitted['state'].items()},
                                    mean=fitted['mean'].tolist(), std=fitted['std'].tolist(),
                                    class_weights=fitted['class_weights'].tolist()))
            age_scores.append(statistics.mean(subject_scores))
        by_seed.append(statistics.mean(age_scores))
    return dict(score=statistics.median(by_seed), records=records, reason=None)


def _balanced_rehearsal_indices(cache, rng):
    labels = cache['tensors']['labels'].numpy()
    classes = np.unique(labels)
    order = rng.permutation(classes)
    result = []
    for i in range(TRAIN_SETTINGS['rehearsal_batch']):
        label = order[i % len(order)]
        result.append(int(rng.choice(np.flatnonzero(labels == label))))
    return result


def _head_only_control(old, data, fit_indices, weights, groups, means, stds, scales, seed, epochs, output):
    model = ProjectionAdaptation(old, PROJECTION_SEEDS, scales, means, stds, head_only=True)
    optimizer = torch.optim.AdamW(model.heads.parameters(), lr=TRAIN_SETTINGS['head_lr'],
                                 weight_decay=TRAIN_SETTINGS['weight_decay'])
    c, a = project(old, data['current']), project(old, data['reference'])
    features = torch.stack([(relational_features(c, a, seed=s, scale=scales[i])-means[i])/stds[i]
                            for i, s in enumerate(PROJECTION_SEEDS)])
    y = torch.tensor([r['update_target'] for r in data['rows']], dtype=torch.float32)
    rng = np.random.default_rng(seed)
    names = list(groups)
    for _ in range(epochs):
        order = rng.permutation(len(names))
        for start in range(0, len(order), 16):
            local = [j for k in order[start:start+16] for j in groups[names[k]]]
            indices = fit_indices[local]; w = torch.from_numpy(weights[local]).float()
            loss = torch.stack([(F.binary_cross_entropy_with_logits(head(features[i, indices]).flatten(), y[indices],
                                                                       reduction='none')*w).sum()/w.sum()
                                for i, head in enumerate(model.heads)]).mean()
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.heads.parameters(), 1.); optimizer.step()
    untouched = all(torch.equal(v, model.projection.state_dict()[k]) and torch.equal(v, model.teacher.state_dict()[k])
                    for k, v in old.state_dict().items())
    metrics = []
    masks = role_masks(data['rows'], output['held_subject'])
    with torch.no_grad():
        for i, head in enumerate(model.heads):
            scores = head(features[i]).sigmoid().flatten().numpy()
            for age in (6, 7):
                selected = masks['dev'] & np.array([r['age_updates'] == age for r in data['rows']])
                metrics.append(dict(seed=PROJECTION_SEEDS[i], age=age,
                                    metrics=binary_metrics(y[selected].bool().tolist(), scores[selected].tolist())))
    return dict(teacher_unchanged=untouched, projection_unchanged=untouched, epochs=epochs, dev=metrics,
                state={k: v.detach().cpu() for k, v in model.heads.state_dict().items()})


def _train_epoch(model, optimizer, data, fit_indices, weights, groups, fit_cache, rng, settings):
    """One deterministic optimizer epoch; also used for the isolated epoch6 replay."""
    names = list(groups)
    order = rng.permutation(len(names)); stats, norms = [], []
    first_step = None
    targets = torch.tensor([r['update_target'] for r in data['rows']], dtype=torch.float32)
    model.train()
    for start in range(0, len(order), settings['candidate_batch']):
        local = [j for k in order[start:start+settings['candidate_batch']] for j in groups[names[k]]]
        indices = fit_indices[local]
        rehearsal = rehearsal_batch(fit_cache, _balanced_rehearsal_indices(fit_cache, rng), 'cpu')
        loss, components = model.loss(
            data['current'][indices], data['reference'][indices], targets[indices],
            torch.tensor(weights[local], dtype=torch.float32), rehearsal,
            consistency_weight=fit_cache['manifest']['consistency_weight'],
            temperature=fit_cache['manifest']['temperature'], lambda_s=settings['lambda_s'],
            lambda_a=settings['lambda_a'], lambda_t=settings.get('lambda_t', 1.), detach_components=False)
        if not torch.isfinite(loss):
            raise RuntimeError('nonfinite Projection objective')
        if first_step is None:
            parameters = list(model.projection.parameters())
            vectors = {}
            for name, term in components.items():
                gradient = torch.autograd.grad(term, parameters, retain_graph=True, allow_unused=True)
                vectors[name] = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                                           for p, g in zip(parameters, gradient)])
            first_step = dict(norms={k: float(v.norm()) for k, v in vectors.items()}, cosines={})
            for left, right in (('transition', 'semantic'), ('transition', 'anchor'), ('semantic', 'anchor')):
                denominator = vectors[left].norm()*vectors[right].norm()
                first_step['cosines'][left+'_'+right] = float(vectors[left].dot(vectors[right])/denominator) if denominator > 1e-12 else None
        optimizer.zero_grad(); loss.backward()
        gradients = [p.grad for p in model.projection.parameters() if p.grad is not None]
        if not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError('nonfinite Projection gradient')
        norms.append(float(torch.sqrt(sum(g.square().sum() for g in gradients))))
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], settings['gradient_clip'])
        optimizer.step(); stats.append({k: float(v.detach()) for k, v in components.items()})
    return dict(components={k: statistics.mean(s[k] for s in stats) for k in stats[0]},
                projection_gradient_mean=statistics.mean(norms), first_step_gradients=first_step)


def train_fold(old, data, caches, output, *, held_subject, seed, epochs, provenance, synthetic=False, resume=False,
               settings=None):
    torch.set_num_threads(1)
    settings = {**TRAIN_SETTINGS, 'lambda_t': 1., **(settings or {})}
    output = Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=resume)
    rows = data['rows']; masks = role_masks(rows, held_subject)
    if masks['held'].any() or masks['calibration'].any():
        raise ValueError('parameter training must not receive held/calibration numeric inputs')
    fit_indices = np.flatnonzero(masks['fit'])
    fit_rows = [rows[i] for i in fit_indices]
    weights, weight_report = balanced_candidate_weights(fit_rows)
    groups = defaultdict(list)
    for i, r in enumerate(fit_rows):
        if weights[i] > 0:
            groups[candidate_identity(r)].append(i)
    if not groups:
        raise ValueError('no age-balanced fitting Candidates')
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    c, a = project(old, data['current']), project(old, data['reference'])
    positive_fit = fit_indices[weights > 0]
    w = weights[weights > 0]
    scale = max(float(np.sqrt(np.average((c[positive_fit].double()-a[positive_fit].double()).square().sum(1), weights=w))), 1e-6)
    means, stds = [], []
    for projection_seed in PROJECTION_SEEDS:
        x = relational_features(c, a, seed=projection_seed, scale=scale).numpy()[positive_fit]
        mean = np.average(x, axis=0, weights=w)
        means.append(mean); stds.append(np.maximum(np.sqrt(np.average((x-mean)**2, axis=0, weights=w)), 1e-6))
    means, stds = torch.tensor(np.array(means), dtype=torch.float32), torch.tensor(np.array(stds), dtype=torch.float32)
    model = ProjectionAdaptation(old, PROJECTION_SEEDS, [scale]*3, means, stds)
    optimizer = torch.optim.AdamW([dict(params=model.projection.parameters(), lr=settings['projection_lr']),
                                  dict(params=model.heads.parameters(), lr=settings['head_lr'])],
                                 weight_decay=settings['weight_decay'])
    baseline_diagnostics = []
    baseline_retention = semantic_evaluation(old, old, caches['fit'], caches['dev'], diagnostics=baseline_diagnostics)
    write_json(output/'baseline-probe-diagnostics.json', baseline_diagnostics)
    if not semantic_complete(baseline_retention):
        raise ValueError('baseline semantic probe undefined')
    epoch0 = dev_probes(old, data, held_subject)
    if epoch0['score'] is None:
        raise ValueError('epoch0 primary dev probe undefined: '+str(epoch0['reason']))
    identity = dict(provenance=provenance, held_subject=held_subject, seed=seed, synthetic=synthetic,
                    settings=settings, row_fingerprint=_fingerprint(rows),
                    fit_semantic_fingerprint=_fingerprint(caches['fit']['rows']),
                    dev_semantic_fingerprint=_fingerprint(caches['dev']['rows']),
                    semantic_manifest_fingerprints={role: _fingerprint(caches[role]['manifest']) for role in ('fit', 'dev')})
    write_json(output/'fit-weights.json', dict(report=weight_report, rows=fit_rows, weights=weights.tolist()))
    history = [dict(epoch=0, score=epoch0['score'], eligible=True, retention=baseline_retention,
                    dev=epoch0, components=None)]
    first_epoch = 1
    pending = None
    if resume and (output/'last.pt').exists():
        saved = torch.load(output/'last.pt', map_location='cpu', weights_only=False)
        if saved['identity'] != identity or (not synthetic and saved['requested_epochs'] != epochs):
            raise ValueError('resume configuration/provenance mismatch')
        model.load_state_dict(saved['model_state']); optimizer.load_state_dict(saved['optimizer_state'])
        rng.bit_generator.state = saved['rng_state']['numpy_generator']
        torch.set_rng_state(saved['rng_state']['torch'])
        history = saved['history']
        pending = saved.get('pending_evaluation') if not saved.get('evaluation_complete', True) else None
        first_epoch = saved['epoch'] if pending else saved['epoch']+1
    else:
        save_pt(output/'epoch-00.pt', dict(projection_state=old.state_dict(), epoch=0, identity=identity))
    for epoch in range(first_epoch, epochs+1):
        started = time.monotonic()
        if pending is not None:
            if pending['epoch'] != epoch:
                raise ValueError('pending evaluation epoch mismatch')
            summary = pending; pending = None
        else:
            summary = _train_epoch(model, optimizer, data, fit_indices, weights, groups, caches['fit'], rng, settings)
            summary.update(epoch=epoch, optimization_seconds=time.monotonic()-started)
        change = max(float((v-old.state_dict()[k]).abs().max()) for k, v in model.projection.state_dict().items())
        def persist(evaluation_complete):
            save_pt(output/'last.pt', dict(model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                                          epoch=epoch, requested_epochs=epochs, identity=identity, history=history,
                                          evaluation_complete=evaluation_complete,
                                          pending_evaluation=None if evaluation_complete else summary,
                                          rng_state=dict(numpy_generator=rng.bit_generator.state, torch=torch.get_rng_state())))
        save_pt(output/f'epoch-{epoch:02d}.pt', dict(projection_state=model.projection.state_dict(), epoch=epoch, identity=identity))
        persist(False)
        model.eval()
        evaluation_started = time.monotonic()
        dev = dev_probes(model.projection, data, held_subject)
        diagnostics = []
        retention = semantic_evaluation(model.projection, model.teacher, caches['fit'], caches['dev'], diagnostics=diagnostics)
        write_json(output/f'probe-diagnostics-{epoch:02d}.json', diagnostics)
        eligible = dev['score'] is not None and retention_gate(baseline_retention, retention)
        history.append(dict(epoch=epoch, score=dev['score'], eligible=eligible, dev=dev, retention=retention,
                            semantic_complete=semantic_complete(retention), components=summary['components'],
                            first_step_gradients=summary['first_step_gradients'],
                            projection_gradient_mean=summary['projection_gradient_mean'], projection_max_parameter_change=change,
                            seconds=summary['optimization_seconds']+time.monotonic()-evaluation_started))
        persist(True)
        write_json(output/'history.json', history)
        print(json.dumps(dict(phase='projection_train', held_subject=held_subject, seed=seed, epoch=epoch,
                              score=dev['score'], semantic_eligible=eligible, seconds=history[-1]['seconds'])), flush=True)
    if not all(torch.equal(v, model.teacher.state_dict()[k]) for k, v in old.state_dict().items()):
        raise RuntimeError('teacher weights changed')
    control = _head_only_control(old, data, fit_indices, weights, groups, means, stds, [scale]*3, seed, epochs,
                                 dict(held_subject=held_subject))
    save_pt(output/'head-only.pt', control)
    control_summary = {k: v for k, v in control.items() if k != 'state'}
    selected = choose_checkpoint(history)
    selected_path = output/f'epoch-{selected:02d}.pt'
    result = dict(status='locked', held_subject=held_subject, seed=seed, synthetic=synthetic,
                  completed_epochs=epochs, selected_epoch=selected,
                  selected_checkpoint=str(selected_path.resolve()), selected_sha256=sha256(selected_path),
                  improved_dev=selected > 0 and history[selected]['score'] > history[0]['score'],
                  head_only=control_summary, identity=identity, source_manifest_sha256=provenance['source_manifest_sha256'])
    write_json(output/'selection-lock.json', result)
    return result


def _old_projection(stage2_checkpoint, event_dim, semantic_dim=128):
    payload = torch.load(stage2_checkpoint, map_location='cpu', weights_only=False)
    state = {k.removeprefix('semantic_projection.'): v for k, v in payload['model_state'].items()
             if k.startswith('semantic_projection.')}
    model = SemanticProjection(event_dim, semantic_dim)
    model.load_state_dict(state, strict=True)
    return model.eval()


def run(source, output, *, cache_device='cuda:0', smoke=False, resume=False):
    source, output = Path(source), Path(output)
    if output.exists() and not resume:
        raise FileExistsError(output)
    checked = load_sequence_source(source)
    upstream = checked['manifest']; stage2 = Path(upstream['stage2_checkpoint']); config = Path(upstream['config'])
    import yaml
    raw = yaml.safe_load(config.read_text(encoding='utf-8'))
    old = _old_projection(stage2, raw['model']['llm_dim'], raw['model']['semantic_dim'])
    output.mkdir(parents=True, exist_ok=resume)
    spec = Path('docs/stage3-transition-aware-projection-remedies-2026-10-02.md')
    code = {p.name: sha256(p) for p in Path(__file__).parent.glob('parrad_projection_*.py')}
    provenance = dict(source_manifest_sha256=checked['manifest_sha256'], stage2_checkpoint_sha256=sha256(stage2),
                      checkpoint_sha256=upstream['checkpoint_sha256'], config_sha256=sha256(config),
                      code_sha256=code, spec_sha256=sha256(spec), source_split='train')
    manifest = dict(status='preparing', provenance=provenance, settings=TRAIN_SETTINGS, smoke=smoke,
                    validation_test_read=False, production_model_changed=False, completed=[], epoch0_key_errors={})
    if resume:
        previous = json.loads((output/'manifest.json').read_text())
        if previous['provenance'] != provenance or previous['smoke'] != smoke:
            raise ValueError('run resume provenance mismatch')
    write_json(output/'manifest.json', manifest)
    try:
        folds = SUBJECTS[:1] if smoke else SUBJECTS
        seeds = TRAIN_SEEDS[:1] if smoke else TRAIN_SEEDS
        for held_subject in folds:
            root = output/held_subject; root.mkdir(exist_ok=resume)
            data = load_transition_rows(source, checked, old, held_subject,
                                        event_tokens=raw['model']['event_tokens'], llm_dim=raw['model']['llm_dim'])
            manifest['epoch0_key_errors'][held_subject] = data['max_old_key_error']
            write_json(output/'manifest.json', manifest)
            semantic = {}
            for role in ('fit', 'dev'):
                cache = root/('semantic-'+role)
                if not cache.exists():
                    prepare_rehearsal(raw['data']['train_manifest'], cache, held_subject, role=role,
                                      config_path=config, stage2_checkpoint=stage2, device=torch.device(cache_device))
                semantic[role] = load_rehearsal(cache, held_subject, role)
                if semantic[role]['manifest']['checkpoint_sha256'] != provenance['stage2_checkpoint_sha256']:
                    raise ValueError('semantic cache checkpoint mismatch')
            eligible = set()
            for key, rows in checked['rows_by_record'].items():
                for age in (6, 7):
                    selected, _ = select_endpoints(rows, checked['candidates_by_record'][key], cohort='natural',
                                                  endpoint_kind='fixed_age', age=age, history_lengths=(1, 3), task='update')
                    eligible.update((e.recording_key, e.endpoint_row) for e in selected)
            for row in data['rows']:
                row['probe_eligible'] = (row['recording_key'], row['source_row']) in eligible
            for seed in seeds:
                manifest.update(status='training', active_fold=held_subject, active_seed=seed)
                write_json(output/'manifest.json', manifest)
                result = train_fold(old, data, semantic, root/f'seed-{seed}', held_subject=held_subject, seed=seed,
                                    epochs=1 if smoke else 20, provenance=provenance,
                                    resume=resume and (root/f'seed-{seed}').exists())
                manifest['completed'].append(result); write_json(output/'manifest.json', manifest)
        final_source = load_sequence_source(source)
        if final_source['manifest_sha256'] != checked['manifest_sha256']:
            raise RuntimeError('source changed during training')
        manifest.update(status='smoke_complete' if smoke else 'training_complete', completed_unix=time.time(),
                        final_evaluation='pending separate locked-checkpoint evaluation')
        write_json(output/'manifest.json', manifest)
        return manifest
    except Exception as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        write_json(output/'manifest.json', manifest); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-device', default='cuda:0')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    result = run(args.source_dir, args.output_dir, cache_device=args.cache_device, smoke=args.smoke, resume=args.resume)
    print(json.dumps(dict(status=result['status'], folds=len(result['completed']))), flush=True)


if __name__ == '__main__':
    main()
