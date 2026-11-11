"""Projection-only TRAIN adaptation contracts; no controller or upstream optimization."""
from __future__ import annotations

import copy
from collections import Counter, defaultdict
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_mmwave.models.semantic import SemanticProjection
from continuous_mmwave.parrad_stage3_linear_probe import SUBJECTS
from continuous_mmwave.parrad_stage3_sequence_features import _matrix
from continuous_mmwave.stages.stage2 import supervised_symmetric_alignment_loss


def role_masks(rows, held_subject):
    if held_subject not in SUBJECTS or any(
        r.get('source_split') != 'train' or r.get('subject_id') not in SUBJECTS
        or r.get('session_id') not in ('set000', 'set001', 'set002', 'set003')
        or r.get('sensor_id') not in ('TI_xWR14xx', 'TI_xWR68xx') for r in rows
    ):
        raise ValueError('adaptation accepts only approved TRAIN subjects/sessions/sensors')
    held = np.array([r['subject_id'] == held_subject for r in rows], bool)
    sessions = np.array([r['session_id'] for r in rows])
    fit = ~held & np.isin(sessions, ['set000', 'set001'])
    dev = ~held & (sessions == 'set002')
    cal = ~held & (sessions == 'set003')
    return dict(fit=fit, dev=dev, calibration=cal, held=held, probe_fit=fit | dev)


def candidate_identity(r):
    return r['recording_key'], r['candidate_id']


def balanced_candidate_weights(rows):
    for r in rows:
        if (r.get('source_split') != 'train' or r.get('cohort') != 'natural'
                or r.get('session_id') not in ('set000', 'set001')
                or r.get('subject_id') not in SUBJECTS or type(r.get('update_target')) is not bool
                or type(r.get('age_updates')) is not int or not 3 <= r['age_updates'] <= 7):
            raise ValueError('weights can only use natural age3–7 fit updates')
    identities = [(candidate_identity(r), r['age_updates']) for r in rows]
    if len(set(identities)) != len(rows):
        raise ValueError('duplicate Candidate/age')
    age_classes = defaultdict(set)
    for r in rows:
        age_classes[r['age_updates']].add(r['update_target'])
    excluded = sorted(a for a, classes in age_classes.items() if len(classes) != 2)
    keep = [i for i, r in enumerate(rows) if r['age_updates'] not in excluded]
    counts = Counter(candidate_identity(rows[i]) for i in keep)
    mass = defaultdict(float)
    for i in keep:
        r = rows[i]
        mass[r['age_updates'], int(r['update_target'])] += 1 / counts[candidate_identity(r)]
    factors = {key: float(np.clip((mass[key[0], 0] + mass[key[0], 1]) / (2 * value), .25, 4))
               for key, value in list(mass.items())}
    weights = np.zeros(len(rows), np.float64)
    totals = defaultdict(float)
    for i in keep:
        r = rows[i]
        weights[i] = factors[r['age_updates'], int(r['update_target'])] / counts[candidate_identity(r)]
        totals[candidate_identity(r)] += weights[i]
    for i in keep:
        weights[i] /= totals[candidate_identity(rows[i])]
    buckets = []
    for age in sorted(age_classes):
        for label in (0, 1):
            selected = [i for i in keep if rows[i]['age_updates'] == age and rows[i]['update_target'] == bool(label)]
            buckets.append(dict(age=age, label=label, count=len(selected), base_mass=mass.get((age, label), 0.),
                                final_mass=float(weights[selected].sum()), factor=factors.get((age, label))))
    return weights, dict(excluded_ages=excluded, candidates=len(counts), buckets=buckets,
                         candidate_total_min=1. if counts else None, candidate_total_max=1. if counts else None,
                         max_update_weight=float(weights.max()) if len(weights) else None)


def relational_features(current, reference, *, seed, scale):
    """Differentiable K1 implementation of the immutable 130-input probe interface."""
    if (current.ndim != 2 or current.shape != reference.shape or current.shape[1] != 128
            or not math.isfinite(float(scale)) or scale <= 0
            or not torch.isfinite(current).all() or not torch.isfinite(reference).all()):
        raise ValueError('requires paired finite128-dimensional keys and positive scale')
    c, a = current.double(), reference.double()
    delta = c - a
    matrix = _matrix(128, seed).to(device=c.device)
    projected = (delta.float() @ matrix).double() / scale
    denom = (c.square().sum(1) * a.square().sum(1)).sqrt().clamp_min(torch.finfo(torch.float64).tiny)
    cosine = ((c * a).sum(1) / denom).clamp(-1, 1)
    # vector_norm has a defined zero gradient at exact equal pairs.
    distance = torch.linalg.vector_norm(delta, dim=1) / scale
    return torch.cat((projected, (1 - cosine)[:, None], distance[:, None]), 1).to(current.dtype)


def semantic_metrics(events, texts, labels):
    if events.shape != texts.shape or events.ndim != 2 or labels.shape != (len(events),):
        raise ValueError('semantic metrics require paired keys and action labels')
    similarity = F.normalize(events.float(), dim=-1) @ F.normalize(texts.float(), dim=-1).T
    positive = labels[:, None].eq(labels[None, :])
    if not len(labels):
        raise ValueError('empty semantic cohort')
    e_hit = labels[similarity.argmax(1)].eq(labels).float()
    t_hit = labels[similarity.argmax(0)].eq(labels).float()
    unique = labels.unique()
    return dict(event_to_text=float(torch.stack([e_hit[labels == k].mean() for k in unique]).mean()),
                text_to_event=float(torch.stack([t_hit[labels == k].mean() for k in unique]).mean()),
                margin=float(similarity[positive].mean() - similarity[~positive].mean())
                if (~positive).any() else None)


def retention_gate(old, new):
    if not old or set(old) != set(new):
        return False
    for branch, metrics in old.items():
        if set(metrics) != set(new[branch]):
            return False
        for name, baseline in metrics.items():
            actual = new[branch][name]
            if baseline is None or actual is None or not math.isfinite(actual):
                return False
            if name == 'consistency_loss':
                if actual > baseline + .02:
                    return False
            elif actual < baseline - .0200000001 or (name == 'margin' and baseline > 0 and actual <= 0):
                return False
    return True


def choose_checkpoint(history):
    eligible = [r for r in history if r['eligible'] and r['score'] is not None and math.isfinite(r['score'])]
    if not eligible:
        raise ValueError('no eligible checkpoint, including epoch0')
    return max(eligible, key=lambda r: (r['score'], -r['epoch']))['epoch']


class ProjectionAdaptation(nn.Module):
    def __init__(self, old: SemanticProjection, seeds, scales, means, stds, *, head_only=False):
        super().__init__()
        self.projection = copy.deepcopy(old)
        self.teacher = copy.deepcopy(old).requires_grad_(False).eval()
        self.projection.requires_grad_(not head_only)
        self.seeds, self.scales = tuple(seeds), tuple(scales)
        if len(seeds) != len(scales) or means.shape != (len(seeds), 130) or stds.shape != means.shape:
            raise ValueError('one normalization and scale per training head')
        if not torch.isfinite(means).all() or not torch.isfinite(stds).all() or not (stds > 0).all():
            raise ValueError('finite fit-only normalization required')
        self.register_buffer('means', means.float())
        self.register_buffer('stds', stds.float())
        self.heads = nn.ModuleList(nn.Linear(130, 1) for _ in seeds)
        for head in self.heads:
            nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)

    def loss(self, current, reference, target, weights, rehearsal, *, consistency_weight=.1,
             temperature=.07, lambda_s=1., lambda_a=1., lambda_t=1., detach_components=True):
        if weights.shape != target.shape or not (weights >= 0).all() or weights.sum() <= 0:
            raise ValueError('positive total Candidate mass required')
        c, a = self.projection(current), self.projection(reference)
        losses = []
        for i, head in enumerate(self.heads):
            features = relational_features(c, a, seed=self.seeds[i], scale=self.scales[i])
            logits = head((features - self.means[i]) / self.stds[i]).flatten()
            losses.append((F.binary_cross_entropy_with_logits(logits, target, reduction='none') * weights).sum()
                          / weights.sum())
        transition = torch.stack(losses).mean()
        keys = {name: self.projection(rehearsal[name], rehearsal['mask'] if name == 'text' else None)
                for name in ('stream', 'commit', 'text')}
        semantic = sum(supervised_symmetric_alignment_loss(keys[name], keys['text'], rehearsal['labels'], temperature)
                       for name in ('stream', 'commit')) / 2
        semantic += consistency_weight * (1 - F.cosine_similarity(keys['stream'], keys['commit'])).mean()
        with torch.no_grad():
            teacher = {name: self.teacher(rehearsal[name], rehearsal['mask'] if name == 'text' else None)
                       for name in keys}
            old_c, old_a = self.teacher(current), self.teacher(reference)
        rehearsal_anchor = torch.stack([(1 - F.cosine_similarity(keys[name], teacher[name])).mean()
                                        for name in keys]).mean()
        pair_anchor = ((1 - F.cosine_similarity(c, old_c)).mean()
                       + (1 - F.cosine_similarity(a, old_a)).mean()) / 2
        anchor = (rehearsal_anchor + pair_anchor) / 2
        total = lambda_t * transition + lambda_s * semantic + lambda_a * anchor
        parts = dict(transition=transition, semantic=semantic, anchor=anchor)
        return total, {k: v.detach() if detach_components else v for k, v in parts.items()}
