"""Deterministic reliability-gated leaky evidence; this is not an SPRT."""
from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class EvidenceConfig:
    rho: float = .95
    tau_stability: float = .5
    tau_difference: float = .6
    tau_evidence: float = .2
    evidence_weight: float = 1.
    h_plus: float = 3.
    h_minus: float = -3.
    warmup_s: float = .5
    timeout_s: float = 2.
    eps: float = 1e-8

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('nonfinite evidence configuration')
        if not (0 < self.rho <= 1 and 0 <= self.tau_stability < 1 and 0 < self.tau_difference < 1
                and 0 <= self.tau_evidence and self.evidence_weight >= 0 and self.h_minus < 0 < self.h_plus
                and 0 <= self.warmup_s < self.timeout_s and 0 < self.eps < .01):
            raise ValueError('invalid sequential evidence configuration')


@dataclass(frozen=True)
class EvidenceResult:
    score: float
    increment: float
    reliability: float
    decision: str


class SequentialEvidence:
    def __init__(self, config: EvidenceConfig):
        self.config, self.score, self.last_age = config, 0., -1.

    def update(self, difference, stability, evidence_difference, age_s):
        values = (difference, stability, evidence_difference, age_s)
        if not all(math.isfinite(v) for v in values) or not (0 <= difference <= 1 and 0 <= stability <= 1
            and evidence_difference >= 0 and age_s >= 0 and age_s >= self.last_age):
            raise ValueError('invalid/nonmonotonic evidence input')
        c = self.config
        g = max(0.,min(1.,(stability-c.tau_stability)/(1-c.tau_stability)))
        log_evidence = math.log((difference+c.eps)/(1-difference+c.eps))-math.log(c.tau_difference/(1-c.tau_difference))
        increment = g*(log_evidence+c.evidence_weight*(evidence_difference-c.tau_evidence))
        self.score = c.rho*self.score+increment
        self.last_age = age_s
        decision = 'none'
        if age_s+1e-9 >= c.warmup_s:
            if self.score >= c.h_plus:
                decision = 'confirm'
            elif self.score <= c.h_minus:
                decision = 'reject'
            elif age_s+1e-9 >= c.timeout_s:
                decision = 'timeout_reject'
        return EvidenceResult(self.score,increment,g,decision)


@dataclass
class Episode:
    candidate_id: str
    active_id: str
    proposal_time: float
    observations: list[dict] = field(default_factory=list)
    decision: str = 'pending'
    end_time: float | None = None
    refined_boundary: float | None = None
    gt_transition_exists: int | None = None

    def to_dict(self):
        return asdict(self)


def episode_label(episode, segments, active_label, *, min_stable_s=.5):
    """Offline only: label the sustained state at the lifecycle end.

    Unknown coverage returns None; a transient B followed by A is negative.
    segments contain (start,end,label); no GT enters the online controller.
    """
    if episode.end_time is None or min_stable_s <= 0:
        raise ValueError('finished episode and positive stable duration required')
    rows = sorted(segments)
    if any(not math.isfinite(a) or not math.isfinite(b) or a >= b for a,b,_ in rows):
        raise ValueError('invalid GT intervals')
    if any(rows[i][1] > rows[i+1][0] for i in range(len(rows)-1)):
        raise ValueError('overlapping GT intervals')
    end = episode.end_time
    last = next(((a,b,label) for a,b,label in rows if a < end <= b),None)
    if last is None:
        return None
    a, _, label = last
    cursor = episode.proposal_time
    for left,right,_ in rows:
        if right <= cursor:
            continue
        if left > cursor:
            return None
        cursor = right
        if cursor >= end:
            break
    if cursor < end:
        return None
    if label == active_label:
        return 0
    return int(end-max(a,episode.proposal_time) >= min_stable_s)


def replay_episode(episode, config):
    accumulator = SequentialEvidence(config)
    for row in episode.observations:
        result = accumulator.update(row['difference'],row['stability'],row['physical_difference'],row['age_s'])
        if result.decision != 'none':
            return result.decision, row['age_s'], result.score
    return 'incomplete', None, accumulator.score


def calibrate_episodes(episodes, configurations, *, max_false_confirm_rate=.05,probe=None,provenance=None):
    """Choose on labelled development episodes only; never on a test split.

    Return all candidate metrics, including failed threshold configurations.
    """
    from .training import validate_bias_probe
    if provenance is None:
        raise ValueError('calibration requires passed current bias probe and episode provenance')
    required = ('frontend_digest','heads_digest','semantic_head_digest','controller_config_digest','episodes_sha256')
    if not all(isinstance(provenance.get(k),str) and len(provenance[k]) == 64 for k in required):
        raise ValueError('calibration source fingerprints required')
    validate_bias_probe(probe,frontend_digest=provenance['frontend_digest'],semantic_head_digest=provenance['semantic_head_digest'])
    if not math.isfinite(max_false_confirm_rate) or not 0 <= max_false_confirm_rate <= 1:
        raise ValueError('invalid calibration false-confirm limit')
    episodes = list(episodes)
    if not episodes or any(ep.gt_transition_exists not in (0,1) for ep in episodes):
        raise ValueError('calibration needs complete labelled episodes')
    positives = sum(ep.gt_transition_exists == 1 for ep in episodes)
    negatives = len(episodes)-positives
    if not positives or not negatives:
        raise ValueError('calibration requires positive and negative episodes')
    results = []
    for config in configurations:
        decisions = [replay_episode(ep,config) for ep in episodes]
        tp = sum(ep.gt_transition_exists == 1 and d[0] == 'confirm' for ep,d in zip(episodes,decisions))
        fp = sum(ep.gt_transition_exists == 0 and d[0] == 'confirm' for ep,d in zip(episodes,decisions))
        delays = [d[1] for ep,d in zip(episodes,decisions) if ep.gt_transition_exists == 1 and d[0] == 'confirm']
        results.append({'config':asdict(config),'recall':tp/positives,'false_confirm_rate':fp/negatives,
                        'mean_confirm_age_s':sum(delays)/len(delays) if delays else None,
                        'incomplete':sum(d[0] == 'incomplete' for d in decisions),
                        'eligible':fp/negatives <= max_false_confirm_rate})
    eligible = [r for r in results if r['eligible'] and not r['incomplete']]
    selected = min(eligible,key=lambda r:(-r['recall'],r['mean_confirm_age_s'] if r['mean_confirm_age_s'] is not None else math.inf)) if eligible else None
    return {'selected':selected,'all_configurations':results,'split':'development-episodes',
            'source':dict(provenance),'max_false_confirm_rate':max_false_confirm_rate,
            'probe_criteria_sha256':probe['criteria_sha256']}
