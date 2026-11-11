"""Locked24-condition prescreen followed by unique complete policy replay."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict,replace
from itertools import product
import json
from pathlib import Path

import torch

from .boundary import localize_boundary
from .checkpoint import file_sha,module_digest
from .collect import COLLECTION_CONTROLLER,COLLECTION_EVIDENCE,collect_candidate_episodes,visible_pair_label,recording_intervals
from .evidence import SequentialEvidence,EvidenceConfig
from .prepare import write_json
from .training import probe_semantic_head,config_digest


CRITERIA=dict(min_early_auroc=.75,min_mature_auroc=.80,max_same_early_fpr=.10,max_abs_same_age_correlation=.30)


def fixed_grid():
    return [replace(COLLECTION_EVIDENCE,tau_difference=tau,h_plus=float(positive),h_minus=float(negative),evidence_weight=float(physical))
            for tau,positive,negative,physical in product((.55,.60),(2,3,4),(-2,-3),(0,1))]


def score_episodes(episodes,config=None):
    decisions=[]; tp=fp=positive=negative=unknown=incomplete=0; ages=[]
    for episode in episodes:
        decision=episode['decision']; end=episode['end_time']; age=None
        if config is not None:
            accumulator=SequentialEvidence(config)
            decision='incomplete'
            for row in episode['observations']:
                result=accumulator.update(row['difference'],row['stability'],row['physical_difference'],row['age_s'])
                if result.decision!='none':
                    decision=result.decision; end=row['timestamp']; age=row['age_s']
                    break
        label=visible_pair_label(episode['gt_intervals'],reference_time=episode['reference_time'],current_time=end)
        if label==1:
            positive+=1; tp+=decision=='confirm'
            if decision=='confirm':
                ages.append(age if age is not None else end-episode['proposal_time']+.18)
        elif label==0:
            negative+=1; fp+=decision=='confirm'
        else:
            unknown+=decision=='confirm'
        incomplete+=decision in ('incomplete','pending')
        decisions.append(dict(episode_id=episode['episode_id'],decision=decision,simulated_end=end,label_at_end=label))
    recall=tp/positive if positive else None
    fcr=fp/negative if negative else None
    return dict(positive_episodes=positive,negative_episodes=negative,true_confirms=tp,false_confirms=fp,
        false_confirm_rate=fcr,recall=recall,unknown_confirms=unknown,incomplete=incomplete,
        mean_confirm_age_s=sum(ages)/len(ages) if ages else None,
        eligible=bool(recall is not None and fcr is not None and recall>=.7 and fcr<=.05 and not unknown and not incomplete),
        decisions=decisions)


def probe_collection(stage,frontend,collection,output):
    pairs=collection['pairs']
    rows=[]
    for start in range(0,len(pairs),256):
        selected=pairs[start:start+256]
        rows.append(dict(active=torch.stack([r['active'] for r in selected]),candidate=torch.stack([r['candidate'] for r in selected]),
            target=torch.tensor([r['target'] for r in selected],dtype=torch.float32),
            age_seconds=torch.tensor([r['age_seconds'] for r in selected]),episode_ids=[r['episode_id'] for r in selected]))
    probe=probe_semantic_head(stage.heads.semantic,rows,frontend_digest=module_digest(frontend),
        cache_sha256=collection['manifest']['collection_sha256'],split='development',criteria=CRITERIA,
        early_range=(.54,.72),mature_range=(1.08,1.98),min_candidates=20)
    write_json(output,probe)
    return probe


def boundary_scores(collection,index,*,replacement=None):
    from ..parrad_stage3_diagnostics import match_boundaries_seconds
    predictions=defaultdict(list)
    if replacement is None:
        for episode in collection['episodes']:
            if episode['decision']=='confirm':
                if episode['refined_boundary'] is None:
                    raise RuntimeError('confirmed episode lacks refined boundary')
                predictions[episode['record_uid']].append(episode['refined_boundary'])
    else:
        for window in collection['boundary_windows']:
            selected=localize_boundary(window['frames'],proposal_time=window['proposal_time'],confirm_time=window['confirm_time'],
                active_start=window['active_start'],delta_pre_s=.54,weights=replacement)
            predictions[window['record_uid']].append(selected.timestamp)
    rows=[]
    for source in index['records']:
        if source['role']!='dev':
            continue
        record=torch.load(source['path'],map_location='cpu',weights_only=False)
        intervals=recording_intervals(record)
        truth=[right[0] for left,right in zip(intervals,intervals[1:]) if left[2]>=0 and right[2]>=0 and left[2]!=right[2]]
        row=dict(record_uid=source['uid'],subject_id=source['metadata']['subject_id'],sensor_id=source['metadata']['sensor_id'],
                 predictions=sorted(predictions[source['uid']]),truth=truth,metrics={})
        for tolerance in (.18,.36,.54,1.):
            result=match_boundaries_seconds(row['predictions'],truth,tolerance)
            row['metrics'][str(tolerance)]=dict(tp=result.true_positives,fp=result.false_positives,fn=result.false_negatives,
                precision=result.precision,recall=result.recall,f1=result.f1,
                mae=sum(result.absolute_errors_s)/len(result.absolute_errors_s) if result.absolute_errors_s else None)
        rows.append(row)
    if not rows:
        raise ValueError('complete dev boundary scores missing')
    return dict(recordings=rows,macro_f1=sum(r['metrics']['0.36']['f1'] for r in rows)/len(rows),
        macro_precision=sum(r['metrics']['0.36']['precision'] for r in rows)/len(rows),
        macro_recall=sum(r['metrics']['0.36']['recall'] for r in rows)/len(rows),
        macro_mae=sum(r['metrics']['0.36']['mae'] for r in rows if r['metrics']['0.36']['mae'] is not None)
                  /max(1,sum(r['metrics']['0.36']['mae'] is not None for r in rows)))


def calibrate_and_replay(frontend,stage,index,output):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    initial=collect_candidate_episodes(frontend,stage,index,output/'initial',role='dev')
    probe=probe_collection(stage,frontend,initial,output/'initial-probe.json')
    if probe['passed'] is not True:
        return dict(passed=False,stop='initialization-probe',probe=probe)
    grid=[]
    for config in fixed_grid():
        metrics=score_episodes(initial['episodes'],config)
        grid.append(dict(config=asdict(config),**metrics))
    write_json(output/'grid-all24.json',grid)
    eligible=[(i,row) for i,row in enumerate(grid) if row['eligible']]
    if not eligible:
        return dict(passed=False,stop='no-eligible-accumulator',probe=probe,grid=grid)
    selected_index,selected=min(eligible,key=lambda item:(-item[1]['recall'],item[1]['mean_confirm_age_s'],item[0]))
    evidence=EvidenceConfig(**selected['config'])
    replay=collect_candidate_episodes(frontend,stage,index,output/'selected-complete-replay',role='dev',evidence_config=evidence)
    replay_metric=score_episodes(replay['episodes'])
    replay_probe=probe_collection(stage,frontend,replay,output/'selected-current-probe.json')
    write_json(output/'selected-complete-metrics.json',replay_metric)
    if not replay_metric['eligible'] or replay_probe['passed'] is not True:
        return dict(passed=False,stop='unique-selected-closed-loop-replay',probe=replay_probe,selected=selected)
    if evidence.evidence_weight==0:
        return dict(passed=False,stop='physical-evidence-not-qualified',selected=selected,probe=replay_probe)
    control=collect_candidate_episodes(frontend,stage,index,output/'lambda0-control',role='dev',
        evidence_config=replace(evidence,evidence_weight=0.))
    control_metric=score_episodes(control['episodes'])
    primary_boundary=boundary_scores(replay,index); control_boundary=boundary_scores(control,index)
    paired=dict(primary=replay_metric,control=control_metric,primary_boundary=primary_boundary,control_boundary=control_boundary)
    write_json(output/'lambda0-paired-control.json',paired)
    if (control_metric['false_confirm_rate'] is None or
        replay_metric['false_confirm_rate']>control_metric['false_confirm_rate']+.01 or
        primary_boundary['macro_recall']<control_boundary['macro_recall']-.02):
        return dict(passed=False,stop='physical-evidence-control-regression',paired=paired)
    weights=((1.,1.,0.),(1.,1.,.5),(1.,1.,1.),(1.,.5,1.))
    boundary_grid=[dict(weights=list(w),metrics=boundary_scores(replay,index,replacement=w)) for w in weights]
    write_json(output/'boundary-grid-all4.json',boundary_grid)
    winner=min(range(4),key=lambda i:(-boundary_grid[i]['metrics']['macro_f1'],boundary_grid[i]['metrics']['macro_mae'],i))
    local,radar,semantic=weights[winner]
    controller=replace(COLLECTION_CONTROLLER,boundary_local_weight=local,boundary_radar_weight=radar,boundary_semantic_weight=semantic)
    final=collect_candidate_episodes(frontend,stage,index,output/'boundary-unique-complete-replay',role='dev',
        controller_config=controller,evidence_config=evidence)
    final_metric=score_episodes(final['episodes'])
    final_probe=probe_collection(stage,frontend,final,output/'final-current-probe.json')
    final_boundary=boundary_scores(final,index)
    write_json(output/'final-complete-metrics.json',dict(episodes=final_metric,boundary=final_boundary))
    if not final_metric['eligible'] or final_probe['passed'] is not True:
        return dict(passed=False,stop='unique-boundary-closed-loop-replay',metrics=final_metric,probe=final_probe)
    calibration=dict(selected=selected,all_configurations=grid,split='development-episodes',
        source=final['manifest']['provenance'],max_false_confirm_rate=.05,probe_criteria_sha256=final_probe['criteria_sha256'],
        selected_index=selected_index,complete_policy_replay_passed=True,final_metrics=final_metric,
        final_collection_sha256=final['manifest']['collection_sha256'])
    report=dict(passed=True,controller_config=asdict(controller),evidence_config=asdict(evidence),probe=final_probe,
        calibration=calibration,boundary=final_boundary,paired=paired)
    write_json(output/'calibration-result.json',report)
    return report
