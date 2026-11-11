"""Screen frozen B3 evidence on calibration recordings, then replay closed-loop."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path

import numpy as np
import torch

from continuous_mmwave.event_gating.data import sha256, write_json
from continuous_mmwave.event_gating.metrics import match_ordered
from continuous_mmwave.event_gating.mvp.data import load_records, split_fit_calibration
from continuous_mmwave.event_gating.mvp.model import rollout
from continuous_mmwave.event_gating.mvp.runner import load_checkpoint, modules, record_identity
from replay_saef_lif_tau import TAUS, read, replay


THRESHOLDS = tuple(n/100 for n in range(75, 301, 5))
TOLERANCE = .5
TARGET_RECALL = .90


def scoring_spec(record, period):
    truth = record['boundaries']
    known = record['known_frames']
    valid = known.copy()
    valid[1:] |= known[:-1]
    valid[0] = False
    interior = np.zeros(len(known), dtype=bool)
    for instance in record['instances']:
        interior[instance['start']+1:instance['stop']] = True
    return truth, valid, interior, period


def score(fire, spec, ends=None):
    truth, valid, interior, period = spec
    fires = np.flatnonzero(fire).tolist()
    pairs = match_ordered(fires, truth, lambda i,j:-abs(fires[i]-truth[j]),
                          lambda i,j:abs(fires[i]-truth[j])*period <= TOLERANCE+1e-12)
    used = {i for i,_ in pairs}
    unmatched = np.asarray([f for i,f in enumerate(fires) if i not in used], dtype=np.int64)
    fp = int(valid[unmatched].sum())
    delays = [abs(fires[i]-truth[j])*period for i,j in pairs]
    result = dict(tp=len(pairs), fp=fp, fn=len(truth)-len(pairs),
                  valid_s=float(valid.sum()*period), unknown_s=float((~valid).sum()*period),
                  unknown=len(unmatched)-fp, overseg=int(interior[unmatched].sum()), abs_delay_sum=sum(delays))
    if ends is not None:
        result.update(end_hits=sum(truth[j] in ends for _,j in pairs), end_total=len(ends))
    return result


def summarize(rows):
    total = Counter()
    for row in rows:
        total.update(row)
    tp,fp,fn = (total[k] for k in ('tp','fp','fn'))
    return dict(total, recall=tp/(tp+fn), fp_min=60*fp/total['valid_s'],
                unknown_per_min=60*total['unknown']/total['unknown_s'],
                f1=2*tp/(2*tp+fp+fn), mae_s=total['abs_delay_sum']/tp if tp else None)


def grid_record(file, evidence, spec):
    return file, {(tau,threshold):score(replay(evidence,spec[-1],tau,threshold)[1],spec)
                  for tau in TAUS for threshold in THRESHOLDS}


def select(grid, budget):
    out = []
    for tau in sorted(TAUS, reverse=True):
        rows = [r for r in grid if r['tau']==tau]
        at_budget = max((r for r in rows if r['fp_min']<=budget+1e-12),
                        key=lambda r:(r['recall'],-r['fp_min'],r['threshold']))
        feasible = [r for r in rows if r['recall']>=TARGET_RECALL]
        at_recall = min(feasible,key=lambda r:(r['fp_min'],-r['recall'],-r['threshold'])) if feasible else None
        out.append(dict(tau=tau,fixed_fp=at_budget,fixed_recall=at_recall))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--workers',type=int,default=6)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    checkpoint = args.run/'model.pt'
    digest = sha256(checkpoint)
    saved = load_checkpoint(checkpoint,args.device)
    config,action,_,state = modules(saved,args.device)
    records = load_records(read(args.config))
    _,calibration,validation = split_fit_calibration(records)
    protocol = saved['protocol']
    assert record_identity(records,config) == protocol['record_identity']
    assert {r['file']:r['provenance']['continuous_hidden']['sha256'] for r in records} == protocol['source_hashes']
    assert [r['file'] for r in calibration] == protocol['calibration_files']
    assert [r['file'] for r in validation] == protocol['validation_files']
    assert saved['calibration']['B3'] == {'mechanism':'lif','threshold':1.0,'tau':1.0}
    assert config.minimum_duration_s is None
    assert {r['file'] for r in calibration}.isdisjoint(r['file'] for r in validation)
    specifications = {r['file']:scoring_spec(r,config.frame_period_s) for r in calibration+validation}

    def closed(record, tau, threshold):
        x = torch.as_tensor(np.array(record['x'],copy=True),device=args.device,dtype=torch.float32)
        return rollout(x,action,state,config,mechanism='lif',threshold=threshold,tau=tau)['trace']

    frozen = {}
    for r in calibration:
        file = r['file']
        t = frozen[file] = closed(r,1.,1.)
        voltage,fire = replay(t['evidence'],config.frame_period_s,1.,1.)
        np.testing.assert_array_equal(fire,t['fire'])
        np.testing.assert_allclose(voltage,t['voltage'],atol=1e-12,rtol=0)
        np.savez_compressed(args.output/(Path(file).stem+'-calibration-reference.npz'),**t)
    print('Original B3 frozen on calibration recordings; 1.0/1.0 reproduced.',flush=True)

    grid_scores = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(grid_record,file,t['evidence'],specifications[file]) for file,t in frozen.items()]
        for job in as_completed(jobs):
            file,values = job.result()
            grid_scores[file] = values
            print(f'Calibration grid recordings scored: {len(grid_scores)}/{len(jobs)}',flush=True)
    grid = [dict(tau=tau,threshold=threshold,**summarize(
                [grid_scores[r['file']][tau,threshold] for r in calibration]))
            for tau in TAUS for threshold in THRESHOLDS]
    baseline = next(r for r in grid if r['tau']==r['threshold']==1.)
    selected = select(grid,baseline['fp_min'])
    frozen_plan = dict(checkpoint_sha256=digest, matching_tolerance_s=TOLERANCE, taus=TAUS,
                thresholds=THRESHOLDS, fixed_fp_cap=baseline['fp_min'], fixed_recall_target=TARGET_RECALL,
                calibration_recordings=len(calibration), validation_recordings=len(validation),
                selection=selected, open_loop_grid=grid, open_loop_baseline=baseline,
                weighting='pooled boundary counts and annotated valid duration',
                optimizer_steps=0, validation_used_for_selection=False)
    write_json(args.output/'selection-locked.json',frozen_plan)
    print('Selection locked before any validation replay.',flush=True)

    keys = {(1.,1.)}
    for row in selected:
        keys.update((row['tau'],choice['threshold']) for choice in (row['fixed_fp'],row['fixed_recall']) if choice)
    results = []
    for tau,threshold in sorted(keys,reverse=True):
        values = {}
        for name,group in [('calibration',calibration),('validation',validation)]:
            scores = []
            for record in group:
                file = record['file']
                trace = closed(record,tau,threshold)
                scored = score(trace['fire'],specifications[file])
                if tau==threshold==1. and name=='validation':
                    with np.load(args.run/'B3'/(Path(file).stem+'-trace.npz')) as original:
                        np.testing.assert_array_equal(trace['fire'],original['fire'])
                    previous = read(args.run/'B3'/(Path(file).stem+'-metrics.json'))['boundary']['0.5']
                    assert all(scored[k]==previous[k] for k in ('tp','fp','fn'))
                scores.append(scored)
            values[name] = summarize(scores)
        results.append(dict(tau=tau,threshold=threshold,**values))
        print(f'Closed loop completed: tau={tau} threshold={threshold}',flush=True)
    assert sha256(checkpoint)==digest
    write_json(args.output/'summary.json',dict(**frozen_plan,closed_loop=results,
               validation_baseline_fire_reproduced=True))


if __name__=='__main__':
    main()
