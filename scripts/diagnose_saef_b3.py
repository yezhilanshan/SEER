"""Read-only model replay and endpoint attribution for an existing SAEF run."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
import torch

from continuous_mmwave.event_gating.data import sha256
from continuous_mmwave.event_gating.metrics import match_ordered
from continuous_mmwave.event_gating.mvp.data import load_records
from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
from continuous_mmwave.event_gating.mvp.model import rollout
from continuous_mmwave.event_gating.mvp.runner import load_checkpoint, modules, record_identity


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def replay(features, action, head, config, parameters, reset=None):
    # Capture the actual pre-update state supplied to EvidenceNet at each frame.
    states = torch.empty_like(features)
    index = 0

    def capture(_module, args):
        nonlocal index
        states[index].copy_(args[2])
        index += 1

    handle = head.register_forward_pre_hook(capture)
    try:
        result = rollout(features, action, head, config, **parameters, teacher_boundaries=reset)
    finally:
        handle.remove()
    assert index == len(features)
    trace = result['trace']
    trace['state_before'] = states.cpu().numpy()
    x = features.cpu().numpy()
    np.testing.assert_allclose(np.linalg.norm(x-trace['state_before'], axis=1),
                               trace['state_distance'], atol=2e-6, rtol=2e-6)
    # The age supplied at frame t is since the preceding Fire, excluding any Fire at t.
    frame = trace['frame']
    last = np.maximum.accumulate(np.where(trace['fire'], frame, 0))
    trace['event_age_s'] = (frame - np.r_[0, last[:-1]]) * config.frame_period_s
    return result


def timing_pairs(fires, truth, strict, period, outer):
    used_fires = {m['predicted'] for m in strict}
    used_truth = {m['truth'] for m in strict}
    remaining_fires = sorted(set(fires) - used_fires)
    remaining_truth = sorted(set(truth) - used_truth)

    def gap(i, j):
        return abs(remaining_fires[i]-remaining_truth[j]) * period

    # Do not reuse a Fire already assigned by the official metric, or an
    # out-of-window Fire that is closer to another annotated endpoint.
    nearest = [min(abs(f-b) for b in truth) for f in remaining_fires] if truth else []
    pairs = match_ordered(remaining_fires, remaining_truth, lambda i, j: -gap(i, j),
                          lambda i, j: .5 < gap(i, j) <= outer+1e-12
                          and abs(remaining_fires[i]-remaining_truth[j]) == nearest[i])
    return {remaining_truth[j]: remaining_fires[i] for i, j in pairs}


def endpoint_trace(trace, x, stop, half):
    left = max(0, stop-half)
    right = min(len(x), stop+half+1)
    window = slice(left, right)
    before = max(0, stop-2*half)
    d = trace['state_distance']
    drift = None
    if stop-half > before and stop+half <= len(x):
        old = x[before:stop-half].mean(0)
        post = x[stop:stop+half].mean(0)
        direction = post-old
        scale = float(direction @ direction)
        if scale > 1e-10:
            # Positive change indicates movement along the old-to-post feature direction.
            drift = float((trace['state_before'][stop]-trace['state_before'][left]) @ direction / scale)
    return dict(peak_i=float(trace['evidence'][window].max()),
                peak_v=float(trace['voltage'][window].max()),
                i_at_end=float(trace['evidence'][stop]), v_at_end=float(trace['voltage'][stop]),
                age_s=float(trace['event_age_s'][stop]), d_at_end=float(d[stop]),
                d_change=float(d[stop]-d[left]), state_drift=drift)


def classify(nearby, delayed, evidence, voltage, theta):
    # Priority is observable competition, then a unique outside-window Fire,
    # then adequate drive without threshold crossing, then weak drive.
    if nearby:
        return 'Matching Failure'
    if delayed is not None:
        return 'Timing Failure'
    if voltage >= theta:
        raise AssertionError('voltage crossed threshold without Fire in the strict window')
    return 'Integration Failure' if evidence >= theta else 'Evidence Failure'


def summarize(rows, arms):
    misses = [r for r in rows if not r['B3_hit']]
    out = {'endpoints': len(rows), 'B3_misses': len(misses),
           'failure_counts': dict(Counter(r['failure'] for r in misses)),
           'timing_radius_counts': {str(t): dict(Counter(r[f'failure_{t}'] for r in misses))
                                    for t in (.75, 1.5)},
           'by_arm': {}, 'by_failure': {}}
    for arm in arms:
        recovered = sum(r[arm+'_hit'] for r in misses)
        lost = sum(r['B3_hit'] and not r[arm+'_hit'] for r in rows)
        out['by_arm'][arm] = dict(hit=sum(r[arm+'_hit'] for r in rows), recovered=recovered, lost=lost)
        for field in ('age_s', 'gt_duration_s'):
            counts = []
            for low, high in ((0,.5),(.5,1),(1,2),(2,float('inf'))):
                key = field if field == 'gt_duration_s' else arm+'_'+field
                group = [r for r in rows if low <= r[key] < high]
                counts.append([low, None if np.isinf(high) else high, len(group), sum(not r[arm+'_hit'] for r in group)])
            out['by_arm'][arm][field+'_bins_total_miss'] = counts
    for failure in out['failure_counts']:
        group = [r for r in misses if r['failure']==failure]
        out['by_failure'][failure] = {'n':len(group), 'oracle_recovered':sum(r['B3_GT_hit'] for r in group),
             'start_only_recovered':sum(r['B3_START_hit'] for r in group), 'B2_hit':sum(r['B2_hit'] for r in group),
             'timing_early':sum(r['timing_delay_s'] is not None and r['timing_delay_s']<0 for r in group),
             'timing_late':sum(r['timing_delay_s'] is not None and r['timing_delay_s']>0 for r in group)}
    for name, group in [('all_misses', misses), ('all_hits', [r for r in rows if r['B3_hit']])]:
        out[name] = {}
        for arm in arms:
            out[name][arm] = {k: float(np.median(values)) if values else None
                for k in ('peak_i','peak_v','i_at_end','v_at_end','age_s','d_at_end','d_change','state_drift')
                for values in [[r[arm+'_'+k] for r in group if r[arm+'_'+k] is not None]]}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    ckpt = args.run/'model.pt'
    before = sha256(ckpt)
    saved = load_checkpoint(ckpt, args.device)
    config, action, local, state = modules(saved, args.device)
    records = load_records(read(args.config))
    if record_identity(records, config) != saved['protocol']['record_identity']:
        raise ValueError('checkpoint and recording identity disagree')
    expected = saved['protocol']['source_hashes']
    if {r['file']: r['provenance']['continuous_hidden']['sha256'] for r in records} != expected:
        raise ValueError('checkpoint and continuous feature hashes disagree')
    files = set(saved['protocol']['validation_files'])
    arm_names = ('B2','B3','B3_GT','B3_START')
    period = config.frame_period_s
    half = int(.5 / period)
    theta = saved['calibration']['B3']['threshold']
    if config.minimum_duration_s is not None:
        raise ValueError('this diagnostic assumes threshold crossing alone triggers LIF')
    endpoint_rows = []
    replay_check = []
    per_record = []
    for record in records:
        if record['file'] not in files:
            continue
        stem = Path(record['file']).stem
        x = np.array(record['x'], dtype=np.float32)
        features = torch.as_tensor(x, device=args.device)
        starts = sorted({v['start'] for v in record['instances'] if 0<v['start']<len(x)})
        arms = {'B2':(local,'B2',None), 'B3':(state,'B3',None),
                'B3_GT':(state,'B3',record['boundaries']), 'B3_START':(state,'B3',starts)}
        traces, matches = {}, {}
        for arm, (head, parameters, reset) in arms.items():
            result = replay(features, action, head, config, saved['calibration'][parameters], reset)
            trace = traces[arm] = result['trace']
            if arm in ('B2','B3'):
                with np.load(args.run/arm/(stem+'-trace.npz')) as official:
                    np.testing.assert_array_equal(trace['fire'], official['fire'])
                    for k in ('evidence','voltage','state_distance'):
                        np.testing.assert_allclose(trace[k], official[k], atol=2e-5, rtol=2e-5)
                replay_check.append(f'{stem}/{arm}')
            metric = evaluate_rollout(result, record, period)
            matches[arm] = metric['boundary']['0.5']['matches']
            per_record.append(dict(file=record['file'], arm=arm,
                boundary={k:v for k,v in metric['boundary']['0.5'].items() if k!='matches'},
                event_f1_iou_05=metric['events']['0.5']['f1']))
            np.savez_compressed(args.output/(stem+'-'+arm+'.npz'), **trace)
        hit = {arm:{m['truth'] for m in match} for arm, match in matches.items()}
        competing = {m['predicted']:m['truth'] for m in matches['B3']}
        fires = np.flatnonzero(traces['B3']['fire']).tolist()
        timing = {r:timing_pairs(fires, record['boundaries'], matches['B3'], period, r) for r in (.75,1.,1.5)}
        for instance in record['instances']:
            b = instance['stop']
            if not 0 < b < len(x):
                continue
            row = dict(file=record['file'], instance_id=instance['instance_id'], label=instance['label'],
                       start=instance['start'], stop=b, stop_s=b*period, gt_duration_s=(b-instance['start'])*period)
            for arm, trace in traces.items():
                row[arm+'_hit'] = b in hit[arm]
                row.update({arm+'_'+k:v for k,v in endpoint_trace(trace,x,b,half).items()})
            nearby = [f for f in fires if abs(f-b)*period <= .5+1e-12]
            row['nearby_fire_frames'] = nearby
            row['nearby_claimed_by'] = [competing.get(f) for f in nearby]
            row['timing_fire_frame'] = timing[1.].get(b)
            row['timing_delay_s'] = None if b not in timing[1.] else (timing[1.][b]-b)*period
            for radius, tag in ((.75,'failure_0.75'),(1.,'failure'),(1.5,'failure_1.5')):
                row[tag] = 'Matched' if row['B3_hit'] else classify(nearby, timing[radius].get(b),
                            row['B3_peak_i'], row['B3_peak_v'], theta)
            endpoint_rows.append(row)
        print(f'{stem}: B2/B3 replay verified, GT/all and GT/start counterfactuals saved', flush=True)
    summary = summarize(endpoint_rows, arm_names)
    if summary['endpoints'] != 511 or summary['B3_misses'] != 98 or len(replay_check) != 8:
        raise ValueError('expected 511 official end boundaries and 98 B3 misses from run_v1')
    assert sum(summary['failure_counts'].values()) == 98 and sha256(ckpt) == before
    with (args.output/'endpoints.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(endpoint_rows[0]))
        writer.writeheader()
        writer.writerows(endpoint_rows)
    write(args.output/'summary.json', {**summary,'checkpoint_sha256':before,'threshold':theta,
          'calibrated_B3':saved['calibration']['B3'],'replays_verified':replay_check,
          'counterfactual_diagnostic_only':True,'optimizer_steps':0,'per_record':per_record})


if __name__ == '__main__':
    main()
