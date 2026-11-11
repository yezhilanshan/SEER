"""Experiment A: replay saved B3 evidence with only the LIF time constant changed."""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout


TAUS = (0.25, 0.5, 0.75, 1.0)
THRESHOLD = 1.0


def read(path):
    return json.loads(path.read_text())


def replay(evidence, period, tau, threshold=THRESHOLD):
    decay = math.exp(-period/tau)
    voltage = np.zeros(len(evidence), dtype=np.float64)
    fire = np.zeros(len(evidence), dtype=bool)
    state = 0.0
    for frame in range(1, len(evidence)):
        candidate = decay*state + (1-decay)*float(evidence[frame])
        voltage[frame] = candidate
        fire[frame] = candidate >= threshold
        state = 0.0 if fire[frame] else candidate
    return voltage, fire


def boundary_summary(metrics, tolerance, baseline):
    totals = Counter()
    delays, paired, paired_original = [], [], []
    for file, metric in metrics.items():
        b = metric['boundary'][tolerance]
        totals.update({k:b[k] for k in ('tp','fp','fn','valid_duration_s','unscored_unknown','unknown_duration_s')})
        previous = {m['truth']:m['delay_s'] for m in baseline[file]['boundary'][tolerance]['matches']}
        for match in b['matches']:
            delay = match['delay_s']
            delays.append(delay)
            if match['truth'] in previous:
                paired.append(abs(delay))
                paired_original.append(abs(previous[match['truth']]))
    tp, fp, fn = (totals[k] for k in ('tp','fp','fn'))
    return dict(totals, recall=tp/(tp+fn), precision=tp/(tp+fp), f1=2*tp/(2*tp+fp+fn),
                fp_per_min=60*fp/totals['valid_duration_s'],
                unknown_per_min=60*totals['unscored_unknown']/totals['unknown_duration_s'],
                delay_mean_s=float(np.mean(delays)), abs_delay_mean_s=float(np.mean(np.abs(delays))),
                abs_delay_p90_s=float(np.quantile(np.abs(delays),.9)),
                paired_n=len(paired), paired_mae_s=float(np.mean(paired)),
                paired_baseline_mae_s=float(np.mean(paired_original)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--diagnosis', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    protocol = read(args.run/'protocol.json')
    calibration = read(args.run/'calibration.json')['parameters']['B3']
    assert calibration == {'mechanism':'lif','threshold':THRESHOLD,'tau':1.0}
    assert protocol['config']['minimum_duration_s'] is None
    period = protocol['config']['frame_period_s']
    lock = read(args.config)['locks']['feature_manifest']
    manifest = Path(lock['path']).read_bytes()
    assert hashlib.sha256(manifest).hexdigest() == lock['sha256']
    files = set(protocol['validation_files'])
    records = [r for r in json.loads(manifest) if r['file'] in files]
    assert len(records) == len(files) == 4 and all(r['role']=='validation' for r in records)
    with (args.diagnosis/'endpoints.csv').open(newline='') as f:
        cases = [r for r in csv.DictReader(f) if r['failure']=='Integration Failure']
    integration = {(r['file'], int(r['stop'])) for r in cases}
    assert len(integration) == 49 and {f for f,_ in integration} <= files
    args.output.mkdir(exist_ok=False)
    metrics = {tau:{} for tau in TAUS}
    for record in records:
        file = record['file']
        stem = Path(file).stem
        with np.load(args.run/'B3'/(stem+'-trace.npz')) as trace:
            frame, evidence = trace['frame'], trace['evidence']
            assert len(frame) == len(evidence) == record['frames']
            assert np.isfinite(evidence).all()
            for tau in TAUS:
                voltage, fire = replay(evidence, period, tau)
                if tau == 1.0:
                    np.testing.assert_array_equal(fire, trace['fire'])
                    np.testing.assert_allclose(voltage, trace['voltage'], rtol=0, atol=1e-12)
                result = {'trace': {'frame':frame,'fire':fire}, 'events':[]}
                metrics[tau][file] = evaluate_rollout(result, record, period)
                np.savez_compressed(args.output/f'{stem}-tau-{tau}.npz', frame=frame, voltage=voltage, fire=fire)
    hit = {tau:{(file,m['truth']):m['delay_s'] for file,metrics_file in group.items()
                for m in metrics_file['boundary']['0.5']['matches']} for tau,group in metrics.items()}
    ends = {(r['file'],i['stop']) for r in records for i in r['instances'] if 0<i['stop']<r['frames']}
    baseline = metrics[1.0]
    assert not integration.intersection(hit[1.0]) and len(ends)==511 and len(ends.intersection(hit[1.0]))==413
    rows = []
    for tau in TAUS:
        b = boundary_summary(metrics[tau], '0.5', baseline)
        over = Counter()
        for m in metrics[tau].values():
            o = m['oversegmentation']
            over.update({k:o[k] for k in ('fires_inside_instances','extra_fires_inside_instances','affected_instances')})
        found = integration.intersection(hit[tau])
        rows.append(dict(tau=tau, integration_recovered=len(found), integration_total=len(integration),
                         end_hits=len(ends.intersection(hit[tau])),
                         end_new=len(ends.intersection(hit[tau].keys()-hit[1.0].keys())),
                         end_lost=len(ends.intersection(hit[1.0].keys()-hit[tau].keys())),
                         boundary=b, boundary_02=boundary_summary(metrics[tau], '0.2', baseline),
                         oversegmentation=dict(over)))
    assert rows[-1]['boundary']['tp']==885 and rows[-1]['boundary']['fp']==224
    with (args.output/'integration_cases.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['file','stop_frame',*[f'tau_{tau}_matched_delay_s' for tau in TAUS]])
        for file,stop in sorted(integration):
            writer.writerow([file,stop,*[hit[tau].get((file,stop)) for tau in TAUS]])
    result = dict(open_loop=True, optimizer_steps=0, threshold=THRESHOLD, baseline_replayed=True,
                  matching_tolerance_s=.5, recordings=len(records), rows=rows)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,allow_nan=False))


if __name__ == '__main__':
    main()
