"""Fine-grid autonomous B3 replay and a unified calibration Pareto frontier."""
import argparse
import csv
import json
from pathlib import Path


GRID = {0.75: tuple(n/1000 for n in range(1000,1251,25)),
        1.0: tuple(n/1000 for n in range(850,1001,25))}
REFINEMENT = {0.75: (1.140, 1.145, 1.155), 1.0: (.880, .885, .890, .895)}


def frontier(rows):
    out = []
    for row in sorted(rows,key=lambda r:(r['fp_min'],-r['recall'],-r['end_recall'])):
        if not out or row['recall']>out[-1]['recall']:
            out.append(row)
    return out


def sweep(args):
    import numpy as np
    import torch
    from continuous_mmwave.event_gating.data import sha256, write_json
    from continuous_mmwave.event_gating.mvp.data import load_records, split_fit_calibration
    from continuous_mmwave.event_gating.mvp.model import rollout
    from continuous_mmwave.event_gating.mvp.runner import load_checkpoint, modules, record_identity
    from sweep_saef_tau_threshold import scoring_spec, score, summarize

    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    model = args.run/'model.pt'
    digest = sha256(model)
    saved = load_checkpoint(model,args.device)
    config, action, _, state = modules(saved,args.device)
    records = load_records(json.loads(args.config.read_text()))
    _,calibration,_ = split_fit_calibration(records)
    assert record_identity(records,config)==saved['protocol']['record_identity']
    assert {r['file']:r['provenance']['continuous_hidden']['sha256'] for r in records}==saved['protocol']['source_hashes']
    assert [r['file'] for r in calibration]==saved['protocol']['calibration_files']
    assert saved['calibration']['B3']=={'mechanism':'lif','threshold':1.,'tau':1.}
    previous = json.loads(args.previous.read_text())
    assert previous['checkpoint_sha256']==digest
    reference = {(r['tau'],r['threshold']):r['calibration'] for r in previous['closed_loop']}
    budget = previous['fixed_fp_cap']
    grid = GRID if args.coarse is None else REFINEMENT
    all_thresholds = {tau:sorted(set(GRID[tau]) | set(grid[tau])) for tau in GRID}
    calibration_inputs = []
    for r in calibration:
        x = torch.as_tensor(np.array(r['x'],copy=True),device=args.device,dtype=torch.float32)
        ends = {i['stop'] for i in r['instances'] if 0<i['stop']<r['frames']}
        calibration_inputs.append((r['file'],x,scoring_spec(r,config.frame_period_s),ends))
    plan = dict(tolerance_s=.5, fp_budget=budget, target_recall=.90,
                thresholds_by_tau=all_thresholds, calibration_files=[r['file'] for r in calibration],
                checkpoint_sha256=digest, mode='closed_loop', optimizer_steps=0,
                validation_rollouts=0)
    write_json(args.output/'protocol.json',plan)
    rows, per_record, checks = [], [], []
    if args.coarse is not None:
        coarse = json.loads((args.coarse/'summary.json').read_text())
        for key in ('tolerance_s','fp_budget','target_recall','calibration_files','checkpoint_sha256',
                    'mode','optimizer_steps','validation_rollouts'):
            assert coarse[key]==plan[key]
        rows = coarse['rows']
        checks = coarse['reproduced_previous_points']
        assert {(r['tau'],r['threshold']) for r in rows} == {(tau,h) for tau,hs in GRID.items() for h in hs}
        with (args.coarse/'per-record.csv').open(newline='') as f:
            per_record = list(csv.DictReader(f))
    for tau,thresholds in grid.items():
        for threshold in thresholds:
            scores = []
            for file,x,spec,ends in calibration_inputs:
                trace = rollout(x,action,state,config,mechanism='lif',threshold=threshold,tau=tau)['trace']
                result = score(trace['fire'],spec,ends=ends)
                scores.append(result)
                per_record.append(dict(file=file,tau=tau,threshold=threshold,**result))
                if tau==threshold==1.:
                    ref = args.previous.parent/(Path(file).stem+'-calibration-reference.npz')
                    with np.load(ref) as original:
                        np.testing.assert_array_equal(trace['fire'],original['fire'])
            summary = summarize(scores)
            row = dict(tau=tau,threshold=threshold,**summary,
                       end_recall=summary['end_hits']/summary['end_total'])
            if (tau,threshold) in reference:
                assert all(row[k]==reference[tau,threshold][k] for k in ('tp','fp','fn','overseg','unknown'))
                checks.append([tau,threshold])
            rows.append(row)
            print(f"{tau:.2f}/{threshold:.3f}: recall {row['recall']:.2%}, end {row['end_recall']:.2%}, FP/min {row['fp_min']:.2f}",flush=True)
    assert sha256(model)==digest and len(rows)==sum(map(len,all_thresholds.values())) and len(checks)==4
    rows.sort(key=lambda r:(r['tau'],r['threshold']))
    a = [r for r in rows if r['tau']==.75]
    b = [r for r in rows if r['tau']==1.]
    best = lambda group: max(group,key=lambda r:(r['recall'],r['end_recall'],-r['fp_min']),default=None)
    result = dict(**plan,rows=rows,frontier=frontier(rows),reproduced_previous_points=checks,
                  baseline=next(r for r in b if r['threshold']==1.),
                  a_under_cap=best(r for r in a if r['fp_min']<=budget+1e-12),
                  a_fp_15_to_16=best(r for r in a if 15<=r['fp_min']<=16),
                  a_nearest_cap=min(a,key=lambda r:(abs(r['fp_min']-budget),-r['recall'])),
                  b_at_recall_90=min((r for r in b if r['recall']>=.90),
                                     key=lambda r:(r['fp_min'],-r['recall']),default=None),
                  a_at_recall_90=min((r for r in a if r['recall']>=.90),
                                     key=lambda r:(r['fp_min'],-r['recall']),default=None))
    for name,data in [('grid.csv',rows),('per-record.csv',per_record)]:
        with (args.output/name).open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    write_json(args.output/'summary.json',result)


def plot(args):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    s = json.loads(args.summary.read_text())
    fig, ax = plt.subplots(figsize=(8.8,5.1),layout='constrained')
    for tau,color in ((1.,'#2256a0'),(.75,'#117960')):
        rows = sorted((r for r in s['rows'] if r['tau']==tau),key=lambda r:r['fp_min'])
        ax.plot([r['fp_min'] for r in rows],[r['recall'] for r in rows],
                'o--',color=color,label=f'{tau:g} s threshold grid',lw=1,ms=5,alpha=.8)
    f = s['frontier']
    ax.step([r['fp_min'] for r in f],[r['recall'] for r in f],where='post',
            color='#23262e',lw=1.45,label='Best recall at or below budget')
    ax.axvline(s['fp_budget'],color='#666',ls=':',lw=1)
    ax.axhline(s['target_recall'],color='#666',ls=':',lw=1)
    for key,label,offset in [('baseline','Original B3',(6,8)),('a_nearest_cap','0.75 s, nearest budget',(6,-16)),
                             ('b_at_recall_90','1 s, lowest FP at 90%',(-115,12))]:
        r = s[key]
        if r:
            ax.scatter(r['fp_min'],r['recall'],marker='*',s=105,color='#23262e',zorder=4)
            ax.annotate(f"{label}\nθ={r['threshold']:.3f}",(r['fp_min'],r['recall']),
                        xytext=offset,textcoords='offset points',fontsize=8)
    ax.set(title='B3 closed-loop Pareto curve on six calibration recordings',
           xlabel='False fires / annotated minute',ylabel='Boundary recall (±0.5 s)')
    ax.set_ylim(top=max(r['recall'] for r in s['rows'])+.025)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=.2)
    ax.legend(loc='lower right',fontsize=8)
    fig.savefig(args.figure,dpi=200)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    run = sub.add_parser('sweep')
    for name in ('run','config','previous','output'):
        run.add_argument('--'+name, type=Path, required=True)
    run.add_argument('--device',default='cpu')
    run.add_argument('--coarse',type=Path,help='Existing coarse sweep to extend with the locked refinement grid')
    draw = sub.add_parser('plot')
    draw.add_argument('summary',type=Path)
    draw.add_argument('figure',type=Path)
    args = parser.parse_args()
    (sweep if args.command=='sweep' else plot)(args)


if __name__=='__main__':
    main()
