"""Plot aggregate open-loop calibration curves and selected closed-loop points."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', type=Path)
    parser.add_argument('figure', type=Path)
    args = parser.parse_args()
    data = json.loads(args.summary.read_text(encoding='utf-8-sig'))
    cap, target = data['fixed_fp_cap'], data['fixed_recall_target']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), layout='constrained', sharey=True)
    colors = dict(zip(sorted(data['taus'], reverse=True), ('#2256a0','#117960','#b85d10','#9c3d77')))
    selected = {(r['tau'],choice['threshold']) for r in data['selection']
                for choice in (r['fixed_fp'],r['fixed_recall']) if choice}
    for tau, color in colors.items():
        rows = [r for r in data['open_loop_grid'] if r['tau']==tau]
        best = -1.
        frontier = []
        for row in sorted(rows, key=lambda r:(r['fp_min'],-r['recall'])):
            if row['recall'] > best:
                frontier.append(row)
                best = row['recall']
        xs = [r['fp_min'] for r in frontier]
        ys = [r['recall'] for r in frontier]
        axes[0].scatter([r['fp_min'] for r in rows],[r['recall'] for r in rows],color=color,s=9,alpha=.16)
        axes[0].step(xs,ys,where='post',color=color,label=f'{tau:g} s',linewidth=1.7)
        for result in data['closed_loop']:
            if result['tau']!=tau:
                continue
            original = next(r for r in rows if r['threshold']==result['threshold'])
            if (tau,result['threshold']) in selected:
                axes[0].scatter(original['fp_min'],original['recall'],color=color,edgecolor='white',s=43,zorder=3)
            closed = result['calibration']
            axes[1].scatter(closed['fp_min'],closed['recall'],color=color,edgecolor='white',s=43,zorder=3)
            axes[1].annotate('', xy=(closed['fp_min'],closed['recall']),
                             xytext=(original['fp_min'],original['recall']),
                             arrowprops=dict(arrowstyle='->',color=color,lw=1,alpha=.55))
    base = data['open_loop_baseline']
    axes[0].scatter(base['fp_min'],base['recall'],marker='*',color='#222222',s=112,zorder=4,label='Original B3')
    fixed = [r['fixed_recall']['fp_min'] for r in data['selection'] if r['fixed_recall']]
    closed_fp = [r['calibration']['fp_min'] for r in data['closed_loop']]
    limit = max([cap*1.5, *fixed, *closed_fp])*1.08
    for ax, title in zip(axes,('Frozen evidence: threshold sweep','Locked points: open to closed loop')):
        ax.set(title=title,xlabel='False fires / annotated minute',xlim=(0,limit),ylim=(.65,1.01))
        ax.axvline(cap,color='#646464',linestyle='--',linewidth=.9)
        ax.axhline(target,color='#646464',linestyle=':',linewidth=1)
        ax.grid(alpha=.2)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_ylabel('Boundary recall (±0.5 s)')
    axes[0].legend(loc='lower right',title='LIF time constant',fontsize=8)
    fig.suptitle('B3 operating points on six training-subject calibration recordings',fontsize=13)
    fig.savefig(args.figure,dpi=200)


if __name__=='__main__':
    main()
