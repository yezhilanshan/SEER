"""Standalone scientific figure for the predeclared scene comparisons."""
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot(evidence):
    rows=json.loads((evidence/'scene-bootstrap-summary.json').read_text())
    scenes=['all','long_stable','fast_proxy','slow_proxy','repeated','boundary','transient_proxy']
    labels=['All recordings','Long stable candidate','Fast change proxy','Slow transition proxy',
            'Same-class repeated','Near GT boundary','Transient proxy (2 recordings)']
    fig,axes=plt.subplots(1,2,figsize=(12,5.4),sharey=True)
    for ax,key in zip(axes,('mse','nll')):
        for j,(h,color) in enumerate(zip((.2,.6,1.),('#0072B2','#D55E00','#009E73'))):
            chosen=[next(r for r in rows if r['scene']==scene and r['horizon']==h)[key]['paired'] for scene in scenes]
            centers=np.array([r['mean'] for r in chosen]);bounds=np.array([r['ci95'] for r in chosen])
            ax.errorbar(centers,np.arange(len(scenes))+(j-1)*.2,
                xerr=np.stack((centers-bounds[:,0],bounds[:,1]-centers)),fmt='o',markersize=4,
                capsize=2,color=color,label=f'{h:.1f} s')
        ax.axvline(0,color='0.3',linestyle='--',linewidth=1)
        ax.set_xlabel(f'Event-conditioned minus GRU {key.upper()} (lower is better)')
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.set_yticks(np.arange(len(scenes)),labels)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis();axes[1].legend(loc='upper left',frameon=False)
    fig.suptitle('Stage 1 diagnostic: paired recording Bootstrap 95% intervals',fontsize=13)
    fig.text(.5,.015,'Subject7 only; 3 fixed training seeds; retrospective overlapping proxies; no multiplicity correction.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.95))
    fig.savefig(evidence/'scene-comparison.png',dpi=180)
    fig.savefig(evidence/'scene-comparison.pdf')


if __name__=='__main__':plot(Path(sys.argv[1]))
