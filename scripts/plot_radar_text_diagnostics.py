"""Render exportable 14-class diagnostic figures from machine-readable results."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


NAMES = ['fall on floor', 'get into bed', 'get out of bed', 'lie in bed', 'roll in bed',
         'sit down on bed', 'sit down on chair', 'sit in bed', 'stand up from bed',
         'stand up from chair', 'stand up from floor', 'walk to bed', 'walk to chair', 'walk to room']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True)
    args = parser.parse_args()
    root = Path(args.results)
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    classification = json.loads((root/'frame-classification.json').read_text())
    csv_rows = []
    for split, report in classification.items():
        for subset in ('all_valid','steady'):
            metrics = report[subset]
            counts = np.array(metrics['confusion_matrix'])
            support = counts.sum(1)
            normalized = np.divide(counts, support[:,None], out=np.zeros_like(counts,dtype=float), where=support[:,None]>0)
            for row in metrics['per_class']:
                csv_rows.append(dict(split=split,subset=subset,action=NAMES[row['class_id']],**row))
            for name,matrix in [('counts',counts),('row-normalized',normalized)]:
                fig,ax = plt.subplots(figsize=(12,10))
                image=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1 if name=='row-normalized' else None)
                colorbar=fig.colorbar(image,ax=ax,fraction=.04,pad=.02)
                colorbar.set_label('Fraction (cell labels: %)' if name=='row-normalized' else 'Frames')
                ax.set(xticks=np.arange(14),yticks=np.arange(14),xticklabels=NAMES,yticklabels=NAMES,
                       xlabel='Predicted class',ylabel='Ground-truth labeled class',
                       title=f'{split}: {subset} ({metrics["frames"]:,} frames)\nMacro-F1={metrics["macro_f1"]:.3f}; frame accuracy={metrics["frame_accuracy"]:.3f}')
                plt.setp(ax.get_xticklabels(),rotation=55,ha='right',fontsize=9)
                plt.setp(ax.get_yticklabels(),fontsize=9)
                threshold=matrix.max()*.55
                for i in range(14):
                    for j in range(14):
                        value=matrix[i,j]
                        text=f'{value*100:.0f}' if name=='row-normalized' else str(value)
                        ax.text(j,i,text,ha='center',va='center',fontsize=7,color='white' if value>threshold else 'black')
                fig.tight_layout()
                stem=figures/f'{split}-{subset}-{name}'
                fig.savefig(str(stem)+'.png',dpi=170)
                fig.savefig(str(stem)+'.svg')
                plt.close(fig)
    with (root/'per-class-frame-metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    probes=json.loads((root/'linear-probes.json').read_text())
    features=list(probes)
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(len(features))
    for offset,split in [(-.18,'validation'),(.18,'test')]:
        bars=ax.bar(x+offset,[probes[key]['results'][split]['macro_f1'] for key in features],width=.35,label=split)
        ax.bar_label(bars,fmt='%.3f',fontsize=9)
    ax.set(xticks=x,xticklabels=features,ylabel='14-class Macro-F1',ylim=(0,1),title='Frozen features, identical linear probes\nTest subject was previously observed; descriptive results')
    ax.legend();fig.tight_layout();fig.savefig(figures/'linear-probes.png',dpi=170);fig.savefig(figures/'linear-probes.svg');plt.close(fig)
    print(json.dumps({'figures':str(figures),'png_files':len(list(figures.glob('*.png')))}))


if __name__=='__main__':
    main()
