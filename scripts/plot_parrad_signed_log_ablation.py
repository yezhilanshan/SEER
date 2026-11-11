"""Plot actual results using the existing system Matplotlib (Python 3.5 compatible)."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(p.read_text())


def main(root):
    for seed in [17,29,43]:
        pair=root if seed==17 else root/('seed-'+str(seed))
        if not all((pair/arm/'results.json').exists() for arm in ['P0','P1']):continue
        assets=pair/'assets';assets.mkdir(exist_ok=True)
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        for arm in ['P0','P1']:
            h=read(pair/arm/'history.json');x=[v['epoch'] for v in h]
            axes[0].plot(x,[v['train_loss'] for v in h],label=arm)
            axes[1].plot(x,[v['validation']['subject_macro_f1']*100 for v in h],label=arm)
        axes[0].set_title('Train weighted CE');axes[1].set_title('Validation subject mean Macro-F1 (%)')
        for ax in axes:ax.set_xlabel('Epoch');ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(str(assets/'curves.png'),dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(13,6))
        for ax,arm in zip(axes,['P0','P1']):
            cm=np.array(read(pair/arm/'results.json')['validation']['confusion_matrix'],dtype=float)
            norm=np.divide(cm,np.maximum(cm.sum(1,keepdims=True),1))
            image=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
            ax.set_title(arm+' validation: row-normalized');ax.set_xlabel('Predicted class ID');ax.set_ylabel('True class ID')
            ax.set_xticks(range(14));ax.set_yticks(range(14))
            for i in range(14):
                for j in range(14):
                    if norm[i,j]>.02:ax.text(j,i,'{:.0f}'.format(norm[i,j]*100),ha='center',va='center',fontsize=6,color='white' if norm[i,j]>.5 else 'black')
            fig.colorbar(image,ax=ax,fraction=.046)
        fig.tight_layout();fig.savefig(str(assets/'confusion.png'),dpi=160);plt.close(fig)
        for arm in ['P0','P1']:
            # Per-arm complete curves and raw 14x14 counts remain alongside checkpoints.
            h=read(pair/arm/'history.json');result=read(pair/arm/'results.json')['validation']
            np.savetxt(str(pair/arm/'confusion-matrix.csv'),np.array(result['confusion_matrix']),fmt='%d',delimiter=',')
            (pair/arm/'per-class.json').write_text(json.dumps(result['per_class'],indent=2))
            fig,axes=plt.subplots(1,3,figsize=(13,4))
            x=[v['epoch'] for v in h]
            axes[0].plot(x,[v['train_loss'] for v in h]);axes[0].set_title(arm+' train weighted CE')
            axes[1].plot(x,[100*v['validation']['subject_macro_f1'] for v in h]);axes[1].set_title('Validation subject mean F1 (%)')
            axes[2].plot(x,[100*v['validation']['frame_accuracy'] for v in h]);axes[2].set_title('Validation accuracy (%)')
            for ax in axes:ax.set_xlabel('Epoch');ax.grid(alpha=.2)
            fig.tight_layout();fig.savefig(str(pair/arm/'training-curves.png'),dpi=160);plt.close(fig)
            fig,ax=plt.subplots(figsize=(7,6));cm=np.array(result['confusion_matrix'],dtype=float)
            image=ax.imshow(cm/np.maximum(cm.sum(1,keepdims=True),1),vmin=0,vmax=1,cmap='Blues',interpolation='nearest')
            ax.set_title(arm+' validation confusion');ax.set_xticks(range(14));ax.set_yticks(range(14))
            ax.set_xlabel('Predicted class ID');ax.set_ylabel('True class ID');fig.colorbar(image,ax=ax)
            fig.tight_layout();fig.savefig(str(pair/arm/'confusion-matrix.png'),dpi=160);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(Path(p.parse_args().root))
