"""C/D readiness gates, diagnostic weight plots and explicit label export."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from romsl.geometry import load_extrinsics, read_config
from romsl.doppler import load_doppler, echo_support, weight_statistics
from romsl.mmfi_io import load_recording
from romsl.targets import build_cd_labels


def plot_weights(path, weights, masks, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(10,3.5))
    axes[0].hist(weights[masks],bins=np.linspace(.25,1,17),edgecolor='white')
    axes[0].set(xlabel='Echo weight',ylabel='Valid frame count',title=title)
    axes[1].plot(np.flatnonzero(masks),weights[masks],'.',markersize=2)
    axes[1].set(xlabel='Frame index',ylabel='Echo weight')
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)


def plot_action_weights(path, actions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=sorted(actions)
    values=[np.concatenate(actions[key]) for key in labels]
    kept=[(key,value) for key,value in zip(labels,values) if len(value)]
    if not kept:
        return
    fig,ax=plt.subplots(figsize=(12,4))
    ax.boxplot([value for _,value in kept],labels=[key for key,_ in kept],showfliers=False)
    ax.set(xlabel='Action',ylabel='Echo weight',title='Per-action valid-frame echo weights')
    ax.tick_params(axis='x',rotation=60)
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/mvp.yaml')
    parser.add_argument('--extrinsics',type=Path,default=PROJECT/'configs/extrinsics_e01.yaml')
    parser.add_argument('--doppler',type=Path,default=PROJECT/'configs/doppler_e01.yaml')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--split',choices=['train','val'],default='train')
    parser.add_argument('--exp',choices=['C','D'],default='D')
    parser.add_argument('--emit-labels',action='store_true')
    parser.add_argument('--synthetic',action='store_true')
    args=parser.parse_args()
    if args.synthetic and args.emit_labels:
        parser.error('Synthetic diagnostics cannot emit formal labels')
    if args.emit_labels and args.data_root is None:
        parser.error('--emit-labels requires --data-root')
    args.output.mkdir(parents=True,exist_ok=False)
    report={'mode':'synthetic' if args.synthetic else 'readiness',
            'labels_emitted':False,'geometry_status':'BLOCKED_GEOMETRY',
            'doppler_status':'BLOCKED_DOPPLER'}
    geometry, spec = None, None
    for name,path,load in [('geometry',args.extrinsics,load_extrinsics),('doppler',args.doppler,load_doppler)]:
        try:
            value=load(path)
            report[name+'_status']='VERIFIED'
            if name=='geometry':
                geometry=value
            else:
                spec=value
        except ValueError as exc:
            report[name+'_reason']=str(exc)
    if args.synthetic:
        spec={'verified':True,'sign_verified':True,'doppler_col':4,'units':'m/s',
              'positive_direction':'away_from_radar'}
        cases={'variable':[0,1,2,4,8]*20,'saturated':[8]*100}
        report['synthetic_cases']={}
        for name,counts in cases.items():
            frames=[]
            for count in counts:
                points=np.zeros((8,5))
                points[:count,4]=.2
                frames.append(points)
            support=echo_support(frames,spec)
            report['synthetic_cases'][name]=weight_statistics(support)
            plot_weights(args.output/(name+'.png'),support['weights'],support['frame_mask'],
                         'Synthetic '+name)
    elif args.emit_labels and (geometry is None or (args.exp=='D' and spec is None)):
        report['export_status']='BLOCKED'
    elif args.data_root is not None and (spec is not None or args.emit_labels):
        config=read_config(args.config)
        joints=read_config(args.config.with_name('joints.yaml'))
        summaries, all_weights, all_masks, action_weights = {}, [], [], {}
        stage=Path(tempfile.mkdtemp(prefix='.labels-',dir=args.output)) if args.emit_labels else None
        for subject in config['data']['splits'][args.split]:
            for action in range(1,28):
                rec=load_recording(args.data_root/config['data']['scene']/subject/f'A{action:02}')
                if spec is not None:
                    support=echo_support(rec.points_per_frame,spec)
                    summaries[rec.recording_id]=weight_statistics(support)
                    all_weights.append(support['weights'])
                    all_masks.append(support['frame_mask'])
                    action_weights.setdefault(rec.action_id,[]).append(support['weights'][support['frame_mask']])
                if stage is not None:
                    targets=build_cd_labels(rec.pose,rec.points_per_frame,joints,1/config['data']['fps'],
                        args.extrinsics,pose_unit=config['audit']['pose_unit'],
                        doppler_path=args.doppler if args.exp=='D' else None)
                    np.savez(stage/(rec.recording_id.replace('/','_')+'.npz'),**targets)
        if summaries:
            weights,masks=np.concatenate(all_weights),np.concatenate(all_masks)
            report['weights']=weight_statistics({'weights':weights,'frame_mask':masks})
            report['per_recording_weights']=summaries
            by_action={}
            for key,summary in summaries.items():
                by_action.setdefault(key.split('/')[-1],[]).append(summary)
            report['per_action_weights']={action:{
                'recording_mean_weight_mean':float(np.mean([r['mean'] for r in rows if 'mean' in r])),
                'low_variation_recordings':sum(r['status']=='LOW_DIAGNOSTIC_VARIATION' for r in rows),
            } for action,rows in by_action.items() if any('mean' in r for r in rows)}
            plot_weights(args.output/'weight_distribution.png',weights,masks,args.split+' echo weights')
            plot_action_weights(args.output/'per_action_weights.png',action_weights)
        if stage is not None:
            stage.rename(args.output/'labels')
            report['labels_emitted']=True
            report['export_status']='COMPLETE'
    report['C_ready']=geometry is not None
    report['D_semantics_ready']=geometry is not None and spec is not None
    report['D_weight_status']=report.get('weights',{}).get('status','UNMEASURED')
    report['D_ready']=report['D_semantics_ready'] and report['D_weight_status']=='VARIABLE'
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in
                     ('per_recording_weights','per_action_weights')},indent=2))
    if args.emit_labels and not report['labels_emitted']:
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
