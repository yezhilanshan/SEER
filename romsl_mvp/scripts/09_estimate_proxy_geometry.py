import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from romsl.proxy_geometry import axis_rotations, rigid_fit, fit_icp, nearest_bones, angle_degrees

EDGES=[(0,1),(1,2),(2,3),(0,4),(4,5),(5,6),(0,7),(7,8),(8,9),(9,10),(8,11),(11,12),(12,13),(8,14),(14,15),(15,16)]


def collect(root,subjects,stride=24,limit=16):
    rng=np.random.default_rng(20261106)
    poses,points,frame_indices,record_ids,roots,centers=[],[],[],[],[],[]
    skipped=0
    for subject in subjects:
        for action in range(1,28):
            base=root/'E01'/subject/f'A{action:02}'
            pose=np.load(base/'ground_truth.npy')
            for frame in range(12,len(pose)-3,stride):
                p=np.fromfile(base/'mmwave'/f'frame{frame+1:03}.bin',dtype='<f8').reshape(-1,5)[:,:3]
                p=p[np.isfinite(p).all(-1)]
                if len(p)<6:
                    skipped+=1
                    continue
                median=np.median(p,axis=0)
                distance=np.linalg.norm(p-median,axis=1)
                p=p[distance<=max(.3,3*np.median(distance))]
                if len(p)<6 or not np.isfinite(pose[frame]).all():
                    skipped+=1
                    continue
                if len(p)>limit:
                    p=p[rng.choice(len(p),limit,replace=False)]
                index=len(poses)
                poses.append(pose[frame])
                points.append(p)
                frame_indices.extend([index]*len(p))
                record_ids.extend([f'E01/{subject}/A{action:02}']*len(p))
                roots.append(pose[frame,[1,4]].mean(0))
                centers.append(np.median(p,axis=0))
    pose=np.stack(poses).astype(float)
    points=np.concatenate(points)
    index=np.asarray(frame_indices)
    ids=np.asarray(record_ids)
    unique,counts=np.unique(ids,return_counts=True)
    count_map=dict(zip(unique,counts))
    weights=np.array([1/count_map[key] for key in ids])
    a=pose[index][:,[i for i,_ in EDGES]]
    b=pose[index][:,[j for _,j in EDGES]]
    return {'a':a,'b':b,'points':points,'weights':weights,'ids':ids,'pose':pose,
            'frame_indices':index,'roots':np.stack(roots),'centers':np.stack(centers),'skipped':skipped}


def metrics(data,rotation,translation):
    _,distance=nearest_bones(data['a'],data['b'],data['points'],rotation,translation)
    center=np.linalg.norm(data['roots']@rotation.T+translation-data['centers'],axis=-1)
    return {'point_to_bone_mean_m':float(distance.mean()),'point_to_bone_median_m':float(np.median(distance)),
            'point_to_bone_p95_m':float(np.quantile(distance,.95)),
            'fraction_within_20cm':float((distance<.2).mean()),
            'root_to_cloud_center_median_m':float(np.median(center)),
            'per_recording_median_m':{key:float(np.median(distance[data['ids']==key])) for key in sorted(set(data['ids']))}}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    train=collect(args.root,[f'S{i:02}' for i in range(1,7)])
    val=collect(args.root,['S07','S08'])
    centroid_r,centroid_t=rigid_fit(train['roots'],train['centers'],np.ones(len(train['roots'])))
    candidates=[]
    for index,rotation in enumerate(axis_rotations()+[centroid_r]):
        translation=np.median(train['centers']-train['roots']@rotation.T,axis=0)
        r,t,score=fit_icp(train['a'],train['b'],train['points'],train['weights'],rotation,translation,iterations=35)
        candidates.append({'start':index,'R':r.tolist(),'t':t.tolist(),'score':score})
        print('CANDIDATE',index,score,flush=True)
    candidates.sort(key=lambda c:c['score'])
    winner=candidates[0]
    rotation,translation=np.array(winner['R']),np.array(winner['t'])
    rotation,translation,score=fit_icp(train['a'],train['b'],train['points'],train['weights'],rotation,translation,iterations=100)
    stability=[]
    for subject in [f'S{i:02}' for i in range(1,7)]:
        keep=np.array([key.split('/')[1]!=subject for key in train['ids']])
        r,t,s=fit_icp(train['a'][keep],train['b'][keep],train['points'][keep],train['weights'][keep],
                      rotation,translation,iterations=60)
        stability.append({'held_out_training_subject':subject,'rotation_delta_deg':angle_degrees(r,rotation),
                          'translation_delta_m':float(np.linalg.norm(t-translation)),
                          'origin_world_m':(-r.T@t).tolist(),'score':s})
        print('STABILITY',subject,stability[-1]['rotation_delta_deg'],flush=True)
    sensitivity=[]
    for trim in [.5,.9]:
        r,t,s=fit_icp(train['a'],train['b'],train['points'],train['weights'],
                      rotation,translation,iterations=60,trim=trim)
        sensitivity.append({'trim':trim,'R':r.tolist(),'t':t.tolist(),'score':s,
            'rotation_delta_deg':angle_degrees(r,rotation),'translation_delta_m':float(np.linalg.norm(t-translation))})
    rng=np.random.default_rng(42)
    # Time-shuffle poses only within each validation recording as a diagnostic control.
    shuffled_a,shuffled_b=val['a'].copy(),val['b'].copy()
    for key in sorted(set(val['ids'])):
        keep=np.flatnonzero(val['ids']==key)
        source=rng.permutation(keep)
        shuffled_a[keep]=val['a'][source]
        shuffled_b[keep]=val['b'][source]
    _,shuffled_distance=nearest_bones(shuffled_a,shuffled_b,val['points'],rotation,translation)
    root_singular=np.linalg.svd(train['roots']-train['roots'].mean(0),compute_uv=False)
    result={
        'status':'PROXY_GEOMETRY_ONLY','verified':False,'from_frame':'world','to_frame':'radar',
        'translation_unit':'m','coordinate_unit_note':'nominal meter interpretation from earlier audit',
        'R':rotation.tolist(),'t':translation.tolist(),'radar_origin_world_m':(-rotation.T@translation).tolist(),
        'source':'training-only robust point-to-bone ICP, unknown scattering correspondences',
        'fit_subjects':[f'S{i:02}' for i in range(1,7)],'validation_subjects':['S07','S08'],
        'test_subjects_accessed':[],
        'sampling':{'frame_stride':24,'max_points_per_frame':16,'robust_trim':.7},
        'train_frames':len(train['roots']),'train_points':len(train['points']),'val_frames':len(val['roots']),'val_points':len(val['points']),
        'train':metrics(train,rotation,translation),'validation':metrics(val,rotation,translation),
        'centroid_baseline':{'R':centroid_r.tolist(),'t':centroid_t.tolist(),'train':metrics(train,centroid_r,centroid_t),
                             'validation':metrics(val,centroid_r,centroid_t)},
        'root_trajectory_singular_values':root_singular.tolist(),'candidates':candidates,
        'leave_one_subject_out':stability,'trim_sensitivity':sensitivity,
        'validation_shuffled_bone_mean_m':float(shuffled_distance.mean()),
        'validation_shuffled_bone_median_m':float(np.median(shuffled_distance)),
        'elapsed_sec':time.monotonic()-started,
        'limitation':'Closest skeletal segments are surrogate correspondences; translation can absorb skin/scatterer bias. Not a verified physical calibration.'
    }
    (args.output/'proxy_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PROXY_COMPLETE',json.dumps({key:result[key] for key in ['R','t','radar_origin_world_m','train_frames','val_frames','elapsed_sec']}),flush=True)


if __name__=='__main__':
    main()
