"""Official change anchors, unknown masks and recording-isolated calibration."""
import numpy as np


def reset_targets(meta):
    frames=int(meta['frames']); n=frames//2
    ids=np.full(frames,-1,dtype=np.int64); labels=np.full(frames,-100,dtype=np.int64); points=[]; previous=0
    for index,row in enumerate(sorted(meta['instances'],key=lambda r:r['start'])):
        s,e=int(row['start']),int(row['stop'])
        if not previous<=s<e<=frames: raise ValueError('invalid official instances')
        ids[s:e]=index; labels[s:e]=row['label']; points.extend((s,e)); previous=e
    paired=ids[:2*n].reshape(n,2); semantic=labels[:2*n:2].copy()
    semantic[(paired[:,0]<0)|(paired[:,0]!=paired[:,1])]=-100
    boundary=np.zeros(n,dtype=bool)
    for point in points:
        if point//2<n: boundary[point//2]=True
    valid=semantic>=0
    for t in np.flatnonzero(boundary): valid[max(0,t-1):t+2]=False
    valid|=boundary
    groups={}
    for point in sorted(set(points)): groups.setdefault(point//2,[]).append(point)
    scorable={t:anchors for t,anchors in groups.items() if 0<t<n}
    return {'labels':semantic,'boundary':boundary,'valid':valid,'known_frames':ids>=0,
            'truth':np.array([min(v) for _,v in sorted(scorable.items())],dtype=np.int64),
            'raw_truth':np.array(sorted(set(points)),dtype=np.int64),
            'anchor_groups':[{'update':t,'anchors':v,'scorable':0<t<n} for t,v in sorted(groups.items())],
            'merged_anchor_count':sum(len(v)-1 for v in scorable.values()),
            'excluded_recording_edge_anchors':sum(len(v) for t,v in groups.items() if t not in scorable),
            'instances':meta['instances']}


def split_policy_records(records):
    groups={}
    for rid,r in enumerate(records):
        if r['role']=='train': groups.setdefault(r['subject'],[]).append(rid)
    if not groups: raise ValueError('no train recordings')
    calibration=[]
    for ids in groups.values():
        if len(ids)<2: raise ValueError('two recordings per training subject required')
        calibration.append(max(ids,key=lambda i:records[i]['file']))
    return [i for i,r in enumerate(records) if r['role']=='train' and i not in calibration],sorted(calibration)


def stream_schedule(records,ids,steps,chunk,seed):
    return np.array(list(iter_stream_schedule(records,ids,steps,chunk,seed)),dtype=np.int64)


def iter_stream_schedule(records,ids,steps,chunk,seed):
    rng=np.random.default_rng(seed); order=rng.permutation(ids).tolist(); positions={i:0 for i in ids}
    for step in range(steps):
        rid=order[step%len(order)]; start=positions[rid]; stop=min(start+chunk,len(records[rid]['motion']))
        yield rid,start,stop
        positions[rid]=0 if stop==len(records[rid]['motion']) else stop
