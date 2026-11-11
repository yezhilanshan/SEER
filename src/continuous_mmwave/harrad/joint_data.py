"""Frozen CNN targets and common causal Oracle prefixes for HARrad A/B/AB."""
import numpy as np
import torch

HORIZONS = (1, 3, 5)


def make_record(meta, cells, frame_features):
    if meta['role'] not in ('train', 'validation'):
        raise ValueError('forbidden role')
    frames = meta['frames']; n = frames // 2
    if n <= 5 or cells.shape != (frames,4,64) or frame_features.shape != (frames,64):
        raise ValueError('invalid or too-short frozen cache')
    labels = np.full(frames, -100, dtype=np.int64)
    ids = np.full(frames, -1, dtype=np.int64)
    starts = []; last = 0
    for index, row in enumerate(sorted(meta['instances'], key=lambda r:r['start'])):
        s, e = row['start'], row['stop']
        if not last <= s < e <= frames:
            raise ValueError('invalid instance intervals')
        labels[s:e] = row['label']; ids[s:e] = index; last = e
        if s//2 < n:
            starts.append(s//2)
    paired_ids = ids[:2*n].reshape(n,2)
    unknown = (paired_ids < 0).any(1)
    mixed = (~unknown) & (paired_ids[:,0] != paired_ids[:,1])
    semantic = labels[:2*n:2].copy()
    semantic[(paired_ids[:,0] < 0) | (paired_ids[:,0] != paired_ids[:,1])] = -100
    latest = np.zeros(n, dtype=np.int64)
    latest[starts] = starts; latest = np.maximum.accumulate(latest)
    times = np.arange(n)
    crossing = np.stack([np.array([np.any((np.array(starts)>t)&(np.array(starts)<=t+h)) for t in times]) for h in HORIZONS],1)
    return {'file':meta['file'],'role':meta['role'],'subject':meta['subject'],'frames':frames,
            'dropped_tail_frames':frames%2,'cells':cells[:2*n].reshape(n,2,4,64),
            'motion':np.asarray(frame_features[:2*n].reshape(n,2,64).mean(1),dtype='float32'),
            'semantic':semantic,'unknown':unknown,'mixed':mixed,
            'latest_start':latest,'crossing':crossing,'legal':np.arange(n-5,dtype=np.int64)}


def fit_stats(records):
    train = [r for r in records if r['role']=='train']
    if not train:
        raise ValueError('train records required')
    n = sum(len(r['motion']) for r in train)
    mean = sum(r['motion'].astype('float64').sum(0) for r in train)/n
    variance = sum(np.square(r['motion'].astype('float64')).sum(0) for r in train)/n - mean**2
    std = np.sqrt(np.maximum(variance,1e-6))
    errors = []
    for r in train:
        m = (r['motion'].astype('float64')-mean)/std; t=r['legal']
        errors.append(np.stack([np.square(m[t+h]-m[t]).mean(0) for h in HORIZONS]))
    coordinate_mse = np.mean(errors,axis=0)
    persistence = coordinate_mse.mean(1)
    if not np.all(persistence>0):
        raise ValueError('degenerate frozen prediction target')
    return {'mean':mean.astype('float32'),'std':std.astype('float32'),
            'persistence_mse':persistence.astype('float32'),
            'logvar':np.log(np.clip(coordinate_mse,np.exp(-6),np.exp(2))).astype('float32')}


def fixed_design(records, seeds, steps, evaluation_per_record=128):
    train = [i for i,r in enumerate(records) if r['role']=='train']
    eligible = {i:r['legal'][r['semantic'][r['legal']]>=0] for i,r in enumerate(records)}
    semantic_train = [i for i in train if len(eligible[i])]
    if not train or not semantic_train:
        raise ValueError('no eligible training data')
    result = {}
    for seed in seeds:
        rng = np.random.default_rng(seed); schedule=[]
        for _ in range(steps):
            a=int(rng.choice(semantic_train)); b=int(rng.choice(train))
            schedule.append([(a,int(rng.choice(eligible[a]))),(b,int(rng.choice(records[b]['legal'])))])
        result[f'schedule_{seed}']=np.array(schedule,dtype=np.int64)
    selected={}
    for panel,seed in ((0,1042),(1,2042)):
        rng=np.random.default_rng(seed)
        for i,r in enumerate(records):
            if r['role']!='validation': continue
            candidates=r['legal'] if panel==0 else eligible[i]
            for t in rng.choice(candidates,min(evaluation_per_record,len(candidates)),replace=False):
                flags=selected.setdefault((i,int(t)),[0,0]); flags[panel]=1
    result['validation']=np.array([[i,t,*flags] for (i,t),flags in sorted(selected.items())],dtype=np.int64)
    return result


def collate_origins(records, origins, stats):
    parts=[]
    for rid,t in origins:
        r=records[int(rid)]; t=int(t)
        if not 0<=t<len(r['legal']): raise ValueError('illegal common prediction origin')
        start=max(int(r['latest_start'][t]),t-31,0)
        m=(r['motion']-stats['mean'])/stats['std']
        parts.append((r['cells'][start:t+1],m[t],np.stack([m[t+h] for h in HORIZONS]),
                      int(r['semantic'][t]),int(rid),r['crossing'][t],m[max(0,t-1)],
                      np.array([r['unknown'][t+h] for h in HORIZONS]),
                      np.array([r['mixed'][t+h] for h in HORIZONS])))
    length=max(len(p[0]) for p in parts); batch=len(parts)
    cells=torch.zeros(batch,length,2,4,64); mask=torch.zeros(batch,length,dtype=torch.bool)
    for i,p in enumerate(parts):
        cells[i,:len(p[0])]=torch.from_numpy(np.array(p[0],copy=True)); mask[i,:len(p[0])]=True
    return {'cells':cells,'mask':mask,'current':torch.from_numpy(np.stack([p[1] for p in parts])),
            'future':torch.from_numpy(np.stack([p[2] for p in parts])),
            'labels':torch.tensor([p[3] for p in parts]),'record_ids':torch.tensor([p[4] for p in parts]),
            'crossing':torch.from_numpy(np.stack([p[5] for p in parts])),
            'previous':torch.from_numpy(np.stack([p[6] for p in parts])),
            'future_unknown':torch.from_numpy(np.stack([p[7] for p in parts])),
            'future_mixed':torch.from_numpy(np.stack([p[8] for p in parts]))}
