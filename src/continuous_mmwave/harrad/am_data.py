"""Train-only reference selection and fixed causal semantic diagnostic panels."""
import numpy as np
import torch
from .joint_data import collate_origins


def select_bank(records,per_class=32,classes=tuple(range(12))):
    rng=np.random.default_rng(314159); result=[]
    for label in classes:
        pools=[]
        for rid,r in enumerate(records):
            if r['role']!='train': continue
            candidates=r['legal'][r['semantic'][r['legal']]==label].copy(); rng.shuffle(candidates)
            if len(candidates): pools.append((rid,candidates.tolist()))
        if not pools: raise ValueError(f'no train reference support for class {label}')
        for index in range(max(len(v) for _,v in pools)):
            for rid,values in pools:
                if index<len(values): result.append((rid,int(values[index]),int(label)))
                if sum(row[2]==label for row in result)>=per_class: break
            if sum(row[2]==label for row in result)>=per_class: break
    return np.asarray(result,dtype=np.int64)


def diagnostic_panels(records,design,train_size=512):
    selected=design['validation'][design['validation'][:,3]==1].copy()
    train=[i for i,r in enumerate(records) if r['role']=='train']
    if not train: raise ValueError('train panel requires training recordings')
    rng=np.random.default_rng(271828); rows=[]
    for index,rid in enumerate(train):
        r=records[rid]; candidates=r['legal'][r['semantic'][r['legal']]>=0]
        quota=train_size//len(train)+(index<train_size%len(train))
        for t in rng.choice(candidates,min(len(candidates),quota),replace=False): rows.append((rid,int(t),0,1))
    return {'train':np.array(sorted(rows),dtype=np.int64),'validation':selected}


@torch.no_grad()
def build_bank(model,records,stats,rows,device,batch_size=16):
    if rows.ndim!=2 or rows.shape[1]!=3 or len(set(map(tuple,rows[:,:2].tolist())))!=len(rows):
        raise ValueError('invalid or duplicate reference bank origins')
    vectors=[]
    for rid,t,label in rows:
        if records[rid]['role']!='train' or label<0 or records[rid]['semantic'][t]!=label:
            raise ValueError('nontrain or incorrectly labelled reference bank')
    for start in range(0,len(rows),batch_size):
        batch=collate_origins(records,rows[start:start+batch_size,:2],stats)
        vectors.append(model.reference_vectors(batch['cells'].to(device),batch['mask'].to(device)).detach())
    return {'vectors':torch.cat(vectors),'labels':torch.as_tensor(rows[:,2],device=device),
            'origins':rows.copy()}
