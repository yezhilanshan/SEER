"""Official interval contracts, fit-only normalization and complete-instance loading."""
from __future__ import annotations

import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset

CLASSES = ('standing_up','sitting_down','going_out_of_room','entering_room','putting_on_cloth',
           'putting_off_cloth','drumming','shaking','swiping_left','swiping_right','thumb_down','thumb_up')


def subject_role(subject: str, splits: dict) -> str:
    flat = [s for values in splits.values() for s in values]
    if len(flat) != len(set(flat)) or set(splits) != {'train','validation','test'}:
        raise ValueError('subject split must be disjoint and contain train/validation/test')
    roles = [role for role, subjects in splits.items() if subject in subjects]
    if len(roles) != 1:
        raise ValueError(f'subject absent or ambiguous in split: {subject}')
    return roles[0]


def read_instances(path: str | Path, lengths: dict[str,int]) -> list[dict]:
    result=[]
    with Path(path).open(newline='',encoding='utf-8-sig') as f:
        for r in csv.reader(f):
            if not r: continue
            if len(r)!=5: raise ValueError('instance CSV must have five columns')
            file,start,stop,cls,subject=r
            start,stop=int(start),int(stop)
            if (Path(file).name!=file or '/' in file or '\\' in file or file not in lengths
                    or not 0<=start<stop<=lengths[file] or cls not in CLASSES or file.split('_')[0]!=subject):
                raise ValueError(f'invalid annotation: {r}')
            result.append({'file':file,'start':start,'stop':stop,'label':CLASSES.index(cls),'class':cls,'subject':subject})
    if not result: raise ValueError('empty annotation CSV')
    result.sort(key=lambda r:(r['file'],r['start'],r['stop']))
    previous={}
    for i,r in enumerate(result):
        if r['start']<previous.get(r['file'],-1): raise ValueError('overlapping or duplicate instance')
        previous[r['file']]=r['stop']; r['instance_id']=i
    return result


def frame_annotations(instances: list[dict], file: str, length: int) -> tuple[np.ndarray,np.ndarray]:
    labels=np.full(length,-100,np.int16); ids=np.full(length,-1,np.int32)
    for r in instances:
        if r.get('file',file)!=file: continue
        s,e=r['start'],r['stop']
        if not 0<=s<e<=length or np.any(ids[s:e]>=0): raise ValueError('invalid instance interval')
        labels[s:e]=r['label']; ids[s:e]=r['instance_id']
    return labels,ids


def fit_normalizers(records: list[dict]) -> dict[str,np.ndarray]:
    stats={}; count=0
    for kind in ('md','rd'):
        total=square=None; n=0
        for r in records:
            if r['role']!='train': continue
            x=np.load(r['features'][kind],mmap_mode='r'); label=np.load(r['labels_path'],mmap_mode='r')
            for s in range(0,len(x),1024):
                values=np.asarray(x[s:s+1024][label[s:s+1024]>=0],dtype=np.float64)
                if not len(values): continue
                if not np.isfinite(values).all(): raise ValueError('nonfinite train feature')
                if total is None: total=np.zeros(values.shape[1:],np.float64); square=total.copy()
                total+=values.sum(0); square+=np.square(values).sum(0); n+=len(values)
        if not n: raise ValueError('no annotated fit frames')
        mean=total/n; scale=np.sqrt(np.maximum(square/n-mean**2,0)).clip(1e-6)
        stats[kind+'_mean']=mean.astype(np.float32); stats[kind+'_scale']=scale.astype(np.float32)
        if count and n!=count: raise ValueError('feature fit frames differ')
        count=n
    stats['fit_frames']=np.array(count,dtype=np.int64)
    return stats


class ClipDataset(Dataset):
    def __init__(self, rows: list[dict], normalizer: dict, arm: str):
        if arm not in ('md','rd') or not rows: raise ValueError('empty dataset or invalid arm')
        if any(r['role']=='test' for r in rows): raise ValueError('test signal arrays are forbidden in this experiment')
        self.rows=rows; self.mean=np.asarray(normalizer['mean'],np.float32); self.scale=np.asarray(normalizer['scale'],np.float32)
        if np.any(self.scale<=0): raise ValueError('normalization scale must be positive')
        self.cache={}

    def __len__(self): return len(self.rows)

    def __getitem__(self,index):
        r=self.rows[index]; path=r['feature_path']
        if path not in self.cache: self.cache[path]=np.load(path,mmap_mode='r')
        x=np.array(self.cache[path][r['start']:r['stop']],dtype=np.float32,copy=True)
        if len(x)!=r['stop']-r['start'] or not len(x): raise ValueError('incomplete instance')
        x=np.clip((x-self.mean)/self.scale,-8,8)
        if not np.isfinite(x).all(): raise ValueError('nonfinite normalized instance')
        return {'x':torch.from_numpy(x),'label':int(r['label']),'uid':r['uid'],'subject':r['subject'],'instance_id':r['instance_id']}


def collate_clips(items: list[dict]) -> dict:
    lengths=[len(r['x']) for r in items]
    x=items[0]['x'].new_zeros((len(items),max(lengths))+items[0]['x'].shape[1:])
    mask=torch.zeros(len(items),max(lengths),dtype=torch.bool)
    for i,r in enumerate(items): x[i,:lengths[i]]=r['x']; mask[i,:lengths[i]]=True
    return {'x':x,'mask':mask,'labels':torch.tensor([r['label'] for r in items]),
            'uids':[r['uid'] for r in items],'subjects':[r['subject'] for r in items],
            'instance_ids':torch.tensor([r['instance_id'] for r in items])}


def clip_rows(records: list[dict], arm: str) -> list[dict]:
    result=[]
    for record in records:
        if record['role']=='test': raise ValueError('test canonical record forbidden')
        for r in record['instances']:
            result.append({**r,'role':record['role'],'uid':f"{record['file']}:{r['start']}:{r['stop']}",
                           'feature_path':record['features'][arm]})
    return result
