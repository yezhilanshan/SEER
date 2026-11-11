"""Audit controls and validation diagnostics for the RD Signed-Log ablation."""
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Sampler

from .diagnostics import BED_CLASSES, classification_metrics, confusion, steady_mask


def parameter_sha(model):
    """Stable tensor-content SHA, including BN buffers, independent of checkpoint metadata."""
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        x=value.detach().cpu().contiguous()
        digest.update(key.encode()+b'\0'+str(x.dtype).encode()+b'\0'+str(tuple(x.shape)).encode()+b'\0')
        digest.update(x.numpy().tobytes())
    return digest.hexdigest()


class PairedWindowSampler(Sampler):
    """Original recording-local shuffle policy, with an isolated RNG for paired runs."""
    def __init__(self,data,seed):
        self.groups={}
        for i,w in enumerate(data.windows):self.groups.setdefault(w[0],[]).append(i)
        self.size=len(data);self.generator=torch.Generator().manual_seed(seed);self.orders=[]

    def __len__(self):return self.size

    def __iter__(self):
        groups=list(self.groups.values());order=[]
        for gi in torch.randperm(len(groups),generator=self.generator).tolist():
            g=groups[gi]
            order.extend(g[i] for i in torch.randperm(len(g),generator=self.generator).tolist())
        self.orders.append(order)
        return iter(order)


def validation_diagnostics(data,logits_path,radius=.3):
    """Full-recording boundary masks; no test values and no window-edge blind spots."""
    with np.load(logits_path,allow_pickle=False) as arrays:
        logits,labels,offsets=(arrays[k] for k in ('logits','labels','offsets'))
    if len(offsets)!=len(data.windows)+1:raise ValueError('logits/window mismatch')
    masks={};sources={};matrix=np.zeros((14,14),np.int64);steady=matrix.copy()
    subjects=defaultdict(lambda:np.zeros((14,14),np.int64));sensors=defaultdict(lambda:np.zeros((14,14),np.int64))
    for i,(rid,start,stop,_,_) in enumerate(data.windows):
        row=data.recordings.rows[rid]
        if rid not in masks:
            path=Path(row['path']);path=path if path.is_absolute() else data.recordings.manifest.parent/path
            with np.load(path,allow_pickle=False) as a:
                sources[rid]=a['labels'];masks[rid]=steady_mask(a['labels'],a['timestamps'],radius)
        lo,hi=offsets[i:i+2];y=labels[lo:hi];pred=logits[lo:hi].argmax(-1)
        if not np.array_equal(y,sources[rid][start:stop]):raise ValueError('saved labels/window mismatch')
        cm=confusion(y,pred);matrix+=cm;steady+=confusion(y,pred,masks[rid][start:stop])
        subjects[row['subject_id']]+=cm;sensors[row['sensor_id']]+=cm
    pooled=classification_metrics(matrix)
    return dict(pooled=pooled,steady=classification_metrics(steady),steady_radius_seconds=radius,
                steady_scope='full-recording transitions, before slicing; window edges are not artificial transitions',
                bed_class_ids=list(BED_CLASSES),bed_macro_f1=float(np.mean([pooled['per_class'][i]['f1'] for i in BED_CLASSES])),
                by_subject={k:classification_metrics(v) for k,v in sorted(subjects.items())},
                by_sensor={k:classification_metrics(v) for k,v in sorted(sensors.items())})
