"""Frozen stream feature export and annotation-only instance views for EventFormer."""
from pathlib import Path
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset
from .model import SegmentClassifier
from .streaming import FrozenRDStream
from .preprocess import sha256,write_json
from .training import classification_metrics


class EventDataset(Dataset):
    def __init__(self,rows):
        if not rows or any(r['role']=='test' for r in rows): raise ValueError('empty data or forbidden test data')
        self.rows=rows; self.cache={}
    def __len__(self): return len(self.rows)
    def __getitem__(self,index):
        r=self.rows[index]; p=r['cells_path']
        if p not in self.cache: self.cache[p]=np.load(p,mmap_mode='r')
        x=np.array(self.cache[p][r['start']:r['stop']],copy=True,dtype=np.float32)
        if x.shape!=(r['stop']-r['start'],4,64) or not np.isfinite(x).all(): raise ValueError('invalid cells')
        return {'cells':torch.from_numpy(x),'label':r['label'],'subject':r['subject'],'uid':r['uid']}


def event_collate(items):
    lengths=[len(r['cells']) for r in items]; b=len(items); length=max(lengths)
    cells=torch.zeros(b,length,4,64); mask=torch.zeros(b,length,dtype=torch.bool)
    for i,r in enumerate(items): cells[i,:lengths[i]]=r['cells']; mask[i,:lengths[i]]=True
    return {'cells':cells,'mask':mask,'labels':torch.tensor([r['label'] for r in items]),
            'subjects':[r['subject'] for r in items],'uids':[r['uid'] for r in items]}


def event_rows(records):
    rows=[]
    for r in records:
        if r['role']=='test': raise ValueError('test records forbidden')
        for instance in r['instances']:
            rows.append({'role':r['role'],'subject':r['subject'],'uid':f"{r['file']}:{instance['start']}:{instance['stop']}",
                         'start':instance['start'],'stop':instance['stop'],'label':instance['label'],
                         'instance_id':instance['instance_id'],'cells_path':r['cells_path']})
    return rows


def prepare_features(config,output,device='cuda'):
    output=Path(output); baseline=Path(config['baseline_output']); checkpoint=Path(config['radar_checkpoint'])
    if sha256(checkpoint)!=config['radar_checkpoint_sha256']: raise ValueError('radar checkpoint changed')
    normalizer_path=baseline/'normalizers.npz'
    if sha256(normalizer_path)!=config['normalizers_sha256']: raise ValueError('normalizers changed')
    old_records=json.loads((baseline/'canonical-manifest.json').read_text()); assert len(old_records)==35
    if any(r['role']!=('validation' if r['subject']=='subject7' else 'train') or r['subject'] not in {f'subject{i}' for i in range(1,8)} for r in old_records):
        raise ValueError('unexpected subject split')
    if sum(len(r['instances']) for r in old_records if r['role']=='train')!=3018 or sum(len(r['instances']) for r in old_records if r['role']=='validation')!=511:
        raise ValueError('unexpected instance coverage')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    original=SegmentClassifier('rd').eval(); original.load_state_dict(state['model'])
    frontend=FrozenRDStream(original).to(device)
    with np.load(normalizer_path) as n: mean=n['rd_mean']; scale=n['rd_scale']
    records=[]; reference={}; max_stream_gap=0.; source_hashes=[]
    cache_root=output/'features'; cache_root.mkdir(exist_ok=False)
    with torch.no_grad():
        for number,r in enumerate(old_records):
            if r['role'] not in ('train','validation'): raise ValueError('test cache access forbidden')
            p=Path(r['features']['rd'])
            if sha256(p)!=r['artifacts']['rd.npy']['sha256']: raise ValueError('source RD cache changed')
            source_hashes.append({'file':r['file'],'rd_sha256':r['artifacts']['rd.npy']['sha256']})
            x=np.load(p,mmap_mode='r'); length=len(x); dest=cache_root/Path(r['file']).stem; dest.mkdir()
            cells=np.lib.format.open_memmap(dest/'cells.npy',mode='w+',dtype='float32',shape=(length,4,64))
            vectors=np.lib.format.open_memmap(dest/'frame_features.npy',mode='w+',dtype='float32',shape=(length,64))
            continuous=np.lib.format.open_memmap(dest/'continuous_hidden.npy',mode='w+',dtype='float32',shape=(length,128))
            oracle=np.lib.format.open_memmap(dest/'oracle_hidden.npy',mode='w+',dtype='float32',shape=(length,128)); oracle[:]=0
            carried=None
            for s in range(0,length,64):
                end=min(length,s+64); batch=torch.from_numpy(np.clip((np.array(x[s:end])-mean)/scale,-8,8))[None].to(device)
                result=frontend.forward_chunk(batch,state=carried); carried=result['state']
                cells[s:end]=result['cells'][0].cpu().numpy(); vectors[s:end]=result['frame_features'][0].cpu().numpy()
                continuous[s:end]=result['hidden'][0].cpu().numpy()
            # Explicit irregular-boundary equivalence on the first recording.
            if number==0:
                sample=torch.from_numpy(np.clip((np.array(x[:137])-mean)/scale,-8,8))[None].to(device)
                full=frontend.forward_chunk(sample); hidden=[]; carry=None
                for s,e in [(0,1),(1,32),(32,65),(65,137)]:
                    current=frontend.forward_chunk(sample[:,s:e],state=carry); hidden.append(current['hidden']); carry=current['state']
                gap=float((torch.cat(hidden,1)-full['hidden']).abs().max()); max_stream_gap=max(max_stream_gap,gap)
                if gap>1e-4: raise RuntimeError(f'irregular stream mismatch: {gap}')
            for instance in r['instances']:
                s,e=instance['start'],instance['stop']; carry=None
                for first in range(s,e,64):
                    last=min(e,first+64); features=torch.from_numpy(np.array(vectors[first:last]))[None].to(device)
                    encoded=original.temporal(features,state=carry); carry=encoded.state
                    oracle[first:last]=encoded.frame_hidden[0].cpu().numpy()
                if r['role']=='validation':
                    uid=f"{r['file']}:{s}:{e}"; entry={}
                    for mode,array in [('oracle',oracle),('continuous',continuous)]:
                        for slot,size in [('end',e-s),('u3',6),('u6',12),('u12',24)]:
                            if e-s<size: continue
                            pooled=torch.from_numpy(np.array(array[s:s+size])).to(device).mean(0,keepdim=True)
                            entry[mode+'_'+slot]=original.classifier(pooled)[0].cpu().numpy()
                    reference[uid]={'label':instance['label'],'subject':r['subject'],'logits':entry}
            for array in (cells,vectors,continuous,oracle): array.flush()
            del cells,vectors,continuous,oracle
            artifacts={q.name:{'bytes':q.stat().st_size,'sha256':sha256(q)} for q in dest.glob('*.npy')}
            for q in dest.glob('*.npy'):
                arr=np.load(q,mmap_mode='r')
                for s in range(0,len(arr),1024):
                    if not np.isfinite(arr[s:s+1024]).all(): raise ValueError('nonfinite frozen cache')
            new={'file':r['file'],'role':r['role'],'subject':r['subject'],'frames':length,'instances':r['instances'],
                 'cells_path':str(dest/'cells.npy'),'continuous_path':str(dest/'continuous_hidden.npy'),
                 'oracle_path':str(dest/'oracle_hidden.npy'),'artifacts':artifacts,'artifact_dir':str(dest)}
            records.append(new); write_json(dest/'complete.json',new)
            write_json(output/'status.json',{'phase':'frozen_feature_export','recordings_completed':len(records),'recordings_total':35,'updated_unix':time.time()})
            print(f"frozen features {len(records)}/35 {r['file']}",flush=True)
    teacher_npz=baseline/'runs'/'rd-seed42'/'validation-best.npz'
    with np.load(teacher_npz,allow_pickle=False) as teacher:
        logits=np.stack([reference[uid]['logits']['oracle_end'] for uid in teacher['uids']])
        labels=np.array([reference[uid]['label'] for uid in teacher['uids']]); gap=float(np.abs(logits-teacher['logits']).max())
        changed=int(np.count_nonzero(logits.argmax(1)!=teacher['logits'].argmax(1)))
        if gap>1e-4 or changed: raise RuntimeError(f'pretrained classifier mismatch: {gap}, {changed}')
    arrays={}; metrics={}
    for mode in ('oracle','continuous'):
        for slot in ('end','u3','u6','u12'):
            tag=mode+'_'+slot; ids=[uid for uid,row in reference.items() if tag in row['logits']]
            scores=np.stack([reference[uid]['logits'][tag] for uid in ids]); y=np.array([reference[uid]['label'] for uid in ids])
            subjects=np.array([reference[uid]['subject'] for uid in ids])
            metrics[tag]=classification_metrics(scores,y,subjects)
            for key,value in [('logits',scores),('labels',y),('subjects',subjects),('uids',np.array(ids))]: arrays[tag+'_'+key]=value
    np.savez_compressed(output/'frozen-reference.npz',**arrays)
    write_json(output/'frozen-reference-metrics.json',metrics)
    verification={'passed':True,'source_checkpoint_sha256':sha256(checkpoint),'source_normalizers_sha256':sha256(normalizer_path),
                  'classifier_max_abs':gap,'classifier_argmax_changes':changed,'stream_chunk_max_abs':max_stream_gap,
                  'recordings':len(records),'frames':sum(r['frames'] for r in records),'test_signal_values_read':False,'source_rd_hashes':source_hashes}
    write_json(output/'streaming-verification.json',verification); write_json(output/'feature-manifest.json',records)
    return records
