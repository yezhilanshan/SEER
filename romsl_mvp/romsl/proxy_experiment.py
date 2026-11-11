"""Explicitly exploratory C/D runner. Formal geometry flags remain false."""
import json
from pathlib import Path

import numpy as np
import torch

from .doppler import weight_statistics
from .experiment import (
    attribute_scores, classification_scores, extract_features, fit_probes, loader,
    pretrain, reserve_test, save_checkpoint, seed_all, verify_lock, write_json,
)
from .model import RadarEncoder, RegressionHead, masked_huber
from .pipeline import Windows, digest
from .targets import compute_targets, compute_echo_weight


def prepare_proxy(job, root):
    geometry=json.loads((job/'proxy_geometry.json').read_text())
    config=json.loads((job/'source/configs/mvp.yaml').read_text())
    joints=json.loads((job/'source/configs/joints.yaml').read_text())
    directory=job/'proxy_targets'
    directory.mkdir(exist_ok=False)
    values,masks,weights,frame_masks=[],[],[],[]
    per_recording={}
    for split in ('train','val'):
        rows=[json.loads(line) for line in (job/f'artifacts/manifest_{split}.jsonl').read_text().splitlines()]
        for identifier in sorted({row['recording_id'] for row in rows}):
            base=root/identifier
            pose=np.load(base/'ground_truth.npy')
            frames=[np.fromfile(base/'mmwave'/f'frame{i+1:03}.bin',dtype='<f8').reshape(-1,5) for i in range(len(pose))]
            targets=compute_targets(pose,joints,1/config['data']['fps'],radial=True,geometry=geometry,allow_proxy=True)
            weight=compute_echo_weight(frames,4,'m/s')
            filename=identifier.replace('/','_')+'.npz'
            with np.load(job/'cache'/filename) as original:
                np.testing.assert_array_equal(targets['C_mask'],original['B_mask'])
                observed=original['point_mask'].any(-1)
            np.savez(directory/filename,C=targets['C'],C_mask=targets['C_mask'],echo_weight=weight)
            if split=='train':
                values.append(targets['C'])
                masks.append(targets['C_mask'])
                weights.append(weight)
                frame_masks.append(observed)
                per_recording[identifier]=weight_statistics({'weights':weight,'frame_mask':observed})
    values,masks=np.concatenate(values),np.concatenate(masks)
    count=masks.sum(0)
    mean=(values*masks).sum(0,dtype=np.float64)/count
    std=np.sqrt(np.maximum((((values-mean)**2)*masks).sum(0)/count,1e-12))
    np.savez(job/'artifacts/proxy_stats.npz',mean=mean,std=std)
    summary=weight_statistics({'weights':np.concatenate(weights),'frame_mask':np.concatenate(frame_masks)})
    write_json(job/'weight_diagnostics.json',{'assumption':'zero-based column 4 interpreted as m/s, not author verified',
        'sign_note':'weights use absolute Doppler; raw sign does not affect weights',
        'summary':summary,'per_training_recording':per_recording})
    return summary


class ProxyWindows(Windows):
    def __init__(self,job,split,exp):
        super().__init__(job,split,'B')
        self.proxy_exp=exp
        with np.load(job/'artifacts/proxy_stats.npz') as stats:
            self.proxy_mean=stats['mean'].astype(np.float32)
            self.proxy_std=stats['std'].astype(np.float32)
        self.proxy_loaded={}

    def __getitem__(self,index):
        points,point_mask,_,_=super().__getitem__(index)
        row=self.rows[index]
        if row['cache'] not in self.proxy_loaded:
            with np.load(self.job/'proxy_targets'/row['cache']) as data:
                self.proxy_loaded[row['cache']]={key:data[key] for key in data.files}
        data=self.proxy_loaded[row['cache']]
        sl=slice(row['start'],row['end'])
        result=(points,point_mask,torch.from_numpy((data['C'][sl]-self.proxy_mean)/self.proxy_std),
                torch.from_numpy(data['C_mask'][sl]))
        if self.proxy_exp=='D_proxy':
            return (*result,torch.from_numpy(data['echo_weight'][sl]))
        return result


def smoke_proxy(job,exp,device):
    seed_all(42)
    encoder,head=RadarEncoder().to(device),RegressionHead(5).to(device)
    optimizer=torch.optim.AdamW([*encoder.parameters(),*head.parameters()],lr=.001,weight_decay=.0001)
    dataset=ProxyWindows(job,'train',exp)
    indices=np.random.default_rng(42).choice(len(dataset),32,replace=False)
    batch=[x.to(device) for x in torch.utils.data.default_collate([dataset[int(i)] for i in indices])]
    points,mask,target,target_mask=batch[:4]
    weight=batch[4] if len(batch)==5 else None
    losses=[]
    for step in range(250):
        optimizer.zero_grad(set_to_none=True)
        loss=masked_huber(head(encoder(points,mask)),target,target_mask,mask.any(-1),weight)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite proxy smoke loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([*encoder.parameters(),*head.parameters()],1)
        if not torch.isfinite(norm):
            raise ValueError('Nonfinite proxy smoke gradient')
        optimizer.step()
        losses.append(float(loss.detach()))
    result={'initial':float(np.mean(losses[:10])),'final':float(np.mean(losses[-10:])),
            'steps':250,'selected_indices':indices.tolist()}
    result['passed']=result['final']<.5*result['initial']
    write_json(job/f'{exp}_smoke.json',result)
    if not result['passed']:
        raise ValueError(f'{exp} smoke failed: {result}')
    print('SMOKE',exp,result,flush=True)


def run_proxy(job,device,seed):
    torch.set_num_threads(4)
    config=json.loads((job/'source/configs/mvp.yaml').read_text())
    settings={'seed':seed,'pretrain_epochs':30,'probe_epochs':50}
    verify_lock(job)
    for exp in ('C_proxy','D_proxy'):
        if seed==42:
            smoke_proxy(job,exp,device)
        pretrain(job,exp,device,settings,dataset_class=ProxyWindows)
        fit_probes(job,exp,device,config,settings)
    lock={'role':'supplementary proxy geometry and assumed Doppler analysis',
          'models':{exp:{name:digest(job/'runs'/exp/name) for name in
          ('best.pt','probe_classification_best.pt','probe_motion_best.pt')} for exp in ('C_proxy','D_proxy')}}
    reserve_test(job,lock)
    results={}
    for exp in ('C_proxy','D_proxy'):
        data=extract_features(job,exp,'test',device)
        heads={}
        for task,dimensions in [('classification',8),('motion',5)]:
            head=torch.nn.Linear(128,dimensions).to(device)
            state=torch.load(job/'runs'/exp/f'probe_{task}_best.pt',map_location='cpu',weights_only=True)
            assert state['encoder_sha']==digest(job/'runs'/exp/'best.pt')
            head.load_state_dict(state['head'])
            heads[task]=head.eval()
        results[exp]={'classification':classification_scores(heads['classification'],data,config['data']['probe_actions'],device),
                     'motion':attribute_scores(heads['motion'],data,device)}
    write_json(job/'proxy_results.json',{'seed':seed,'results':results,'test_lock':lock,
        'geometry_verified':False,'Doppler_author_verified':False,
        'role':lock['role'],'motion_units':'nominal m/s'})
    print('PROXY_DONE',seed,flush=True)
