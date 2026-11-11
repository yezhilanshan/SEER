"""Matched reset-length randomization ensemble and continuous-state control."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np


def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2),encoding='utf-8')


def generate(beginnings,cutoffs,frames,edges,count=10,seed=20261106):
    n=len(beginnings);true=np.flatnonzero(beginnings==np.arange(n))
    lengths=np.diff(np.r_[true,n]);ticks=np.r_[0,np.asarray(cutoffs)[1:]-2,frames]
    actual=np.diff(ticks[np.r_[true,n]])
    forbidden=np.isin(ticks[:-1],edges);forbidden[0]=False
    rng=np.random.default_rng(seed);accepted=[];seen=set();attempts=0
    while len(accepted)<count and attempts<2000000:
        permutations=rng.permuted(np.tile(lengths,(1024,1)),axis=1)
        starts=np.column_stack((np.zeros(1024,dtype=int),np.cumsum(permutations,axis=1)[:,:-1]))
        durations=np.diff(ticks[np.column_stack((starts,np.full(1024,n)))],axis=1)
        distance=np.max(np.abs(np.sort(durations,axis=1)-np.sort(actual)),axis=1)
        valid=~forbidden[starts].any(1)&(durations.min(1)==actual.min())&(distance<=2)
        for k in np.flatnonzero(valid):
            key=tuple(starts[k])
            if key in seen or np.array_equal(starts[k],true):continue
            seen.add(key);accepted.append(dict(starts=starts[k].tolist(),update_lengths=permutations[k].tolist(),
                frame_intervals=durations[k].tolist(),maximum_sorted_frame_difference=int(distance[k]),
                mean_sorted_frame_difference=float(np.abs(np.sort(durations[k])-np.sort(actual)).mean())))
            if len(accepted)==count:break
        attempts+=1024
    if len(accepted)!=count:raise RuntimeError(f'only {len(accepted)}/{count} feasible unique permutations after {attempts} draws')
    return dict(true_starts=true.tolist(),true_update_lengths=lengths.tolist(),true_frame_intervals=actual.tolist(),
                candidates=accepted,seed=seed,attempts=attempts)


def prepare(root,out):
    source=root/'stage1-event-prediction-v1';manifest=read(source/'cache-manifest.json')
    features={r['file']:r for r in read(root/'shared-front-event-warmup-v1/feature-manifest.json')}
    schedules=[]
    for i,r in enumerate(manifest):
        with np.load(Path(r['path'])/'indices.npz') as x:
            edges=sorted({v for t in features[r['file']]['instances'] for v in (t['start'],t['stop'])})
            row=generate(x['beginnings'],x['cutoffs'],r['frames'],edges,seed=20261106+i)
        schedules.append(dict(file=r['file'],**row))
    write(out/'schedules.json',schedules)
    return manifest,schedules


def run(root,out):
    import torch
    from continuous_mmwave.harrad.event_semantic import build_production
    from continuous_mmwave.harrad.training import seed_everything
    from continuous_mmwave.harrad.event_prediction import records_from,PredictionModel,batch,metrics
    from continuous_mmwave.harrad.preprocess import sha256
    source=root/'stage1-event-prediction-v1'
    manifest=read(source/'cache-manifest.json');schedules=read(out/'schedules.json')
    records=records_from(manifest)
    config=read(root/'shared-front-event-warmup-v1/config.json')
    weights=root/'shared-front-event-warmup-v1/training/best.pt'
    assert sha256(weights)==read(source/'source-lock.json')['event_sha256']
    for name,info in read(root/'shared-front-event-warmup-v1/input-lock.json')['qwen_files'].items():
        assert sha256(Path(config['qwen_path'])/name)==info['sha256']
    seed_everything(42,4);frozen=build_production(config)
    frozen.load_learned(torch.load(weights,map_location='cpu',weights_only=False)['learned'])
    frozen.requires_grad_(False).eval()
    arms=['gt','no_reset']+[f'random{i:02d}' for i in range(10)]
    cache=[]
    with torch.inference_mode():
        prompt=frozen.prompt(12)
        for rid,r in enumerate(records):
            folders=[out/'cache'/arm/Path(r['file']).stem for arm in arms]
            states=[]
            for folder in folders:
                folder.mkdir(parents=True)
                states.append(np.lib.format.open_memmap(folder/'events.npy',mode='w+',dtype='float32',shape=r['events'].shape))
            resets=[set(schedules[rid]['true_starts']),{0}]+[set(c['starts']) for c in schedules[rid]['candidates']]
            event=frozen.former.initial_state(12).clone();last_reset_frame=np.zeros(12,dtype=np.int64)
            for i in range(len(r['radar'])):
                reset=torch.tensor([i in s for s in resets],device='cuda')
                event=torch.where(reset[:,None,None],frozen.former.initial_state(12),event)
                last_reset_frame[np.array([i in s for s in resets])]=int(r['cutoffs'][i])-2
                age=int(r['cutoffs'][i])-last_reset_frame
                tokens=torch.as_tensor(np.array(r['radar'][i:i+1]),device='cuda').expand(12,-1,-1)
                maturity=torch.tensor(np.minimum(age/15/1.2,1),device='cuda',dtype=torch.float32)
                event=frozen.former.stream_update(event,tokens,maturity,None,prompt).event_tokens
                values=event.float().cpu().numpy()
                if not np.isfinite(values).all():raise RuntimeError('nonfinite event state')
                for j,state in enumerate(states):state[i]=values[j]
                if i%200==0:write(out/'status.json',dict(phase='event_cache',record=rid+1,total=len(records),update=i,total_updates=len(r['radar']),arms=12))
            for arm,folder,state in zip(arms,folders,states):
                state.flush();cache.append(dict(file=r['file'],arm=arm,path=str(folder/'events.npy'),sha256=sha256(folder/'events.npy')))
            write(out/'cache-manifest.json',cache)
            write(out/'status.json',dict(phase='event_cache',records=rid+1,total=len(records),arms=12))
    del frozen;torch.cuda.empty_cache()
    with np.load(source/'target-stats.npz') as x:mean,scale,variance=x['mean'],x['scale'],x['persistence_variance']
    fit=[i for i,r in enumerate(records) if r['role']=='fit'];maximum=max(r['maximum_history'] for r in records)
    jobs=[]
    for arm in arms:
        for r in records:r['events']=np.load(out/'cache'/arm/Path(r['file']).stem/'events.npy',mmap_mode='r')
        for seed in (42,43,44):
            seed_everything(seed,4);rng=np.random.default_rng(seed)
            schedule=[[(int(rid),int(rng.integers(records[rid]['samples']))) for rid in rng.choice(fit,16)] for _ in range(1000)]
            model=PredictionModel('event',maximum).cuda();optim=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
            folder=out/f'{arm}-seed{seed}';folder.mkdir()
            for step,samples in enumerate(schedule,1):
                values=batch(records,samples,mean,scale,'cuda');mu,lv=model(*values[:4])
                loss=(.5*(np.log(2*np.pi)+lv+(mu-values[4]).square()*(-lv).exp())).mean()
                optim.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(loss) or not torch.isfinite(norm):raise RuntimeError('nonfinite training')
                optim.step()
                if step==1 or step%100==0:write(out/'status.json',dict(phase='training',arm=arm,seed=seed,step=step,jobs_done=len(jobs),jobs_total=36))
            model.eval();torch.save(dict(model=model.state_dict(),steps=1000,seed=seed,arm=arm),folder/'final.pt')
            probe=next(i for i,r in enumerate(records) if r['role']=='validation')
            values=batch(records,[(probe,i) for i in range(16)],mean,scale,'cuda')
            with torch.inference_mode():before=tuple(x.cpu() for x in model(*values[:4]))
            replay=PredictionModel('event',maximum).cuda().eval()
            replay.load_state_dict(torch.load(folder/'final.pt',map_location='cpu',weights_only=False)['model'])
            with torch.inference_mode():after=replay(*values[:4])
            difference=max(float((a-b.cpu()).abs().max()) for a,b in zip(before,after));assert difference==0
            write(folder/'replay.json',dict(max_abs_difference=difference,sha256=sha256(folder/'final.pt')))
            del replay
            reports=[]
            for rid,r in enumerate(records):
                if r['role']!='validation':continue
                means=[];logvars=[]
                with torch.inference_mode():
                    for begin in range(0,r['samples'],16):
                        values=batch(records,[(rid,i) for i in range(begin,min(begin+16,r['samples']))],mean,scale,'cuda')
                        mu,lv=model(*values[:4]);means.append(mu.cpu().numpy());logvars.append(lv.cpu().numpy())
                mu,lv=np.concatenate(means),np.concatenate(logvars);target=(r['future']-mean)/scale
                persistence=np.repeat(((r['reference'][r['selected']]-mean)/scale)[:,None],3,1)
                reports.append(dict(file=r['file'],horizons=metrics(mu,lv,target,persistence,variance,r['groups'])))
                np.savez_compressed(folder/(Path(r['file']).stem+'.npz'),mean=mu,logvar=lv,target=target)
            job=dict(arm=arm,seed=seed,steps=1000,records=reports);jobs.append(job)
            write(folder/'complete.json',job);write(out/'jobs.json',jobs)
            del model;torch.cuda.empty_cache()
    write(out/'training-complete.json',dict(jobs=36,updates=36000,original_no_go_unchanged=True))
    write(out/'status.json',dict(phase='training_complete_pending_verification'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=('prepare','run'),required=True);a=p.parse_args()
    if a.phase=='prepare':a.output.mkdir(parents=True,exist_ok=False);prepare(a.root,a.output)
    else:run(a.root,a.output)
