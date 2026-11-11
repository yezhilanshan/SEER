"""Predeclared scene diagnostics, record Bootstrap, and matched random resets."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np


SCENES = ('all','long_stable','fast_proxy','slow_proxy','repeated','boundary','transient_proxy')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')


def bootstrap_records(values):
    """Rows are complete recordings, columns are paired training seeds."""
    values=np.asarray(values,dtype=float)
    if values.ndim!=2 or not len(values):
        raise ValueError('nonempty record by seed matrix required')
    count=len(values)
    if count<2:
        return dict(mean=float(values.mean()),records=count,ci95=None,leave_one_out=None)
    # All n**n equally likely samples of the ordinary n-cluster Bootstrap.
    indexes=np.array(list(itertools.product(range(count),repeat=count)))
    draws=values.mean(1)[indexes].mean(1)
    return dict(mean=float(values.mean()),records=count,
        ci95=np.quantile(draws,[.025,.975],method='inverted_cdf').tolist(),
        leave_one_out=[float(np.delete(values,i,axis=0).mean()) for i in range(count)])


def shuffled_starts(beginnings, seed):
    beginnings=np.asarray(beginnings,dtype=np.int64)
    starts=np.flatnonzero(beginnings==np.arange(len(beginnings)))
    if not len(starts) or starts[0]!=0:
        raise ValueError('record initialization missing')
    lengths=np.diff(np.r_[starts,len(beginnings)])
    permuted=np.random.default_rng(seed).permutation(lengths)
    return np.r_[0,np.cumsum(permuted)[:-1]],lengths,permuted


def smooth(x):
    sums=np.vstack((np.zeros((1,x.shape[1])),np.cumsum(x,axis=0,dtype=float)))
    stop=np.arange(1,len(x)+1); start=np.maximum(0,stop-5)
    return (sums[stop]-sums[start])/(stop-start)[:,None]


def boundary_profiles(x, instances):
    smoothed=smooth(x)
    edges=sorted({p for r in instances for p in (r['start'],r['stop']) if 15<=p<=len(x)-15})
    result=[]
    for b in edges:
        before=x[b-15:b-3].mean(0); after=x[b+3:b+15].mean(0)
        direction=after-before; size=float(np.linalg.norm(direction)/np.sqrt(x.shape[1]))
        progress=(smoothed[b-15:b+15]-before)@direction/max(float(direction@direction),1e-12)
        low=np.flatnonzero(progress>=.2)
        if not len(low): continue
        first=int(low[0]); high=np.flatnonzero(progress[first:]>=.8)
        if not len(high): continue
        last=first+int(high[0])
        monotone=float(np.mean(np.diff(progress[first:last+1])>=0)) if last>first else 1.
        result.append(dict(boundary=b,amplitude=size,transition_frames=last-first,monotone=monotone))
    return result


def scene_masks(x, instances, cutoffs, threshold):
    count=len(cutoffs)
    masks=np.zeros((count,3,len(SCENES)),bool); masks[:,:,0]=True
    edges=np.array(sorted({p for r in instances for p in (r['start'],r['stop']) if 0<p<len(x)}))
    profiles=boundary_profiles(x,instances)
    eligible=[r for r in profiles if r['amplitude']>=threshold['contrast'] and r['monotone']>=.65]
    fast=[r['boundary'] for r in eligible if r['transition_frames']<=3]
    slow=[r['boundary'] for r in eligible if r['transition_frames']>=6]
    repeats=[dict(start=r['start'],gap_frames=r['start']-instances[i-1]['stop'],label=r['label'])
             for i,r in enumerate(instances) if i and instances[i-1]['label']==r['label']]
    repeat=[r['start'] for r in repeats]
    delta=np.r_[0,np.linalg.norm(np.diff(x,axis=0),axis=1)/np.sqrt(x.shape[1])]
    drift=np.r_[0,np.linalg.norm(np.diff(smooth(x),axis=0),axis=1)/np.sqrt(x.shape[1])]
    transient=[]
    for i in np.flatnonzero(delta>threshold['jump']):
        if i<3 or i+4>len(x) or (len(edges) and np.min(np.abs(edges-i))<=8): continue
        returned=np.linalg.norm(x[i+1:i+4].mean(0)-x[i-3:i].mean(0))/np.sqrt(x.shape[1])
        if returned<=threshold['return']: transient.append(int(i))
    for j,c in enumerate(cutoffs):
        near=lambda points,width:any(abs(int(c)-p)<=width for p in points)
        masks[j,:,2]=near(fast,8); masks[j,:,3]=near(slow,8)
        masks[j,:,4]=near(repeat,15); masks[j,:,5]=near(edges,8); masks[j,:,6]=near(transient,3)
        for h,offset in enumerate((3,9,15)):
            masks[j,h,1]=any(r['stop']-r['start']>=45 and r['start']+15<=c-1 and c+offset<=r['stop']-15
                for r in instances) and drift[c-1]<=threshold['drift']
    return masks,dict(profiles=profiles,fast=fast,slow=slow,repeated_starts=repeat,
                      repeated_instances=repeats,transients=transient,internal_edges=edges.tolist())


def diagnostics(root, output):
    source=root/'stage1-event-prediction-v1'
    manifest=read(source/'cache-manifest.json')
    feature={r['file']:r for r in read(root/'shared-front-event-warmup-v1/feature-manifest.json')}
    def raw(row): return np.load(Path(feature[row['file']]['artifact_dir'])/'frame_features.npy',mmap_mode='r')
    fit=[r for r in manifest if r['role']=='fit']
    values=np.concatenate([raw(r) for r in fit])
    mean=values.mean(0,dtype=float); scale=np.maximum(values.std(0,dtype=float),1e-4)
    del values
    changes=[];drifts=[];contrasts=[]
    for row in fit:
        x=(raw(row)-mean)/scale
        changes.extend(np.linalg.norm(np.diff(x,axis=0),axis=1)/8)
        drifts.extend(np.linalg.norm(np.diff(smooth(x),axis=0),axis=1)/8)
        contrasts.extend(p['amplitude'] for p in boundary_profiles(x,feature[row['file']]['instances']))
    threshold=dict(jump=float(np.quantile(changes,.99)),return_=float(np.quantile(changes,.25)),
                   drift=float(np.median(drifts)),contrast=float(np.median(contrasts)))
    threshold['return']=threshold.pop('return_')
    write(output/'scene-thresholds.json',dict(thresholds=threshold,fit_files=[r['file'] for r in fit],
        semantics='frozen-feature proxies; not independently annotated slow/noise ground truth'))
    rows=[]; all_masks={}
    for row in manifest:
        if row['role']!='validation': continue
        with np.load(Path(row['path'])/'indices.npz') as indexes:
            cutoffs=indexes['cutoffs'][indexes['selected']]
        masks,meta=scene_masks((raw(row)-mean)/scale,feature[row['file']]['instances'],cutoffs,threshold)
        all_masks[row['file']]=masks
        np.savez_compressed(output/(Path(row['file']).stem+'-scenes.npz'),masks=masks,cutoffs=cutoffs)
        write(output/(Path(row['file']).stem+'-scene-candidates.json'),meta)
        for seed in (42,43,44):
            for kind in ('gru','event'):
                path=source/f'{kind}-seed{seed}'/(Path(row['file']).stem+'.npz')
                with np.load(path) as data:
                    assert np.array_equal(cutoffs,data['observed_cutoff'])
                    error=(data['mean'].astype(float)-data['target'])**2
                    variance=np.exp(data['logvar'].astype(float))
                    logterm=.5*data['logvar']
                    weighted=.5*error/variance
                    nll=.5*np.log(2*np.pi)+logterm+weighted
                    for g,name in enumerate(SCENES):
                        for h in range(3):
                            mask=masks[:,h,g]
                            if not mask.any(): continue
                            e=error[mask,h].mean(1);v=variance[mask,h].mean(1)
                            correlation=float(np.corrcoef(e,v)[0,1]) if e.std()>0 and v.std()>0 else None
                            rows.append(dict(file=row['file'],seed=seed,kind=kind,scene=name,horizon=(.2,.6,1.)[h],n=int(mask.sum()),
                                mse=float(error[mask,h].mean()),nll=float(nll[mask,h].mean()),variance=float(variance[mask,h].mean()),
                                nll_logvar_term=float(logterm[mask,h].mean()),nll_error_term=float(weighted[mask,h].mean()),
                                coverage90=float((error[mask,h]<=2.705543454*variance[mask,h]).mean()),
                                standardized_error=float((error[mask,h]/variance[mask,h]).mean()),
                                error_variance_correlation=correlation))
    write(output/'record-seed-scene-metrics.json',rows)
    summary=[]
    for scene in SCENES:
        for horizon in (.2,.6,1.):
            chosen=[r for r in rows if r['scene']==scene and r['horizon']==horizon]
            files=sorted({r['file'] for r in chosen})
            if not files:
                summary.append(dict(scene=scene,horizon=horizon,records=0,reason='no support'));continue
            lookup={(r['file'],r['seed'],r['kind']):r for r in chosen}
            item=dict(scene=scene,horizon=horizon,files=files,
                sample_support=sum(lookup[(f,42,'event')]['n'] for f in files),by_seed={})
            for key in ('mse','nll'):
                delta=np.array([[lookup[(f,s,'event')][key]-lookup[(f,s,'gru')][key] for s in (42,43,44)] for f in files])
                item[key]=dict(event=float(np.mean([r[key] for r in chosen if r['kind']=='event'])),
                              gru=float(np.mean([r[key] for r in chosen if r['kind']=='gru'])),paired=bootstrap_records(delta))
                for i,seed in enumerate((42,43,44)):
                    item['by_seed'].setdefault(str(seed),{})[key]=bootstrap_records(delta[:,i:i+1])
            summary.append(item)
    write(output/'scene-bootstrap-summary.json',summary)
    return manifest


def summarize_controls(source, output, results):
    summary=[]
    originals={s:read(source/f'event-seed{s}/complete.json') for s in (42,43,44)}
    for condition in ('swap','retrained'):
        control={r['seed']:r['records'] for r in results if r['condition']==condition}
        if len(control)!=3: continue
        files=sorted(r['file'] for r in control[42])
        for h in range(3):
            item=dict(condition=condition,horizon=(.2,.6,1.)[h],files=files)
            for key in ('mse','nll'):
                delta=[];correct=[];random=[]
                for f in files:
                    a=[next(r for r in originals[s]['records'] if r['file']==f)['horizons'][h][key] for s in (42,43,44)]
                    b=[next(r for r in control[s] if r['file']==f)['horizons'][h][key] for s in (42,43,44)]
                    correct.append(a);random.append(b);delta.append(np.array(a)-b)
                item[key]=dict(correct=float(np.mean(correct)),random=float(np.mean(random)),
                    correct_minus_random=bootstrap_records(delta),
                    by_seed={str(s):bootstrap_records(np.array(delta)[:,i:i+1]) for i,s in enumerate((42,43,44))})
            summary.append(item)
    write(output/'random-control-bootstrap-summary.json',summary)


def random_control(root, output, manifest):
    import torch
    from continuous_mmwave.harrad.event_semantic import build_production
    from continuous_mmwave.harrad.training import seed_everything
    from continuous_mmwave.harrad.event_prediction import records_from,PredictionModel,batch,metrics
    from continuous_mmwave.harrad.preprocess import sha256
    source=root/'stage1-event-prediction-v1'
    records=records_from(manifest)
    config=read(root/'shared-front-event-warmup-v1/config.json')
    weights=root/'shared-front-event-warmup-v1/training/best.pt'
    assert sha256(weights)==read(source/'source-lock.json')['event_sha256']
    seed_everything(42,4)
    frozen=build_production(config)
    frozen.load_learned(torch.load(weights,map_location='cpu',weights_only=False)['learned'])
    frozen.requires_grad_(False).eval()
    schedules=[]
    with torch.inference_mode():
        for rid,r in enumerate(records):
            starts,old_lengths,new_lengths=shuffled_starts(r['beginnings'],20261105+rid)
            folder=output/'random-cache'/Path(r['file']).stem;folder.mkdir(parents=True)
            states=np.lib.format.open_memmap(folder/'events.npy',mode='w+',dtype='float32',shape=r['events'].shape)
            reset=set(starts.tolist()); event=None;age=0;prompt=frozen.prompt(1)
            for i in range(len(r['radar'])):
                if i in reset: event=frozen.former.initial_state(1);age=0
                age+=2
                tokens=torch.as_tensor(np.array(r['radar'][i:i+1]),device='cuda')
                maturity=torch.tensor([min(age/15/1.2,1.)],device='cuda')
                event=frozen.former.stream_update(event,tokens,maturity,None,prompt).event_tokens
                states[i]=event[0].float().cpu().numpy()
            states.flush()
            true_starts=np.flatnonzero(r['beginnings']==np.arange(len(r['beginnings'])))
            overlap=len(set(starts[1:])&set(true_starts[1:]))
            schedules.append(dict(file=r['file'],random_starts=starts.tolist(),true_starts=true_starts.tolist(),
                internal_reset_overlap=overlap,internal_reset_count=len(starts)-1,
                lengths_preserved=sorted(old_lengths.tolist())==sorted(new_lengths.tolist()),
                effectively_changed=not np.array_equal(old_lengths,new_lengths),sha256=sha256(folder/'events.npy')))
            r['events']=np.load(folder/'events.npy',mmap_mode='r')
            write(output/'random-reset-schedules.json',schedules)
            write(output/'status.json',dict(phase='random_event_cache',records=rid+1,total=len(records)))
    del frozen;torch.cuda.empty_cache()
    with np.load(source/'target-stats.npz') as data:
        mean,scale,variance=data['mean'],data['scale'],data['persistence_variance']
    fit=[i for i,r in enumerate(records) if r['role']=='fit']
    maximum=max(r['maximum_history'] for r in records)
    results=[]
    for seed in (42,43,44):
        rng=np.random.default_rng(seed)
        schedule=[[(int(rid),int(rng.integers(records[rid]['samples']))) for rid in rng.choice(fit,16)] for _ in range(1000)]
        for condition in ('swap','retrained'):
            seed_everything(seed,4);model=PredictionModel('event',maximum).cuda()
            folder=output/f'{condition}-seed{seed}';folder.mkdir()
            if condition=='swap':
                model.load_state_dict(torch.load(source/f'event-seed{seed}/final.pt',map_location='cpu',weights_only=False)['model'])
            else:
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=.01)
                for step,samples in enumerate(schedule,1):
                    values=batch(records,samples,mean,scale,'cuda');mu,lv=model(*values[:4]);target=values[4]
                    loss=(.5*(np.log(2*np.pi)+lv+(mu-target).square()*(-lv).exp())).mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward()
                    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    if not torch.isfinite(loss) or not torch.isfinite(norm): raise RuntimeError('nonfinite random-control update')
                    optimizer.step()
                    if step==1 or step%20==0:write(output/'status.json',dict(phase='random_predictor_training',seed=seed,step=step))
                torch.save(dict(model=model.state_dict(),steps=1000,seed=seed),folder/'final.pt')
            model.eval()
            probe=next((i for i,r in enumerate(records) if r['role']=='validation'))
            values=batch(records,[(probe,0)],mean,scale,'cuda')
            with torch.inference_mode():before=model(*values[:4])
            replay=PredictionModel('event',maximum).cuda().eval()
            checkpoint=folder/'final.pt' if condition=='retrained' else source/f'event-seed{seed}/final.pt'
            replay.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
            with torch.inference_mode():after=replay(*values[:4])
            difference=max(float((a-b).abs().max()) for a,b in zip(before,after))
            if difference!=0:raise RuntimeError('checkpoint replay differs')
            write(folder/'checkpoint-replay.json',dict(max_abs_difference=difference,checkpoint_sha256=sha256(checkpoint)))
            del replay
            rows=[]
            for rid,r in enumerate(records):
                if r['role']!='validation':continue
                means=[];logvars=[]
                with torch.inference_mode():
                    for begin in range(0,r['samples'],16):
                        ids=range(begin,min(begin+16,r['samples']))
                        values=batch(records,[(rid,i) for i in ids],mean,scale,'cuda')
                        mu,lv=model(*values[:4]);means.append(mu.cpu().numpy());logvars.append(lv.cpu().numpy())
                mu,lv=np.concatenate(means),np.concatenate(logvars)
                target=(r['future']-mean)/scale
                persistence=np.repeat(((r['reference'][r['selected']]-mean)/scale)[:,None],3,1)
                rows.append(dict(file=r['file'],horizons=metrics(mu,lv,target,persistence,variance,r['groups'])))
                np.savez_compressed(folder/(Path(r['file']).stem+'.npz'),mean=mu,logvar=lv,target=target)
            results.append(dict(condition=condition,seed=seed,records=rows))
            write(folder/'complete.json',results[-1]);write(output/'random-control-results.json',results)
            summarize_controls(source,output,results)
            del model;torch.cuda.empty_cache()
    write(output/'complete.json',dict(phase='complete',diagnostic_only=True,original_stage1_no_go_unchanged=True,
        conditions=['correct-existing','swap-random-state','retrained-random-state'],training_seeds=[42,43,44],
        reset_randomization_replicates=1,random_control=results))
    write(output/'status.json',dict(phase='complete'))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('all','diagnostics','random'),default='all')
    args=parser.parse_args()
    if args.phase!='random':
        args.output.mkdir(parents=True,exist_ok=False)
        manifest=diagnostics(args.root,args.output)
    else:
        if not (args.output/'scene-bootstrap-summary.json').exists():raise RuntimeError('A/B phase required')
        manifest=read(args.root/'stage1-event-prediction-v1/cache-manifest.json')
    if args.phase!='diagnostics':random_control(args.root,args.output,manifest)
