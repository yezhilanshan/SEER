"""Three frozen RD backbones; validation-locked temporal/readout transfer study."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from continuous_mmwave.radar_text.align_readout import validate_subjects,validate_window_inputs
from continuous_mmwave.radar_text.data import caption_from_actions
from continuous_mmwave.radar_text.diagnostics import (period,confusion,classification_metrics,
    steady_mask,BED_CLASSES,smooth_predictions)
from continuous_mmwave.radar_text.p1 import (make_targets,decode_outputs,sequence_summary,
    probability_smooth,transition_prior,hmm_path,segment_paths)
from continuous_mmwave.radar_text.rd_transfer import make_backbone,normalize_rd,TemporalReadout
from continuous_mmwave.radar_text.runner import datasets,read_settings,read_checkpoint,seed_everything
from continuous_mmwave.radar_text.sliding_baselines import complete_window

ORIGINAL_SHA='03ae1027da2baf2210bfda363497de2913c9ff044423f2158e03a6b4041a6d56'
ARMS={'R0':'stage1-cnn','R1':'random-resnet','R2':'xfi-resnet'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def input_sha(sample):
    h=hashlib.sha256()
    for key in ('rd','md','timestamps'):
        x=sample[key].numpy();h.update(key.encode());h.update(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes())
    return h.hexdigest()


def rank(result,ordered=False):
    a,b=result['frame']['macro_f1'],result['window']['ordered']['metrics']['sequence_exact']
    return (b,a,result['window']['count_accuracy']) if ordered else (a,b,result['window']['count_accuracy'])


def loss_for(frame,out,labels,targets,ids,weights):
    fw,pw,cw=weights
    return (F.cross_entropy(frame.flatten(0,1),labels[ids].flatten(),weight=fw,ignore_index=-100)+
            F.binary_cross_entropy_with_logits(out[:,:14],targets['presence'][ids],pos_weight=pw)+
            F.cross_entropy(out[:,14:17],targets['count'][ids],weight=cw)+
            F.cross_entropy(out[:,17:].reshape(-1,15),targets['slots'][ids].reshape(-1)))


def padded(cache,mean,std,device):
    lengths=torch.tensor(np.diff(cache['offsets']),device=device);t=int(lengths.max());n=len(lengths)
    x=np.zeros((n,t,cache['X'].shape[1]),np.float32);y=np.full((n,t),-100,np.int64)
    for i,(a,b) in enumerate(zip(cache['offsets'][:-1],cache['offsets'][1:])):
        x[i,:b-a]=(cache['X'][a:b]-mean)/std;y[i,:b-a]=cache['labels'][a:b]
    return torch.from_numpy(x).to(device),lengths,torch.from_numpy(y).to(device)


def window_metrics(output,actions,mask):
    presence,counts,seqs=decode_outputs(output);targets=make_targets(actions)
    expected=targets['presence'].numpy().astype(bool);gt=targets['count'].numpy()+1
    tp=(presence[mask]&expected[mask]).sum(0);fp=(presence[mask]&~expected[mask]).sum(0);fn=(~presence[mask]&expected[mask]).sum(0)
    cm=np.zeros((3,3),int);np.add.at(cm,(gt[mask]-1,counts[mask]-1),1)
    return dict(presence_macro_f1=float((2*tp/np.maximum(1,2*tp+fp+fn)).mean()),
        count_accuracy=float(np.mean(counts[mask]==gt[mask])),count_mae=float(np.abs(counts[mask]-gt[mask]).mean()),
        count_confusion=cm.tolist(),ordered=sequence_summary([dict(reference_actions=a,predicted_actions=s)
            for a,s,keep in zip(actions,seqs,mask) if keep]))


def evaluate(model,cache,mean,std,windows,device):
    x,lengths,y=padded(cache,mean,std,device);frames=[];outputs=[];model.eval()
    with torch.no_grad():
        for ids in torch.arange(len(x),device=device).split(32):
            _,_,f,o=model(x[ids],lengths[ids]);outputs.append(o.cpu())
            frames.extend(f[j,:n].cpu().numpy() for j,n in enumerate(lengths[ids].tolist()))
    logits=np.concatenate(frames);output=torch.cat(outputs);pred=logits.argmax(-1)
    frame=classification_metrics(confusion(cache['labels'],pred))
    mask=np.array([w['full_window'] for w in windows]);actions=[w['actions'] for w in windows]
    return dict(frame=frame,window=window_metrics(output,actions,mask)),logits,output


def smoke(device):
    seed_everything(17);n,t=9,17;lengths=torch.tensor([17,12,9]*3,device=device)
    labels=torch.randint(0,14,(n,t),device=device);labels[torch.arange(t,device=device)[None]>=lengths[:,None]]=-100
    targets={k:v.to(device) for k,v in make_targets([[1],[2,3],[4,5,4]]*3).items()}
    for width in (128,512):
        model=TemporalReadout(width).to(device);optim=torch.optim.AdamW(model.parameters(),lr=.001)
        x=torch.randn(n,t,width,device=device);ids=torch.arange(n,device=device)
        for _ in range(2):
            optim.zero_grad();_,_,f,o=model(x,lengths)
            loss=loss_for(f,o,labels,targets,ids,(torch.ones(14,device=device),torch.ones(14,device=device),torch.ones(3,device=device)))
            assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optim.step()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        print(json.dumps(dict(smoke='passed',width=width,loss=float(loss.detach()),device=device)),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--device',default='cuda')
    for name in ('pilot','p1','xfi','output'):parser.add_argument('--'+name)
    args=parser.parse_args();torch.set_num_threads(4)
    if args.smoke:return smoke(args.device)
    assert all(getattr(args,k) for k in ('pilot','p1','xfi','output'))
    pilot,p1,xfi,root=map(Path,(args.pilot,args.p1,args.xfi,args.output));device=args.device
    root.mkdir(parents=True,exist_ok=False)
    for name in ('features','backbones','readouts'):(root/name).mkdir()
    started=datetime.now().astimezone().isoformat();source=Path(__file__).resolve().parents[1]
    def dump(name,value):
        path=root/name;tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)
    def status(state,**kwargs):
        payload=dict(state=state,started=started,**kwargs);dump('status.json',payload);print(json.dumps(payload),flush=True)
    try:
        status('preflight');sources=json.loads((source/'source-manifest.json').read_text())
        assert all(sha(source/k)==v for k,v in sources.items())
        previous=json.loads((p1/'provenance.json').read_text());inputs=dict(previous['original_input_sha256'])
        for name in ('provenance.json','windows-train.json','windows-validation.json','windows-test.json'):
            inputs[str(p1/name)]=sha(p1/name)
        inputs[str(xfi)]=sha(xfi)
        assert all(sha(k)==v for k,v in inputs.items())
        export=torch.load(xfi,map_location='cpu',weights_only=True);assert export['original_sha256']==ORIGINAL_SHA
        settings=read_settings(pilot/'align.yaml');sets,phrases=datasets(settings,('train','validation','test'))
        saved=read_checkpoint(pilot/'stage1/best.pt');validate_subjects(saved,{s:d.recordings.rows for s,d in sets.items()})
        rd_state={k[len('rd_cnn.'):]:v for k,v in saved['encoder'].items() if k.startswith('rd_cnn.')}
        backbones={name:make_backbone(kind,rd_state if name=='R0' else xfi if name=='R2' else None).to(device) for name,kind in ARMS.items()}
        initial={name:{k:v.detach().cpu().clone() for k,v in m.state_dict().items()} for name,m in backbones.items()}
        for name,state in initial.items():torch.save(state,root/f'backbones/{name}.pt')
        provenance=dict(original_input_sha256=inputs,source_sha256=sources,original_xfi_sha256=ORIGINAL_SHA,
            converted_xfi_sha256=sha(xfi),torch_version=str(torch.__version__),device=torch.cuda.get_device_name() if device=='cuda' else device,
            subjects={s:sorted({m['subject_id'] for m in d.recordings.rows}) for s,d in sets.items()},protocol=settings['data'],
            input='RD only; MD hashed for original paired-window identity, never passed to feature model',
            recipe='seed17;20epochs;batch32;AdamWlr.001/wd.01;clip1;dropout.1;train-only feature standardization;four equal losses',
            frozen='all frame backbone weights and BN buffers; Qwen not loaded',
            caveat='R0 RD-CNN inherited RD+MD joint training. R1 vs R2 same architecture/budget; R0 is trained reference only.',
            backbone_parameters={k:sum(p.numel() for p in m.parameters()) for k,m in backbones.items()})
        dump('provenance.json',provenance);caches={};windows={};cache_manifest={}
        def extract(split):
            status('extract_RD_only',split=split,windows=0);data=sets[split];xs={k:[] for k in ARMS};ws=[];labels=[];times=[]
            with torch.no_grad():
                for index in range(len(data)):
                    sample=data[index];rid,a,b,sid,actions=data.windows[index];meta=data.recordings.rows[rid]
                    raw=sample['rd'].unsqueeze(0).to(device);n=raw.shape[1]
                    rd=normalize_rd(raw,torch.tensor([n],device=device))[0]
                    for name,model in backbones.items():xs[name].append(torch.cat([model(v) for v in rd.split(64)]).cpu().numpy())
                    t=sample['timestamps'].numpy();dt=period(t);duration=float(t[-1]-t[0]+dt)
                    ws.append(dict(sample_id=sid,actions=actions,subject=meta['subject_id'],sensor=meta['sensor_id'],start=a,stop=b,
                        period=dt,duration=duration,full_window=complete_window(duration,dt),input_sha256=input_sha(sample)))
                    labels.append(sample['labels'].numpy());times.append(t)
                    if (index+1)%200==0:status('extract_RD_only',split=split,windows=index+1)
            validate_window_inputs(ws,json.loads((p1/f'windows-{split}.json').read_text()))
            windows[split]=ws;dump(f'windows-{split}.json',ws);offsets=np.r_[0,np.cumsum([len(t) for t in times])]
            common=dict(offsets=offsets,labels=np.concatenate(labels),timestamps=np.concatenate(times))
            caches[split]={}
            for name in ARMS:
                cache=dict(common,X=np.concatenate(xs[name]));path=root/f'features/{split}-{name}.npz';np.savez(path,**cache)
                caches[split][name]=cache;cache_manifest[f'{split}-{name}']=dict(sha256=sha(path),bytes=path.stat().st_size)
            dump('feature-cache-manifest.json',cache_manifest)
        extract('train');extract('validation');training={};models={}
        targets={k:v.to(device) for k,v in make_targets([w['actions'] for w in windows['train']]).items()}
        for name in ARMS:
            tr,va=caches['train'][name],caches['validation'][name]
            mean=tr['X'].mean(0);std=tr['X'].std(0).clip(1e-5);x,lengths,y=padded(tr,mean,std,device)
            seed_everything(17);model=TemporalReadout(x.shape[-1]).to(device);optim=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            freq=torch.bincount(y[y>=0],minlength=14).float();fw=freq.sum()/(14*freq.clamp_min(1))
            pos=targets['presence'].sum(0);pw=(len(x)-pos)/pos.clamp_min(1)
            freqc=torch.bincount(targets['count'],minlength=3).float();cw=len(x)/(3*freqc.clamp_min(1))
            history=[];best={};status('train_temporal_readout',arm=name)
            for epoch in range(1,21):
                model.train();losses=[]
                for ids in torch.randperm(len(x),device=device).split(32):
                    optim.zero_grad(set_to_none=True);_,_,frame,out=model(x[ids],lengths[ids]);loss=loss_for(frame,out,y,targets,ids,(fw,pw,cw))
                    assert torch.isfinite(loss);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);optim.step();losses.append(float(loss.detach()))
                result,_,_=evaluate(model,va,mean,std,windows['validation'],device)
                history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=result))
                for mode in ('frame','ordered'):
                    score=rank(result,mode=='ordered')
                    if mode not in best or score>best[mode]:
                        best[mode]=score;torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            mean=torch.from_numpy(mean),std=torch.from_numpy(std),width=x.shape[-1],epoch=epoch,selector=mode),root/f'readouts/{name}-{mode}.pt')
                status('train_temporal_readout',arm=name,epoch=epoch,frame_f1=result['frame']['macro_f1'],ordered_exact=result['window']['ordered']['metrics']['sequence_exact'])
            training[name]=dict(history=history,parameters=sum(p.numel() for p in model.parameters()),width=x.shape[-1],
                selected={mode:dict(epoch=torch.load(root/f'readouts/{name}-{mode}.pt',weights_only=True)['epoch'],sha256=sha(root/f'readouts/{name}-{mode}.pt')) for mode in ('frame','ordered')})
            dump('training.json',training);del x,y,model;torch.cuda.empty_cache() if device=='cuda' else None
        record_labels=[r.labels.numpy() for r in sets['train'].recordings]
        trans,start=transition_prior(record_labels);dump('transition-prior.json',dict(log_transition=trans.tolist(),log_start=start.tolist(),recordings=len(record_labels),source='full train recordings only; Laplace1'))
        def infer(split,name,mode='frame'):
            path=root/f'readouts/{name}-{mode}.pt';payload=torch.load(path,map_location='cpu',weights_only=True)
            model=TemporalReadout(payload['width']).to(device);model.load_state_dict(payload['state']);model.eval().requires_grad_(False)
            return evaluate(model,caches[split][name],payload['mean'].numpy(),payload['std'].numpy(),windows[split],device)
        def rows(split,actions):
            return [dict(sample_id=w['sample_id'],full_window=w['full_window'],subject=w['subject'],sensor=w['sensor'],reference_actions=w['actions'],predicted_actions=a,
                generated=caption_from_actions(a,phrases) if a else 'No activity was predicted.') for w,a in zip(windows[split],actions)]
        def full(rr):return [r for r in rr if r['full_window']]
        def seqrank(c):return tuple(c['validation']['metrics'][k] for k in ('sequence_exact','order_lcs','action_recall'))
        def paths(split,name,logits,cfg,counts=None):
            cache=caches[split][name];actions=[]
            for i,(a,b) in enumerate(zip(cache['offsets'][:-1],cache['offsets'][1:])):
                f=logits[a:b];prob=np.exp(f-f.max(-1,keepdims=True));prob/=prob.sum(-1,keepdims=True)
                times=cache['timestamps'][a:b];prob=probability_smooth(prob,times,cfg['seconds']);lp=np.log(prob.clip(1e-12))
                if counts is None:seq=smooth_predictions(hmm_path(lp,trans,start,cfg['weight']),times,minimum_seconds=cfg['minimum'])
                else:seq=segment_paths(lp,times,cfg['minimum'],cfg['weight']*trans,cfg['weight']*start)[int(counts[i])]
                actions.append(seq)
            return rows(split,actions)
        selection={};status('select_validation_postprocessing')
        for name in ARMS:
            result,logits,out=infer('validation',name);candidates=[]
            for seconds in (0.,.3,.5):
                for minimum in (.6,1.,2.):
                    for weight in (0.,.25,.5,1.):
                        cfg=dict(seconds=seconds,minimum=minimum,weight=weight)
                        candidates.append(dict(config=cfg,validation=sequence_summary(full(paths('validation',name,logits,cfg)))))
            chosen=max(candidates,key=seqrank);counts=decode_outputs(out)[1];count_candidates=[]
            for minimum in (.6,1.,2.):
                cfg=dict(chosen['config'],minimum=minimum)
                count_candidates.append(dict(config=cfg,validation=sequence_summary(full(paths('validation',name,logits,cfg,counts)))))
            selection[name]=dict(hmm=chosen['config'],count=max(count_candidates,key=seqrank)['config'],candidates=candidates,count_candidates=count_candidates)
            status('select_validation_postprocessing',arm=name,hmm_exact=chosen['validation']['metrics']['sequence_exact'])
        formal=max(ARMS,key=lambda name:rank(next(h['validation'] for h in training[name]['history'] if h['epoch']==training[name]['selected']['frame']['epoch']),True))
        dump('selection.json',dict(locked_at=datetime.now().astimezone().isoformat(),selection_split='validation strict12 only',arms=selection,formal_arm= formal,
            epochs={n:{m:v['epoch'] for m,v in e['selected'].items()} for n,e in training.items()},test_features_extracted=False))
        extract('test');results={}
        for split in ('validation','test'):
            predictions={};results[split]={};archive={}
            for name in ARMS:
                cache=caches[split][name];result,logits,out=infer(split,name);ordered_result,ordered_logits,ordered_out=infer(split,name,'ordered')
                presence,counts,seqs=decode_outputs(out);op,oc,os=decode_outputs(ordered_out)
                predictions[name]=dict(ordered=rows(split,seqs),ordered_selected=rows(split,os),counts=counts.tolist(),presence=presence.tolist(),ordered_selected_counts=oc.tolist(),
                    hmm=paths(split,name,logits,selection[name]['hmm']),predicted_count=paths(split,name,logits,selection[name]['count'],counts))
                steady=np.concatenate([steady_mask(cache['labels'][a:b],cache['timestamps'][a:b]) for a,b in zip(cache['offsets'][:-1],cache['offsets'][1:])])
                frame_steady=classification_metrics(confusion(cache['labels'],logits.argmax(-1),steady))
                results[split][name]=dict(frame=result['frame'],frame_steady=frame_steady,window=result['window'],ordered_selected=ordered_result,
                    sequences={method:{subset:sequence_summary(rr if subset=='original_protocol' else full(rr)) for subset in ('full12','original_protocol')}
                        for method,rr in predictions[name].items() if method in ('ordered','ordered_selected','hmm','predicted_count')},
                    bed_macro_f1=float(np.mean([result['frame']['per_class'][i]['f1'] for i in BED_CLASSES])))
                archive[name+'_frame']=logits;archive[name+'_window']=out.numpy();archive[name+'_ordered_frame']=ordered_logits;archive[name+'_ordered_window']=ordered_out.numpy()
            common=caches[split]['R0'];archive.update({k:common[k] for k in ('offsets','labels','timestamps')})
            np.savez(root/f'logits-{split}.npz',**archive);dump(f'predictions-{split}.json',predictions)
        dump('results.json',results)
        for name,model in backbones.items():
            assert all(torch.equal(v.detach().cpu(),initial[name][k]) for k,v in model.state_dict().items())
            assert all(p.grad is None and not p.requires_grad for p in model.parameters())
            assert all(not m.training for m in model.modules())
        assert all(sha(k)==v for k,v in inputs.items());assert all(sha(source/k)==v for k,v in sources.items())
        dump('verification.json',dict(original_inputs_unchanged=len(inputs),source_files_verified=len(sources),backbones_bitwise_unchanged=True,BN_eval=True,
            no_backbone_gradients=True,llm_loaded=False,selection_locked_before_test=True,paired_input_hashes=True,window_counts={s:len(w) for s,w in windows.items()},
            checkpoint_sha256={n:sha(root/f'backbones/{n}.pt') for n in ARMS}))
        status('complete',finished=datetime.now().astimezone().isoformat())
    except BaseException as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
