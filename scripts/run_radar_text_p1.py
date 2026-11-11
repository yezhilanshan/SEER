"""Frozen P1-A/B/C diagnostics; only bounded evidence/count readouts are trained."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from continuous_mmwave.radar_text.align_readout import validate_subjects,validate_window_inputs
from continuous_mmwave.radar_text.config import RadarTextConfig
from continuous_mmwave.radar_text.data import collate_windows,caption_from_actions
from continuous_mmwave.radar_text.diagnostics import period,smooth_predictions
from continuous_mmwave.radar_text.model import RadarEncoder
from continuous_mmwave.radar_text.p1 import (edit_alignment,sequence_summary,make_targets,decode_outputs,EvidenceProbe,
    probability_smooth,transition_prior,hmm_path,segment_paths)
from continuous_mmwave.radar_text.runner import datasets,read_checkpoint,read_settings,seed_everything
from continuous_mmwave.radar_text.sliding_baselines import complete_window,apply_saved_head


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def input_sha(sample):
    h=hashlib.sha256()
    for key in ('rd','md','timestamps'):
        x=sample[key].numpy();h.update(key.encode());h.update(str((x.shape,str(x.dtype))).encode());h.update(x.tobytes())
    return h.hexdigest()


def probe_metrics(output,targets,actions,mask):
    presence,count,seqs=decode_outputs(output)
    expected=targets['presence'].cpu().numpy().astype(bool);gt=targets['count'].cpu().numpy()+1
    tp=(presence[mask]&expected[mask]).sum(0);fp=(presence[mask]&~expected[mask]).sum(0);fn=(~presence[mask]&expected[mask]).sum(0)
    f1=2*tp/np.maximum(1,2*tp+fp+fn)
    confusion=np.zeros((3,3),int);np.add.at(confusion,(gt[mask]-1,count[mask]-1),1)
    rows=[dict(reference_actions=a,predicted_actions=s) for a,s,k in zip(actions,seqs,mask) if k]
    return dict(presence_macro_f1=float(f1.mean()),presence_per_class_f1=f1.tolist(),
                count_accuracy=float(np.mean(count[mask]==gt[mask])),count_mae=float(np.abs(count[mask]-gt[mask]).mean()),
                count_confusion=confusion.tolist(),ordered=sequence_summary(rows))


def main():
    parser=argparse.ArgumentParser()
    for k in ('pilot','p0','output'):parser.add_argument('--'+k,required=True)
    args=parser.parse_args();pilot,p0,root=map(Path,(args.pilot,args.p0,args.output))
    root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir();(root/'probes').mkdir()
    torch.set_num_threads(4);seed_everything(17)
    source=Path(__file__).resolve().parents[1];sources=json.loads((source/'source-manifest.json').read_text())
    assert all(sha(source/k)==v for k,v in sources.items())
    started=datetime.now().astimezone().isoformat();completed=[]
    def dump(name,value):
        p=root/name;t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(p)
    def status(state,**extra):
        dump('status.json',dict(state=state,started=started,completed=completed,**extra));print(json.dumps(dict(state=state,**extra)),flush=True)
    try:
        status('preflight')
        old_prov=json.loads((p0/'provenance.json').read_text());inputs=dict(old_prov['original_input_sha256'])
        for name in ('provenance.json','predictions-validation.json','predictions-test.json','windows-validation.json',
                     'windows-test.json','logits-validation.npz','logits-test.npz','heads/align-H1.pt','head-training.json'):
            inputs[str(p0/name)]=sha(p0/name)
        assert sha(pilot/'align/best.pt')==old_prov['encoder_sha256']
        assert all(sha(k)==v for k,v in inputs.items())
        old={s:json.loads((p0/f'predictions-{s}.json').read_text()) for s in ('validation','test')}
        a_result={};review={}
        for split in old:
            a_result[split]={}
            for subset in ('full12','original_protocol'):
                keep=lambda rows:[r for r in rows if subset=='original_protocol' or r['full_window']]
                first,second=keep(old[split]['B1']['individual']),keep(old[split]['B3'])
                assert [r['sample_id'] for r in first]==[r['sample_id'] for r in second]
                a_result[split][subset]={'B1':sequence_summary(first),'B3':sequence_summary(second)}
                if split=='test' and subset=='full12':
                    for group,flag in (('B1_only',(True,False)),('B3_only',(False,True))):
                        pairs=[]
                        for a,b in zip(first,second):
                            if (bool(a['sequence_exact']),bool(b['sequence_exact']))==flag:
                                pairs.append(dict(sample_id=a['sample_id'],subject=a['subject'],sensor=a['sensor'],
                                     reference_actions=a['reference_actions'],B1=a['predicted_actions'],B3=b['predicted_actions'],
                                     B1_text=a['generated'],B3_text=b['generated'],B1_edit=edit_alignment(a['reference_actions'],a['predicted_actions']),
                                     B3_edit=edit_alignment(a['reference_actions'],b['predicted_actions'])))
                        review[group]=pairs
            dump('p1-a.json',a_result)
        assert len(review['B1_only'])==29 and len(review['B3_only'])==57
        dump('paired-error-review.json',review);completed.append('P1-A')
        settings=read_settings(pilot/'align.yaml');sets,phrases=datasets(settings,('train','validation','test'))
        saved=read_checkpoint(pilot/'align/best.pt');validate_subjects(saved,{s:d.recordings.rows for s,d in sets.items()})
        config=RadarTextConfig(**saved['config']);assert config.tokens==8
        encoder=RadarEncoder(config).cuda().eval().requires_grad_(False);encoder.load_state_dict(saved['encoder'])
        width=saved['projector']['2.weight'].shape[0];assert width==896
        projector=nn.Sequential(nn.Linear(256,width),nn.GELU(),nn.Linear(width,width)).cuda().eval().requires_grad_(False)
        projector.load_state_dict(saved['projector']);initial={n:{k:v.detach().clone() for k,v in m.state_dict().items()} for n,m in [('encoder',encoder),('projector',projector)]}
        head=torch.load(p0/'heads/align-H1.pt',map_location='cpu',weights_only=True)
        for data in sets.values():
            for meta in data.recordings.rows:
                p=Path(meta['path']);p=p if p.is_absolute() else data.recordings.manifest.parent/p
                if str(p) not in inputs:inputs[str(p)]=sha(p)
        dump('provenance.json',dict(original_input_sha256=inputs,source_sha256=sources,align_sha256=sha(pilot/'align/best.pt'),
              frozen=['Align Encoder','Projector','Qwen not loaded'],subjects={s:sorted({m['subject_id'] for m in d.recordings.rows}) for s,d in sets.items()},
              protocol=settings['data'],probe='rank32 linear vs rank32 GELU, train-only coordinate statistics, shared tasks/budget/selection',
              loss='presence BCE posweight train + count CE inversefrequency train + 3slot CE including PAD14',
              selection='val full12 ordered exact, presenceMacroF1,count accuracy,ties earliest',
              count_head='independent 8-bin H -> MLP32 -> 3 count logits; val accuracy then MAE',
              oracle='GTcount diagnostic only, same locked decoder; not theoretical upper bound'))
        caches={};cache_manifest={};record_labels=[]
        def extract(split):
            status('extract_frozen_H_Z_P',split=split,windows=0)
            data=sets[split];hs=[];zs=[];ps=[];windows=[];times=[];logits=[]
            with torch.no_grad():
                for raw in DataLoader(data,batch_size=2,collate_fn=collate_windows,shuffle=False,num_workers=0):
                    lengths=raw['lengths'].cuda();h,z=encoder(raw['rd'].cuda(),raw['md'].cuda(),lengths);p=projector(z)
                    values=apply_saved_head(h,head).cpu().numpy()
                    for i in range(len(lengths)):
                        sample=data[len(windows)];rid,start,stop,sid,actions=data.windows[len(windows)];meta=data.recordings.rows[rid]
                        assert sid==raw['sample_ids'][i]
                        assert torch.equal(sample['rd'],raw['rd'][i,:lengths[i]]) and torch.equal(sample['md'],raw['md'][i,:lengths[i]])
                        t=sample['timestamps'].numpy();dt=period(t);duration=float(t[-1]-t[0]+dt)
                        windows.append(dict(sample_id=sid,actions=actions,subject=meta['subject_id'],sensor=meta['sensor_id'],
                            start=start,stop=stop,period=dt,duration=duration,full_window=complete_window(duration,dt),input_sha256=input_sha(sample)))
                        hs.append(h[i,:len(t)].cpu().numpy());zs.append(z[i].cpu().numpy());ps.append(p[i].cpu().numpy());times.append(t);logits.append(values[i,:len(t)])
                    if len(windows)%200==0:status('extract_frozen_H_Z_P',split=split,windows=len(windows))
            if split!='train':
                expected=json.loads((p0/f'windows-{split}.json').read_text());validate_window_inputs(windows,expected)
                with np.load(p0/f'logits-{split}.npz') as ref:
                    assert np.array_equal(np.concatenate(times),ref['timestamps'])
                    np.testing.assert_allclose(np.concatenate(logits),ref['B1'],atol=5e-6,rtol=5e-6)
            offsets=np.r_[0,np.cumsum([len(v) for v in hs])]
            path=root/f'features/{split}.npz'
            np.savez(path,H=np.concatenate(hs),Z=np.stack(zs),P=np.stack(ps),offsets=offsets,timestamps=np.concatenate(times),logits=np.concatenate(logits))
            cache_manifest[split]=dict(path=f'features/{split}.npz',sha256=sha(path),bytes=path.stat().st_size)
            dump(f'windows-{split}.json',windows);dump('feature-cache-manifest.json',cache_manifest)
            caches[split]=dict(H=hs,Z=np.stack(zs),P=np.stack(ps),windows=windows,times=times,logits=logits)
            if split=='train':
                for sample in data.recordings:record_labels.append(sample.labels.numpy())
        extract('train');extract('validation')
        train=caches['train'];val=caches['validation'];max_length=max(map(len,train['H']))
        def features(cache,kind):
            if kind=='H':
                x=np.zeros((len(cache['H']),max_length,256),np.float32)
                for i,h in enumerate(cache['H']):
                    assert len(h)<=max_length;x[i,:len(h)]=h
                return x.reshape(len(x),-1)
            return cache[kind].reshape(len(cache[kind]),-1)
        def count_features(cache):
            return np.stack([F.adaptive_avg_pool1d(torch.from_numpy(h).T,8).T.numpy().reshape(-1) for h in cache['H']])
        targets={s:make_targets([w['actions'] for w in c['windows']]) for s,c in caches.items()}
        vm=np.array([w['full_window'] for w in val['windows']]);models={};probe_report={}
        def fit(name,tr,va,count_only=False,nonlinear=False):
            status('fit_light_probe',name=name)
            mean=tr.mean(0);std=tr.std(0).clip(1e-5)
            x=torch.from_numpy((tr-mean)/std).cuda();v=torch.from_numpy((va-mean)/std).cuda()
            seed_everything(17);model=EvidenceProbe(tr.shape[1],nonlinear,outputs=3 if count_only else 62).cuda()
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
            y={k:t.cuda() for k,t in targets['train'].items()};vy=targets['validation'];actions=[w['actions'] for w in val['windows']]
            freq=torch.bincount(y['count'],minlength=3).float();cw=len(x)/(3*freq.clamp_min(1))
            pos=y['presence'].sum(0);pw=(len(x)-pos)/pos.clamp_min(1)
            history=[];best=None;path=root/'probes'/f'{name}.pt'
            for epoch in range(1,21):
                model.train();losses=[]
                for ids in torch.randperm(len(x),device='cuda').split(64):
                    optimizer.zero_grad(set_to_none=True);out=model(x[ids])
                    loss=F.cross_entropy(out,y['count'][ids],weight=cw) if count_only else (
                        F.binary_cross_entropy_with_logits(out[:,:14],y['presence'][ids],pos_weight=pw)+
                        F.cross_entropy(out[:,14:17],y['count'][ids],weight=cw)+
                        F.cross_entropy(out[:,17:].reshape(-1,15),y['slots'][ids].reshape(-1)))
                    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach()))
                model.eval()
                with torch.no_grad():out=model(v).cpu()
                if count_only:
                    pred=out.argmax(-1).numpy();gt=vy['count'].numpy()
                    result=dict(count_accuracy=float((pred[vm]==gt[vm]).mean()),count_mae=float(np.abs(pred[vm]-gt[vm]).mean()))
                    rank=(result['count_accuracy'],-result['count_mae'])
                else:
                    result=probe_metrics(out,vy,actions,vm)
                    rank=(result['ordered']['metrics']['sequence_exact'],result['presence_macro_f1'],result['count_accuracy'])
                history.append(dict(epoch=epoch,train_loss=float(np.mean(losses)),validation=result))
                if best is None or rank>best:
                    best=rank;torch.save(dict(state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},mean=torch.from_numpy(mean),std=torch.from_numpy(std),epoch=epoch,
                           width=tr.shape[1],nonlinear=nonlinear,count_only=count_only),path)
            payload=torch.load(path,map_location='cpu',weights_only=True);model.load_state_dict(payload['state']);model.eval().requires_grad_(False)
            info=dict(selected_epoch=payload['epoch'],history=history,parameters=sum(p.numel() for p in model.parameters()),
                      input_dimensions=tr.shape[1],checkpoint_sha256=sha(path),selection='validation only; ties earliest')
            models[name]=(model,payload);probe_report[name]=info;dump('probe-training.json',probe_report)
        for kind in ('H','Z','P'):
            tr,va=features(train,kind),features(val,kind)
            for nonlinear in (False,True):fit(f'{kind}-'+('nonlinear' if nonlinear else 'linear'),tr,va,nonlinear=nonlinear)
        fit('independent-count',count_features(train),count_features(val),count_only=True,nonlinear=True)
        trans,start=transition_prior(record_labels);dump('transition-prior.json',dict(log_transition=trans.tolist(),log_start=start.tolist(),source='full train recordings only; frame transitions incl self, Laplace1',recordings=len(record_labels)))
        del record_labels
        def outputs(cache,name):
            model,payload=models[name];raw=count_features(cache) if name=='independent-count' else features(cache,name.split('-')[0])
            x=torch.from_numpy((raw-payload['mean'].numpy())/payload['std'].numpy()).cuda()
            with torch.no_grad():return model(x).cpu()
        def rows_from_actions(windows,actions):
            return [dict(sample_id=w['sample_id'],full_window=w['full_window'],subject=w['subject'],sensor=w['sensor'],reference_actions=w['actions'],predicted_actions=a,
                    generated=caption_from_actions(a,phrases) if a else 'No activity was predicted.') for w,a in zip(windows,actions)]
        def softmax(x):
            v=np.exp(x-x.max(-1,keepdims=True));return v/v.sum(-1,keepdims=True)
        def path_rows(cache,config):
            seq=[]
            for logits,times in zip(cache['logits'],cache['times']):
                p=probability_smooth(softmax(logits),times,config['seconds'])
                path=hmm_path(np.log(p.clip(1e-12)),trans,start,config['weight'])
                seq.append(smooth_predictions(path,times,minimum_seconds=config['minimum']))
            return rows_from_actions(cache['windows'],seq)
        def full(rows):return [r for r in rows if r['full_window']]
        def rank(summary):return tuple(summary['metrics'][k] for k in ('sequence_exact','order_lcs','action_recall'))
        status('select_probability_HMM')
        candidates=[]
        for seconds in (0.,.3,.5):
            for minimum in (.6,1.,2.):
                for weight in (0.,.25,.5,1.):
                    cfg=dict(seconds=seconds,minimum=minimum,weight=weight)
                    summary=sequence_summary(full(path_rows(val,cfg)))
                    candidates.append(dict(config=cfg,validation=summary))
        prob=max([r for r in candidates if r['config']['weight']==0],key=lambda r:rank(r['validation']))
        hmm=max(candidates,key=lambda r:rank(r['validation']))
        count_pred=outputs(val,'independent-count').argmax(-1).numpy()+1
        def constrained(cache,config,predicted_counts):
            predicted,oracle=[],[]
            for logits,times,w,n in zip(cache['logits'],cache['times'],cache['windows'],predicted_counts):
                p=probability_smooth(softmax(logits),times,config['seconds'])
                paths=segment_paths(np.log(p.clip(1e-12)),times,config['minimum'],config['weight']*trans,config['weight']*start)
                assert int(n) in paths and len(w['actions']) in paths
                predicted.append(paths[int(n)]);oracle.append(paths[len(w['actions'])])
            return rows_from_actions(cache['windows'],predicted),rows_from_actions(cache['windows'],oracle)
        count_candidates=[]
        for minimum in (.6,1.,2.):
            cfg=dict(hmm['config'],minimum=minimum);rows,oracle=constrained(val,cfg,count_pred)
            count_candidates.append(dict(config=cfg,validation=sequence_summary(full(rows)),oracle_validation=sequence_summary(full(oracle))))
        selected_count=max(count_candidates,key=lambda r:rank(r['validation']))
        selection=dict(locked_at=datetime.now().astimezone().isoformat(),selection_split='validation full12 only',
                       probability=prob['config'],hmm=hmm['config'],count=selected_count['config'],
                       candidates=candidates,count_candidates=count_candidates,
                       probe_epochs={n:p['epoch'] for n,(_,p) in models.items()},
                       oracle='same predicted-count-selected settings; uses GT count only for explicitly labeled diagnostic')
        dump('selection.json',selection);completed.extend(['P1-B-fit','P1-C-selection'])
        extract('test');targets['test']=make_targets([w['actions'] for w in caches['test']['windows']])
        b_results={};c_results={}
        for split in ('validation','test'):
            cache=caches[split];predictions={};b_results[split]={}
            for name in models:
                out=outputs(cache,name)
                if name=='independent-count':
                    count=out.argmax(-1).numpy()+1
                    predictions[name]=dict(count=count.tolist(),logits=out.numpy().tolist())
                else:
                    presence,count,seq=decode_outputs(out)
                    predictions[name]=dict(presence=presence.tolist(),count=count.tolist(),ordered=rows_from_actions(cache['windows'],seq))
                    b_results[split][name]={subset:probe_metrics(out,targets[split],[w['actions'] for w in cache['windows']],
                       np.array([subset=='original_protocol' or w['full_window'] for w in cache['windows']])) for subset in ('full12','original_protocol')}
            predictions['majority-count']=int(torch.bincount(targets['train']['count'],minlength=3).argmax())+1
            dump(f'probe-predictions-{split}.json',predictions)
            predicted_counts=np.array(predictions['independent-count']['count'])
            calibrated,oracle=constrained(cache,selection['count'],predicted_counts)
            paths={'B1':old[split]['B1']['individual'],'B3':old[split]['B3'],
                   'probability':path_rows(cache,selection['probability']),'hmm':path_rows(cache,selection['hmm']),
                   'predicted-count':calibrated,'ORACLE-GT-count':oracle}
            dump(f'calibration-predictions-{split}.json',paths)
            c_results[split]={subset:{name:sequence_summary([r for r in rows if subset=='original_protocol' or r['full_window']]) for name,rows in paths.items()} for subset in ('full12','original_protocol')}
        dump('p1-b.json',b_results);dump('p1-c.json',c_results)
        for name,module in [('encoder',encoder),('projector',projector)]:
            assert all(torch.equal(v,initial[name][k]) for k,v in module.state_dict().items())
            assert all(p.grad is None and not p.requires_grad for p in module.parameters())
        assert all(sha(k)==v for k,v in inputs.items());assert all(sha(source/k)==v for k,v in sources.items())
        dump('verification.json',dict(original_inputs_unchanged=len(inputs),source_files_verified=len(sources),
             frozen_encoder_projector_bitwise_unchanged=True,no_encoder_projector_gradients=True,llm_loaded=False,
             max_H_length=max_length,P_width=width,feature_files=3,window_counts={s:len(c['windows']) for s,c in caches.items()},
             selection_locked_before_test=True,paired_B3_input_hashes=True))
        completed.extend(['P1-B-eval','P1-C-eval']);status('complete',finished=datetime.now().astimezone().isoformat())
    except BaseException as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
