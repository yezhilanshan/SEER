"""P0: freshly extract frozen Align features, fit H1/H4, compare paired readouts."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from continuous_mmwave.radar_text.align_readout import p0_specs,readout_summary,validate_window_inputs,validate_subjects,validate_encoder_identity
from continuous_mmwave.radar_text.config import RadarTextConfig
from continuous_mmwave.radar_text.data import collate_windows
from continuous_mmwave.radar_text.diagnostics import period,runs,steady_mask,confusion,classification_metrics
from continuous_mmwave.radar_text.head_ablation import fit_head,evaluate_head
from continuous_mmwave.radar_text.model import RadarEncoder
from continuous_mmwave.radar_text.runner import datasets,read_checkpoint,read_settings,seed_everything
from continuous_mmwave.radar_text.sliding_baselines import GRID,apply_saved_head,choose_thresholds,complete_window,score_windows,paired_exact


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):h.update(part)
    return h.hexdigest()


def input_sha(sample):
    h=hashlib.sha256()
    for key in ('rd','md','timestamps'):
        value=sample[key].numpy()
        h.update(key.encode());h.update(str((value.shape,str(value.dtype))).encode());h.update(value.tobytes())
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser()
    for name in ('pilot','heads','diagnostics','previous','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    pilot,heads,diagnostics,previous,root=map(Path,(args.pilot,args.heads,args.diagnostics,args.previous,args.output))
    root.mkdir(parents=True,exist_ok=False);(root/'features').mkdir();(root/'heads').mkdir()
    torch.set_num_threads(4);seed_everything(17)
    source=Path(__file__).resolve().parents[1]
    sources=json.loads((source/'source-manifest.json').read_text())
    assert all(sha(source/k)==v for k,v in sources.items())
    started=datetime.now().astimezone().isoformat();completed=[]
    def dump(name,value):
        p=root/name;temp=p.with_suffix(p.suffix+'.tmp')
        temp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');temp.replace(p)
    def status(state,**extra):
        dump('status.json',dict(state=state,started=started,completed=completed,**extra))
        print(json.dumps(dict(state=state,**extra)),flush=True)
    try:
        status('preflight')
        old_prov=json.loads((previous/'provenance.json').read_text())
        validate_encoder_identity(sha(pilot/'align/best.pt'),old_prov['original_input_sha256'])
        inputs=dict(old_prov['original_input_sha256'])
        for name in ('comparison.json','predictions-validation.json','predictions-test.json',
                     'windows-validation.json','windows-test.json','logits-validation.npz','logits-test.npz'):
            inputs[str(previous/name)]=sha(previous/name)
        inputs[str(heads/'head-ablation.json')]=sha(heads/'head-ablation.json')
        inputs[str(diagnostics/'feature-index.json')]=sha(diagnostics/'feature-index.json')
        assert all(sha(k)==v for k,v in inputs.items())
        settings=read_settings(pilot/'align.yaml');sets,phrases=datasets(settings,('train','validation','test'))
        assert settings['data']['window_seconds']==12 and settings['data']['stride_seconds']==6
        stage1=read_checkpoint(pilot/'stage1/best.pt');align=read_checkpoint(pilot/'align/best.pt')
        assert align['settings']['phrase_vocabulary']=={str(k):v for k,v in phrases.items()}
        encoder=RadarEncoder(RadarTextConfig(**align['config'])).cuda().eval().requires_grad_(False)
        encoder.load_state_dict(align['encoder'])
        original_state={k:v.detach().clone() for k,v in encoder.state_dict().items()}
        assert not any(p.requires_grad for p in encoder.parameters())
        subjects={s:sorted({m['subject_id'] for m in d.recordings.rows}) for s,d in sets.items()}
        assert not(set(subjects['train'])&set(subjects['validation']))
        assert not(set(subjects['test'])&(set(subjects['train'])|set(subjects['validation'])))
        validate_subjects(align,{s:d.recordings.rows for s,d in sets.items()})
        for split,d in sets.items():
            for meta in d.recordings.rows:
                p=Path(meta['path'])
                if not p.is_absolute():p=d.recordings.manifest.parent/p
                inputs.setdefault(str(p),sha(p))
        dump('provenance.json',dict(source_sha256=sources,original_input_sha256=inputs,subjects=subjects,
             encoder='Frozen Align best epoch3; state equality before/after',encoder_sha256=sha(pilot/'align/best.pt'),
             initialization='Same original Stage1 classifier for both newly trained Align heads, including standardized logit-preserving fold',
             train_protocol='Fresh nonoverlap12s, one original valid frame once, same as previous H1/H4',
             checkpoint_selection='validation nonoverlap frame Macro-F1; ties earliest',
             evaluation_protocol='Fresh original caption sliding12s stride6s, strict full12 primary; original tails supplemental',
             b3='Reused sealed previous predictions, exact per-window RD/MD/timestamps SHA plus Align checkpoint identity required',
             llm_loaded=False,encoder_trainable_parameters=0,head_parameters_each=3598,recipes=p0_specs()))
        prior_index=json.loads((diagnostics/'feature-index.json').read_text())
        feature_manifest={};arrays={};index={}
        def nonoverlap(split):
            data=sets[split];parts=[];entries=[]
            with torch.no_grad():
                for rid,meta in enumerate(data.recordings.rows):
                    sample=data.recordings[rid];times=sample.timestamps.numpy();labels=sample.labels.numpy()
                    dt=period(times);short=np.zeros(len(labels),bool);steady=steady_mask(labels,times)
                    for a,b,_ in runs(labels):
                        if times[b-1]-times[a]+dt<2-1e-8:short[a:b]=True
                    chunks=[];features=[];a=0
                    while a<len(labels):
                        b=max(a+1,int(np.searchsorted(times,times[a]+12-1e-7)))
                        hidden,_=encoder(sample.rd[a:b][None].cuda(),sample.md[a:b][None].cuda(),torch.tensor([b-a],device='cuda'))
                        features.append(hidden[0].cpu().numpy());chunks.append([a,b]);a=b
                    hidden=np.concatenate(features)
                    old=prior_index[split][rid]
                    assert old['subject_id']==meta['subject_id'] and old['sensor_id']==meta['sensor_id']
                    with np.load(diagnostics/'features'/old['cache']) as ref:
                        assert np.array_equal(ref['labels'],labels) and np.array_equal(ref['timestamps'],times)
                        assert np.array_equal(ref['short'],short) and np.array_equal(ref['steady'],steady)
                        assert np.allclose(hidden,ref['align_gru'],atol=5e-6,rtol=5e-6)
                    name=f'features/nonoverlap-{split}-{rid:03}.npz'
                    np.savez(root/name,features=hidden,labels=labels,timestamps=times,short=short,steady=steady)
                    feature_manifest[name]=dict(sha256=sha(root/name),bytes=(root/name).stat().st_size)
                    valid=(labels>=0)&(labels<14)
                    parts.append((hidden[valid],labels[valid],short[valid],steady[valid]))
                    entries.append(dict(subject=meta['subject_id'],sensor=meta['sensor_id'],recording=rid,path=name,chunks=chunks,frames=len(labels)))
                    status('extract_nonoverlap',split=split,recording=rid+1,total=len(data.recordings))
            arrays[split]=tuple(np.concatenate([p[i] for p in parts]) for i in range(4));index[split]=entries
            dump('feature-index.json',index);dump('feature-cache-manifest.json',feature_manifest)
        nonoverlap('train');nonoverlap('validation')
        fitting={s:arrays[s] for s in ('train','validation')};trained={};saved={}
        for spec in p0_specs():
            arm=spec['arm'];status('fit_head',arm=arm)
            path=root/'heads'/f'align-{arm}.pt'
            trained[arm]=fit_head(fitting,spec,stage1['classifier'],path)
            saved[arm]=torch.load(path,map_location='cpu',weights_only=True)
            trained[arm]['checkpoint_sha256']=sha(path)
            trained[arm]['results'],_=evaluate_head(saved[arm],fitting)
            completed.append(arm);dump('head-training.json',trained)
        dump('feature-counts.json',{s:len(a[0]) for s,a in arrays.items()})
        results={};selection=None;sliding_frames={};identities={}
        method_map={'B0':'Stage1_H1','B1':'Align_H1','B2':'Align_H4'}
        for split in ('validation','test'):
            status('extract_sliding',split=split,windows=0)
            data=sets[split];windows=[];times=[];hidden_parts=[];labels=[]
            old_windows=json.loads((previous/f'windows-{split}.json').read_text())
            old_predictions=json.loads((previous/f'predictions-{split}.json').read_text())
            with torch.no_grad():
                for raw in DataLoader(data,batch_size=2,collate_fn=collate_windows,shuffle=False,num_workers=0):
                    lengths=raw['lengths'].cuda();hidden,_=encoder(raw['rd'].cuda(),raw['md'].cuda(),lengths)
                    for i in range(len(lengths)):
                        sample=data[len(windows)];rid,start,stop,sid,actions=data.windows[len(windows)]
                        assert sid==raw['sample_ids'][i]
                        assert torch.equal(sample['rd'],raw['rd'][i,:lengths[i]]) and torch.equal(sample['md'],raw['md'][i,:lengths[i]])
                        meta=data.recordings.rows[rid];t=sample['timestamps'].numpy();dt=period(t);duration=float(t[-1]-t[0]+dt)
                        windows.append(dict(sample_id=sid,actions=actions,subject=meta['subject_id'],sensor=meta['sensor_id'],
                             start=start,stop=stop,duration=duration,period=dt,full_window=complete_window(duration,dt),input_sha256=input_sha(sample)))
                        times.append(t);hidden_parts.append(hidden[i,:len(t)].cpu().numpy());labels.append(sample['labels'].numpy())
                    if len(windows)%100==0:status('extract_sliding',split=split,windows=len(windows))
            identities[split]=validate_window_inputs(windows,old_windows)
            offsets=np.r_[0,np.cumsum([len(t) for t in times])];flat=np.concatenate(hidden_parts)
            name=f'features/sliding-{split}.npz';np.savez(root/name,features=flat,offsets=offsets)
            feature_manifest[name]=dict(sha256=sha(root/name),bytes=(root/name).stat().st_size)
            with np.load(previous/f'logits-{split}.npz') as old:
                assert np.array_equal(old['offsets'],offsets) and np.array_equal(old['labels'],np.concatenate(labels))
                values={'B0':old['B1'].copy()}
            for method,arm in (('B1','H1'),('B2','H4')):
                values[method]=apply_saved_head(torch.from_numpy(flat).cuda(),saved[arm]).cpu().numpy()
            np.savez(root/f'logits-{split}.npz',offsets=offsets,timestamps=np.concatenate(times),labels=np.concatenate(labels),**values)
            dump(f'windows-{split}.json',windows)
            preds={m:[v[a:b].argmax(-1) for a,b in zip(offsets[:-1],offsets[1:])] for m,v in values.items()}
            if split=='validation':
                grid={m:[] for m in preds}
                for m in preds:
                    for med,minimum in GRID:
                        th=dict(median_seconds=med,minimum_seconds=minimum)
                        rows=score_windows(windows,preds[m],times,phrases,th)
                        grid[m].append(dict(th,metrics=readout_summary([r for r in rows if r['full_window']])['metrics']))
                selection=choose_thresholds(grid,split='validation');selection['locked_at']=datetime.now().astimezone().isoformat()
                selection['method_map']=method_map
                dump('threshold-selection.json',selection)
                dump('locked-heads.json',{a:dict(epoch=s['epoch'],sha256=trained[a]['checkpoint_sha256']) for a,s in saved.items()})
            assert selection is not None
            split_rows={'B3':old_predictions['B3']}
            for m in preds:
                split_rows[m]={}
                for variant,th in [('individual',selection['individual'][m]),('common',selection['common']),('unsmoothed',dict(median_seconds=0,minimum_seconds=0))]:
                    split_rows[m][variant]=score_windows(windows,preds[m],times,phrases,th)
            for row,w in zip(split_rows['B3'],windows):
                assert row['sample_id']==w['sample_id'] and row['reference_actions']==w['actions'] and row['full_window']==w['full_window']
            dump(f'predictions-{split}.json',split_rows)
            summary={}
            for subset in ('full12','original_protocol'):
                keep=lambda rows:[r for r in rows if subset=='original_protocol' or r['full_window']]
                summary[subset]={'B3':readout_summary(keep(split_rows['B3']))}
                for m in preds:summary[subset][m]={v:readout_summary(keep(rows)) for v,rows in split_rows[m].items()}
                summary[subset]['paired_vs_B3']={m:{v:paired_exact(keep(rows),keep(split_rows['B3'])) for v,rows in split_rows[m].items()} for m in preds}
                summary[subset]['by_sensor']={s:{m:readout_summary([r for r in keep(split_rows[m] if m=='B3' else split_rows[m]['individual']) if r['sensor']==s]) for m in ('B0','B1','B2','B3')} for s in sorted({r['sensor'] for r in windows})}
            results[split]=summary;dump('comparison.json',dict(method_map=method_map,results=results,selection=selection))
            sliding_frames[split]={m:classification_metrics(confusion(np.concatenate(labels),v.argmax(-1))) for m,v in values.items()}
            dump('sliding-frame-classification.json',sliding_frames);dump('feature-cache-manifest.json',feature_manifest)
            completed.append(split)
        # Test nonoverlap is extracted only after all selections and paired sliding results are fixed.
        nonoverlap('test')
        for arm in saved:
            trained[arm]['results'],_=evaluate_head(saved[arm],arrays)
        dump('head-training.json',trained);dump('feature-counts.json',{s:len(a[0]) for s,a in arrays.items()})
        assert all(torch.equal(v,original_state[k]) for k,v in encoder.state_dict().items())
        assert all(p.grad is None and not p.requires_grad for p in encoder.parameters())
        assert all(sha(k)==v for k,v in inputs.items())
        assert all(sha(source/k)==v for k,v in sources.items())
        dump('verification.json',dict(original_inputs_unchanged=len(inputs),source_files_verified=len(sources),
             encoder_state_bitwise_unchanged=True,encoder_has_no_gradients=True,llm_loaded=False,
             paired_window_inputs_verified=identities,selection_locked_before_test=True,
             fresh_feature_files=len(feature_manifest),head_parameters_trained_each=3598))
        status('complete',finished=datetime.now().astimezone().isoformat())
    except BaseException as error:
        status('failed',error=str(error));raise


if __name__=='__main__':main()
