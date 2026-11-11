"""Fresh sliding-window inference for B0/B1/B2/B3 with identical raw sensor inputs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from continuous_mmwave.radar_text.config import RadarTextConfig
from continuous_mmwave.radar_text.data import collate_windows, caption_from_actions
from continuous_mmwave.radar_text.diagnostics import period
from continuous_mmwave.radar_text.metrics import score_caption
from continuous_mmwave.radar_text.model import RadarEncoder
from continuous_mmwave.radar_text.runner import (
    datasets, read_checkpoint, read_settings, restore_caption_model, check_holdout_subjects, seed_everything,
)
from continuous_mmwave.radar_text.sliding_baselines import (
    GRID, apply_saved_head, choose_thresholds, complete_window, pair_previous,
    paired_exact, score_windows, summarize,
)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024), b''):value.update(block)
    return value.hexdigest()


def input_sha(sample):
    h = hashlib.sha256()
    for key in ('rd','md','timestamps'):
        value = sample[key].numpy()
        h.update(key.encode());h.update(str((value.shape,str(value.dtype))).encode());h.update(value.tobytes())
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--heads', required=True)
    parser.add_argument('--diagnostics', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    pilot, heads_root, diagnostics, root = map(Path, (args.pilot,args.heads,args.diagnostics,args.output))
    root.mkdir(parents=True, exist_ok=False)
    (root/'features').mkdir()
    torch.set_num_threads(4);seed_everything(17)
    started = datetime.now().astimezone().isoformat()
    source = Path(__file__).resolve().parents[1]
    source_hashes = json.loads((source/'source-manifest.json').read_text())
    assert all(sha(source/name)==expected for name,expected in source_hashes.items())
    completed = []

    def dump(name, value):
        path = root/name
        temporary = path.with_suffix(path.suffix+'.tmp')
        temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
        temporary.replace(path)

    def status(state, **extra):
        dump('status.json',dict(state=state,started=started,completed=completed,**extra))

    status('preflight')
    try:
        settings = read_settings(pilot/'align.yaml')
        assert settings['data']['window_seconds']==12 and settings['data']['stride_seconds']==6
        assert settings['training']['batch_size']==2 and settings['max_new_tokens']==96
        sets, phrases = datasets(settings, ('validation','test'))
        old_verification = json.loads((pilot/'verification.json').read_text())
        hashes = dict(old_verification['data_and_base_weights_sha256'])
        hashes.update({str(pilot/stage/'best.pt'):v['sha256'] for stage,v in old_verification['checkpoints'].items()})
        head_report = json.loads((heads_root/'head-ablation.json').read_text())
        for name in ('H1','H4'):
            row = head_report['panels']['original_optimizer'][name]
            hashes[str(heads_root/row['checkpoint'])] = row['checkpoint_sha256']
        assert all(sha(name)==value for name,value in hashes.items()), 'input checkpoint/manifest mismatch'
        for dataset in sets.values():
            for meta in dataset.recordings.rows:
                p = Path(meta['path'])
                if not p.is_absolute():p=dataset.recordings.manifest.parent/p
                hashes[str(p)] = sha(p)
        stage1 = read_checkpoint(pilot/'stage1/best.pt')
        encoder = RadarEncoder(RadarTextConfig(**stage1['config'])).cuda().eval().requires_grad_(False)
        encoder.load_state_dict(stage1['encoder'])
        width = stage1['classifier']['weight'].shape[1]
        saved_heads = {'B0':dict(head=stage1['classifier'],mean=torch.zeros(width),std=torch.ones(width))}
        for method,name in (('B1','H1'),('B2','H4')):
            saved_heads[method] = torch.load(heads_root/f'original_optimizer-{name}.pt',map_location='cpu',weights_only=True)
            assert saved_heads[method]['spec']['panel']=='original_optimizer'
        caption, aligned = restore_caption_model(pilot/'align/best.pt','cuda')
        caption.requires_grad_(False)
        assert not any(p.requires_grad for p in encoder.parameters())
        assert not any(p.requires_grad for p in caption.parameters())
        for split,dataset in sets.items():
            check_holdout_subjects(stage1,dataset.recordings.rows,test=split=='test')
            check_holdout_subjects(aligned,dataset.recordings.rows,test=split=='test')
        assert aligned['settings']['phrase_vocabulary']=={str(k):v for k,v in phrases.items()}
        provenance = dict(source_sha256=source_hashes,original_input_sha256=hashes,
                          settings=settings,head_panel='original_optimizer',heads={'B1':'H1','B2':'H4'},
                          original_training_cache='nonoverlap12s (heads trained previously; not reused for evaluation)',
                          fresh_extraction='Stage1 eval encoder per caption sliding window; all classification heads share hidden tensor',
                          b3='Fresh frozen Align+Qwen inference on identical batch RD/MD/lengths; greedy max96',
                          trainable_parameters=0,seed=17,window_seconds=12,stride_seconds=6,
                          primary_subset='complete nominal12s windows, one sampling period tolerance',
                          sensor_input_identity='same raw tensors passed to encoder and caption.generate in one loop',
                          smoothing_grid=GRID,selection_split='validation only',selection_subset='complete nominal12s',
                          original_partial_tails='also reported to compare against historical484 test windows')
        dump('provenance.json',provenance)
        selection = None
        results, reproduction, feature_manifest = {}, {}, {}
        for split,dataset in sets.items():
            windows, b3_rows, timestamps, hidden_parts, label_parts = [], [], [], [], []
            logits = {method:[] for method in saved_heads}
            old = json.loads((pilot/f'{split}-evaluation.json').read_text())
            prior_windows = json.loads((diagnostics/f'caption-windows-{split}.json').read_text())
            status('fresh_inference',split=split,windows_completed=0,total=len(dataset))
            loader = DataLoader(dataset,batch_size=2,collate_fn=collate_windows,shuffle=False,num_workers=0)
            with torch.no_grad():
                for raw in loader:
                    rd,md,lengths = raw['rd'].cuda(),raw['md'].cuda(),raw['lengths'].cuda()
                    hidden,_ = encoder(rd,md,lengths)
                    values = {method:apply_saved_head(hidden,head).cpu().numpy() for method,head in saved_heads.items()}
                    texts = caption.generate(rd,md,lengths,max_new_tokens=96)
                    for i,text in enumerate(texts):
                        index = len(windows)
                        sample = dataset[index]
                        rid,start,stop,sample_id,actions = dataset.windows[index]
                        assert sample_id==raw['sample_ids'][i] and actions==raw['actions'][i]
                        assert torch.equal(sample['rd'],raw['rd'][i,:lengths[i]]) and torch.equal(sample['md'],raw['md'][i,:lengths[i]])
                        meta = dataset.recordings.rows[rid]
                        time = sample['timestamps'].numpy();dt=period(time)
                        duration=float(time[-1]-time[0]+dt)
                        row = dict(sample_id=sample_id,subject=meta['subject_id'],sensor=meta['sensor_id'],
                                   environment=meta['environment_id'],recording=rid,start=start,stop=stop,actions=actions,
                                   duration=duration,period=dt,full_window=complete_window(duration,dt),input_sha256=input_sha(sample))
                        windows.append(row);timestamps.append(time);label_parts.append(sample['labels'].numpy())
                        hidden_parts.append(hidden[i,:len(time)].cpu().numpy())
                        for method in logits:logits[method].append(values[method][i,:len(time)])
                        b3_rows.append(dict(sample_id=sample_id,subject=row['subject'],sensor=row['sensor'],
                                            full_window=row['full_window'],reference_actions=actions,generated=text,
                                            **score_caption(text,actions,phrases)))
                    if len(windows)%50==0 or len(windows)==len(dataset):
                        status('fresh_inference',split=split,windows_completed=len(windows),total=len(dataset))
                        print(json.dumps(dict(stage='fresh_inference',split=split,windows=len(windows),total=len(dataset))),flush=True)
            assert [(r['sample_id'],r['actions'],r['start'],r['stop']) for r in windows]==[(r['sample_id'],r['actions'],r['start'],r['stop']) for r in prior_windows]
            previous = pair_previous(windows,old['results'])
            offsets=np.r_[0,np.cumsum([len(t) for t in timestamps])]
            dump(f'windows-{split}.json',windows)
            np.savez(root/f'features/sliding-{split}.npz',features=np.concatenate(hidden_parts),offsets=offsets)
            feature_manifest[split]=dict(path=f'features/sliding-{split}.npz',sha256=sha(root/f'features/sliding-{split}.npz'))
            np.savez(root/f'logits-{split}.npz',offsets=offsets,timestamps=np.concatenate(timestamps),
                     labels=np.concatenate(label_parts),**{m:np.concatenate(v) for m,v in logits.items()})
            with np.load(diagnostics/f'caption-logits-{split}.npz') as old_logits:
                current=np.concatenate(logits['B0'])
                assert np.array_equal(offsets,old_logits['offsets'])
                assert np.array_equal(np.concatenate(label_parts),old_logits['labels'])
                reproduction[split]=dict(b0_max_logit_difference=float(np.max(np.abs(current-old_logits['logits']))),
                                          b0_frame_prediction_differences=int((current.argmax(-1)!=old_logits['logits'].argmax(-1)).sum()),
                                          b3_text_differences=sum(a['generated']!=b['generated'] for a,b in zip(b3_rows,previous)),
                                          b3_action_sequence_differences=sum(a['predicted_actions']!=b['predicted_actions'] for a,b in zip(b3_rows,previous)))
            predictions={m:[x.argmax(-1) for x in v] for m,v in logits.items()}
            if split=='validation':
                status('selecting_validation_thresholds')
                grid={}
                for method in predictions:
                    grid[method]=[]
                    for median,minimum in GRID:
                        threshold=dict(median_seconds=median,minimum_seconds=minimum)
                        rows=score_windows(windows,predictions[method],timestamps,phrases,threshold)
                        grid[method].append(dict(threshold,metrics=summarize([r for r in rows if r['full_window']])['metrics']))
                selection=choose_thresholds(grid,split='validation')
                selection['locked_at']=datetime.now().astimezone().isoformat()
                dump('threshold-selection.json',selection)
            assert selection is not None
            split_rows={'B3':b3_rows}
            for method in predictions:
                split_rows[method]={}
                for variant,threshold in [('common',selection['common']),('individual',selection['individual'][method]),
                                           ('unsmoothed',dict(median_seconds=0.,minimum_seconds=0.))]:
                    split_rows[method][variant]=score_windows(windows,predictions[method],timestamps,phrases,threshold)
            dump(f'predictions-{split}.json',split_rows)
            summary={}
            for subset in ('full12','original_protocol'):
                filtered=lambda rows:[r for r in rows if subset=='original_protocol' or r['full_window']]
                methods={'B3':summarize(filtered(b3_rows))}
                for method in predictions:
                    methods[method]={variant:summarize(filtered(rows)) for variant,rows in split_rows[method].items()}
                methods['paired_vs_B3']={method:{variant:paired_exact(filtered(rows),filtered(b3_rows))
                                                  for variant,rows in split_rows[method].items()} for method in predictions}
                methods['by_sensor']={sensor:{method:summarize([r for r in filtered(b3_rows if method=='B3' else split_rows[method]['common']) if r['sensor']==sensor]) for method in ('B0','B1','B2','B3')}
                                      for sensor in sorted({r['sensor'] for r in windows})}
                methods['by_action_count']={str(n):{method:summarize([r for r in filtered(b3_rows if method=='B3' else split_rows[method]['common']) if len(r['reference_actions'])==n]) for method in ('B0','B1','B2','B3')} for n in (1,2,3)}
                summary[subset]=methods
            results[split]=summary
            completed.append(split)
            dump('comparison.json',dict(results=results,reproduction=reproduction,selection=selection))
            dump('feature-cache-manifest.json',feature_manifest)
        status('verifying_immutable_inputs')
        assert all(sha(name)==value for name,value in hashes.items()), 'original input changed during inference'
        dump('verification.json',dict(input_files_unchanged=len(hashes),source_files_verified=len(source_hashes),
                                      fresh_sliding_feature_splits=2,trained_parameters=0,
                                      all_methods_share_input_tensors=True,selection_locked_before_test=True,
                                      reproduction=reproduction,features_retained_server=True))
        status('complete',finished=datetime.now().astimezone().isoformat())
    except BaseException as error:
        status('failed',error=str(error),finished=datetime.now().astimezone().isoformat())
        raise


if __name__=='__main__':
    main()
