"""Evaluate existing HARrad models with shared front-end and metric definitions."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def conservative_events(events, truth, known, threshold=.5, require_class=True):
    from continuous_mmwave.event_gating.metrics import match_ordered

    def iou(i, j):
        p, t = events[i], truth[j]
        overlap = max(0, min(p['stop_frame'], t['stop']) - max(p['start_frame'], t['start']))
        return overlap / (max(p['stop_frame'], t['stop']) - min(p['start_frame'], t['start']))

    def allowed(i, j):
        return iou(i, j) >= threshold and (not require_class or events[i]['class_id'] == truth[j]['label'])

    pairs = match_ordered(events, truth, iou, allowed)
    used = {i for i, _ in pairs}
    unmatched = [p for i, p in enumerate(events) if i not in used]
    false = [p for p in unmatched if known[p['start_frame']:p['stop_frame']].any()]
    mixed = [p for p in false if not known[p['start_frame']:p['stop_frame']].all()]
    tp, fp, fn = len(pairs), len(false), len(truth) - len(pairs)
    per_class = []
    for label in range(12):
        hits = sum(truth[j]['label'] == label for _, j in pairs)
        wrong = sum(p['class_id'] == label for p in false)
        support = sum(t['label'] == label for t in truth)
        denominator = hits + wrong + support
        per_class.append(dict(tp=hits, fp=wrong, fn=support-hits,
                              f1=2*hits/denominator if denominator else 0.))
    return dict(tp=tp, fp=fp, fn=fn, f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,
                mixed_fp=len(mixed), unknown=len(unmatched)-fp, per_class=per_class,
                macro_f1=float(np.mean([p['f1'] for p in per_class])),
                matches=[dict(prediction=i, truth=j, iou=iou(i,j)) for i,j in pairs])


def score(result, record, confirmations):
    from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
    from continuous_mmwave.harrad.training import classification_metrics
    metric = evaluate_rollout(result, record)
    known = np.zeros(record['frames'], bool)
    labels = np.full(record['frames'], -100, dtype=int)
    for row in record['instances']:
        known[row['start']:row['stop']] = True
        labels[row['start']:row['stop']] = row['label']
    metric['conservative'] = {}
    for threshold in (.3, .5):
        metric['conservative'][str(threshold)] = conservative_events(
            result['events'], record['instances'], known, threshold)
    metric['interval_only'] = conservative_events(result['events'], record['instances'], known, .5, False)
    online = np.asarray(result['trace']['probabilities'])
    # Every row is the most recent class distribution available at that frame's cutoff.
    metric['online'] = classification_metrics(np.log(np.maximum(online[known], 1e-30)), labels[known])
    for tolerance, boundary in metric['boundary'].items():
        pairs = boundary['matches']
        truth = {pair['truth'] for pair in pairs}
        boundary['start_hits'] = sum(0<t['start']<record['frames'] and t['start'] in truth for t in record['instances'])
        boundary['end_hits'] = sum(0<t['stop']<record['frames'] and t['stop'] in truth for t in record['instances'])
        delays = [(confirmations[p['predicted']] - p['truth'])/15 for p in pairs]
        boundary['confirmation_delay_s'] = delays
        boundary['confirmation_p90_s'] = float(np.quantile(delays,.9)) if delays else None
        # Includes missed truth in the denominator, and requires post-boundary confirmation.
        boundary['deadline_recall_0.5s'] = sum(0<=d<=.5 for d in delays)/(boundary['tp']+boundary['fn'])
    metric['eof_events'] = sum(e['closure']=='eof' for e in result['events'])
    return metric


def eventformer_rollout(model, cells, cutoffs):
    import torch
    count = len(cells)
    boundaries = sorted({int(p) for p in cutoffs if 0<int(p)<count})
    fires = np.zeros(count, bool)
    fires[boundaries] = True
    probabilities = np.zeros((count,12), dtype=np.float32)
    published_cutoffs = np.zeros(count, dtype=np.int64)
    state = model.former.initial_state(1)
    observed = torch.zeros(1,dtype=torch.long,device='cuda')
    pending = None
    start = 0
    events = []
    times = []

    def distribution(value):
        return model.read(value).softmax(-1)[0].cpu().numpy()

    def close(stop, closure):
        value = state
        if pending is not None:
            pair = torch.cat((pending,torch.zeros_like(pending)),0)[None]
            value, _ = model.advance(pair,torch.tensor([[True,False]],device='cuda'),state,observed)
        prob = distribution(value)
        events.append(dict(start_frame=start,stop_frame=stop,class_id=int(prob.argmax()),
                           probabilities=prob.tolist(),confirmed_at_frame=stop,closure=closure))

    available = distribution(state)
    published_at = 0
    boundary_set = set(boundaries)
    with torch.inference_mode():
        for index in range(count):
            if index in boundary_set:
                close(index,'fire')
                start = index
                state = model.former.initial_state(1)
                observed.zero_()
                pending = None
            frame = torch.as_tensor(np.array(cells[index]),device='cuda')[None]
            if pending is None:
                pending = frame
            else:
                sampled = len(times)<64
                if sampled:
                    torch.cuda.synchronize()
                    begin=time.perf_counter()
                state, observed = model.advance(torch.cat((pending,frame),0)[None],
                    torch.ones(1,2,dtype=torch.bool,device='cuda'),state,observed)
                available = distribution(state)
                published_at = index + 1
                pending = None
                if sampled:
                    torch.cuda.synchronize()
                    times.append(time.perf_counter()-begin)
            probabilities[index]=available
            published_cutoffs[index]=published_at
        close(count,'eof')
    return dict(events=events,trace=dict(frame=np.arange(count),fire=fires,
                probabilities=probabilities,published_cutoff=published_cutoffs)),times


def summarize(rows):
    result = {}
    for arm in sorted({r['arm'] for r in rows}):
        group=[r for r in rows if r['arm']==arm]
        def counts(path):
            values=[]
            for r in group:
                item=r['metric']
                for key in path: item=item[key]
                values.append(item)
            tp=sum(v['tp'] for v in values); fp=sum(v['fp'] for v in values); fn=sum(v['fn'] for v in values)
            return dict(tp=tp,fp=fp,fn=fn,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        boundary=counts(('boundary','0.5'))
        valid=sum(r['metric']['boundary']['0.5']['valid_duration_s'] for r in group)
        boundary['fp_min']=60*boundary['fp']/valid
        boundary['recall']=boundary['tp']/(boundary['tp']+boundary['fn'])
        delay=[d for r in group for d in r['metric']['boundary']['0.5']['confirmation_delay_s']]
        boundary['confirmation_p90_s']=float(np.quantile(delay,.9)) if delay else None
        event=counts(('conservative','0.5'))
        per_class=[]
        for label in range(12):
            cs=[r['metric']['conservative']['0.5']['per_class'][label] for r in group]
            tp=sum(c['tp'] for c in cs); fp=sum(c['fp'] for c in cs); fn=sum(c['fn'] for c in cs)
            per_class.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)
        event['macro_f1']=float(np.mean(per_class))
        result[arm]=dict(boundary=boundary,conservative_event=event,
                         historical_event=counts(('events','0.5')),interval_only=counts(('interval_only',)),
                         record_mean_online_macro_f1=float(np.mean([r['metric']['online']['macro_f1'] for r in group])),
                         unknown_fires=sum(r['metric']['boundary']['0.5']['unscored_unknown'] for r in group),
                         mixed_fp=sum(r['metric']['conservative']['0.5']['mixed_fp'] for r in group),
                         extra_fires=sum(r['metric']['oversegmentation']['extra_fires_inside_instances'] for r in group))
    return result


def export_holdout(root, output, config):
    """Build the same frozen cells/hidden on two already-examined holdout subjects."""
    import h5py
    import torch
    from continuous_mmwave.harrad.data import read_instances
    from continuous_mmwave.harrad.model import SegmentClassifier
    from continuous_mmwave.harrad.preprocess import frame_features
    from continuous_mmwave.harrad.streaming import FrozenRDStream
    raw=Path.home()/'SEER/datasets/HARrad/harrad'
    annotation=raw/'labels/activities.csv'
    assert digest(annotation)=='0069b94786e31459fb5f17196f1be37b47bb71168ffef27caa27944a2ab5b9dd'
    lengths={p.name:None for p in raw.glob('*.hdf5')}
    for name in lengths:
        with h5py.File(raw/name,'r') as f: lengths[name]=len(f['radar'])
    instances=read_instances(annotation,lengths)
    checkpoint=torch.load(config['radar_checkpoint'],map_location='cpu',weights_only=False)
    backbone=SegmentClassifier('rd').eval()
    backbone.load_state_dict(checkpoint['model'])
    frontend=FrozenRDStream(backbone).cuda().eval()
    with np.load(Path(config['baseline_output'])/'normalizers.npz') as stats:
        mean,scale=stats['rd_mean'],stats['rd_scale']
    records=[]
    for file in ('subject8_01.hdf5','subject9_01.hdf5'):
        before=(raw/file).stat()
        count=lengths[file]
        dest=output/Path(file).stem
        dest.mkdir(parents=True,exist_ok=False)
        cells=np.lib.format.open_memmap(dest/'cells.npy',mode='w+',dtype='float32',shape=(count,4,64))
        hidden=np.lib.format.open_memmap(dest/'continuous_hidden.npy',mode='w+',dtype='float32',shape=(count,128))
        state=None
        with h5py.File(raw/file,'r') as stream,torch.inference_mode():
            for start in range(0,count,64):
                rd=np.stack([frame_features(x)[1] for x in stream['radar'][start:start+64]])
                x=torch.from_numpy(np.clip((rd-mean)/scale,-8,8))[None].cuda()
                features=frontend.forward_chunk(x,state=state)
                state=features['state']
                stop=start+len(rd)
                cells[start:stop]=features['cells'][0].cpu().numpy()
                hidden[start:stop]=features['hidden'][0].cpu().numpy()
        cells.flush(); hidden.flush()
        after=(raw/file).stat()
        assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
        records.append(dict(file=file,subject=file.split('_')[0],role='test',frames=count,
            instances=[r for r in instances if r['file']==file],cells_path=str(dest/'cells.npy'),
            continuous_path=str(dest/'continuous_hidden.npy'),artifacts={
                'cells.npy':dict(sha256=digest(dest/'cells.npy')),
                'continuous_hidden.npy':dict(sha256=digest(dest/'continuous_hidden.npy'))}))
    del frontend,backbone,checkpoint
    torch.cuda.empty_cache()
    write(output/'feature-manifest.json',records)
    return records


def main():
    import torch
    from continuous_mmwave.harrad.event_semantic import build_production
    from continuous_mmwave.harrad.training import seed_everything
    from continuous_mmwave.harrad.boundary import BoundaryHead,decode_boundaries
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    root=args.root; out=args.output
    out.mkdir(parents=True,exist_ok=False)
    saef=root.parent/'saef_mvp_20261104_v1'
    warmup=root/'shared-front-event-warmup-v1'
    boundary=root/'shared-front-boundary-v1'
    def status(phase,**kw): write(out/'status.json',dict(phase=phase,**kw,updated_unix=time.time()))
    seed_everything(42,4)
    config=read(warmup/'config.json')
    assert config['radar_checkpoint_sha256']=='13674006bb1088fba10dcfbfb028aa38e15b560eb3b4eb79575c91a6436dc03f'
    assert digest(config['radar_checkpoint'])==config['radar_checkpoint_sha256']
    complete=read(warmup/'training/complete.json')
    assert digest(warmup/'training/best.pt')==complete['best_sha256']
    model=build_production(config)
    model.load_learned(torch.load(warmup/'training/best.pt',map_location='cpu',weights_only=False)['learned'])
    model.requires_grad_(False).eval()
    features=read(warmup/'feature-manifest.json')
    records=[r for r in features if r['role']=='validation']
    assert len(records)==4
    status('exporting_previously_evaluated_subject8_9')
    holdout=export_holdout(root,out/'holdout-features',config)
    records+=holdout
    protocol=dict(scope='existing trained native-rate systems, subject7 development and subject8/9 cross-subject confirmation',
                  idea3='supervised start/end head + EventFormer; predictive Reset pending training',
                  idea6='locked SAEF B2/B3-A/B3-B',frontend_sha256=config['radar_checkpoint_sha256'],
                  event_sha256=complete['best_sha256'],saef_sha256=digest(saef/'run_v1/model.pt'),
                  event_primary='unmatched predictions touching known frames count as false positives',
                  primary_tolerance_s=.5,primary_iou=.5,subject7_used_for_selection=True,
                  native_updates_hz=dict(idea3=7.5,idea6=15),
                  calibration='historical operating points; equal-FP-budget mechanism comparison pending',
                  online_readout='hold last published class until next causal update, including after reset',
                  seeds=[42],test_signal_values_read=True,script_sha256=digest(__file__))
    write(out/'protocol.json',protocol)
    saef_protocol = read(saef/'run_v1/protocol.json')
    assert saef_protocol['settings']['locks']['radar_checkpoint']['sha256'] == config['radar_checkpoint_sha256']
    locked_protocol = read(saef/'locked_saef_subject7_v1/locked-protocol.json')
    assert locked_protocol['checkpoint_sha256'] == protocol['saef_sha256']
    assert read(saef/'locked_saef_subject8_9_v1/locked-protocol.json') == locked_protocol
    saef_records = {r['file']:r for r in read(saef/'rebuilt_features_v1/feature-manifest.json')}
    saef_records.update({r['file']:r for r in holdout})
    imported_files = {}
    head=BoundaryHead().cuda().eval()
    saved=torch.load(boundary/'training/best.pt',map_location='cpu',weights_only=False)
    head.load_state_dict(saved['model'])
    boundary_meta=read(boundary/'data/manifest.json')
    inputs={r['file']:r for r in boundary_meta}
    rows=[]
    start_time=time.time()
    for index,record in enumerate(records):
        file=record['file']; stem=Path(file).stem
        source_record = saef_records[file]
        assert all(record[k] == source_record[k] for k in ('subject','frames','instances'))
        cells=np.load(record['cells_path'],mmap_mode='r')
        assert digest(record['cells_path'])==record['artifacts']['cells.npy']['sha256']
        if record['subject']=='subject7':
            item=inputs[file]
            with np.load(Path(item['cache'])/'targets.npz') as targets:
                cutoffs=targets['cutoffs'].copy()
            x=np.load(Path(item['cache'])/'inputs.npy')
        else:
            from continuous_mmwave.harrad.boundary import continuous_inputs
            hidden=np.load(record['continuous_path'],mmap_mode='r')
            x,cutoffs=continuous_inputs(hidden)
            with np.load(boundary/'data/normalizers.npz') as stats:
                x=np.clip((x-stats['mean'])/stats['scale'],-8,8).astype('float32')
        with torch.inference_mode(): probability=head(torch.as_tensor(x,device='cuda')).sigmoid().cpu().numpy()
        emit=[decode_boundaries(probability[:,k],cutoffs,saved['thresholds'][name]) for k,name in enumerate(('start','end'))]
        union=np.unique(np.concatenate(emit)).tolist()
        conditions=[('idea3-start-end',union)]
        for arm in ('B2','B3-A','B3-B'):
            locked_group='locked_saef_subject7_v1' if record['subject']=='subject7' else 'locked_saef_subject8_9_v1'
            folder=saef/locked_group/arm
            for path in (folder/(stem+'-trace.npz'),folder/(stem+'-events.json')):
                imported_files[str(path)] = digest(path)
            with np.load(folder/(stem+'-trace.npz'),allow_pickle=False) as data:
                trace={k:data[k].copy() for k in data.files}
            events=read(folder/(stem+'-events.json'))
            result=dict(events=events,trace=trace)
            np.testing.assert_array_equal(trace['frame'],np.arange(record['frames']))
            confirmations={int(i):int(i)+1 for i in np.flatnonzero(trace['fire'])}
            status('native_saef',arm=arm,file=file)
            metric=score(result,record,confirmations)
            dest=out/(arm+'-native'); dest.mkdir(exist_ok=True)
            write(dest/(stem+'-metrics.json'),metric)
            write(dest/(stem+'-events.json'),events)
            rows.append(dict(arm=arm+'-native',subject=record['subject'],file=file,metric=metric))
            # A causal common-reader comparison applies a fire only after observing frame i.
            conditions.append((arm+'-shared-reader',[i+1 for i in np.flatnonzero(trace['fire'])]))
        for arm,emissions in conditions:
            status('eventformer',arm=arm,file=file,record=index+1)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            begin=time.time()
            result,times=eventformer_rollout(model,cells,emissions)
            torch.cuda.synchronize()
            inference_seconds=time.time()-begin
            confirmations={int(i):int(i) for i in np.flatnonzero(result['trace']['fire'])}
            metric=score(result,record,confirmations)
            metric['runtime']=dict(seconds=inference_seconds,
                scope='EventFormer causal replay only; excludes detector and metric computation',
                sampled_update_ms_median=1000*float(np.median(times)),
                sampled_update_ms_p95=1000*float(np.quantile(times,.95)),
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            dest=out/arm; dest.mkdir(exist_ok=True)
            write(dest/(stem+'-metrics.json'),metric)
            write(dest/(stem+'-events.json'),result['events'])
            np.savez_compressed(dest/(stem+'-trace.npz'),**result['trace'])
            rows.append(dict(arm=arm,subject=record['subject'],file=file,metric=metric))
        write(out/'partial-summary.json',summarize(rows))
        write(out/'imported-files.json',imported_files)
    assert all(digest(path)==value for path,value in imported_files.items())
    subjects=sorted({r['subject'] for r in rows})
    report=dict(protocol=protocol,pooled=summarize(rows),records=rows,seconds=time.time()-start_time,
        by_subject={s:summarize([r for r in rows if r['subject']==s]) for s in subjects},
        heldout_pooled=summarize([r for r in rows if r['subject']!='subject7']))
    write(out/'complete.json',report)
    status('complete')


if __name__=='__main__':
    main()
