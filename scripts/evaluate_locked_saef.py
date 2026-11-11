"""Confirm frozen SAEF operating points, then evaluate untouched HARrad subjects."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from continuous_mmwave.event_gating.data import locked_path, sha256, write_json
from continuous_mmwave.event_gating.mvp.data import load_records
from continuous_mmwave.event_gating.mvp.metrics import evaluate_rollout
from continuous_mmwave.event_gating.mvp.model import rollout
from continuous_mmwave.event_gating.mvp.runner import load_checkpoint, modules, record_identity
from continuous_mmwave.harrad.data import CLASSES, read_instances
from continuous_mmwave.harrad.model import SegmentClassifier
from continuous_mmwave.harrad.preprocess import frame_features
from continuous_mmwave.harrad.streaming import FrozenRDStream


POINTS = {'B2': {'mechanism':'lif','tau':1.,'threshold':1.},
          'B3-A': {'mechanism':'lif','tau':1.,'threshold':1.},
          'B3-B': {'mechanism':'lif','tau':1.,'threshold':.895}}
HOLDOUT = {'subject8_01.hdf5':{'frames':9017,'bytes':2364266736,'instances':141},
           'subject9_01.hdf5':{'frames':9016,'bytes':2364004592,'instances':182}}
ACTIVITIES_SHA256 = '0069b94786e31459fb5f17196f1be37b47bb71168ffef27caa27944a2ab5b9dd'


def read(path):
    return json.loads(Path(path).read_text())


def source_digests():
    import continuous_mmwave.event_gating.mvp.metrics as metrics_module
    import continuous_mmwave.event_gating.mvp.model as model_module
    import continuous_mmwave.event_gating.metrics as matching_module
    import continuous_mmwave.harrad.preprocess as preprocess_module
    import continuous_mmwave.harrad.streaming as streaming_module
    return {'script':sha256(__file__), **{m.__name__:sha256(m.__file__) for m in (
            metrics_module,model_module,matching_module,preprocess_module,streaming_module)}}


def summary(rows):
    boundaries = Counter()
    events = Counter()
    for r in rows:
        m = r['metric']
        boundaries.update({k:m['boundary']['0.5'][k] for k in ('tp','fp','fn','valid_duration_s','unscored_unknown')})
        events.update({k:m['events']['0.5'][k] for k in ('tp','fp','fn')})
    start_hits = sum(r['start_hits'] for r in rows)
    start_total = sum(r['start_total'] for r in rows)
    end_hits = sum(r['end_hits'] for r in rows)
    end_total = sum(r['end_total'] for r in rows)
    delays = [x['delay_s'] for r in rows for x in r['metric']['boundary']['0.5']['matches']]

    def f1(c):
        denom = 2*c['tp']+c['fp']+c['fn']
        return 2*c['tp']/denom if denom else 0.

    return dict(boundary_f1=f1(boundaries), boundary_recall=boundaries['tp']/(boundaries['tp']+boundaries['fn']),
                boundary_tp=boundaries['tp'], boundary_fp=boundaries['fp'], boundary_fn=boundaries['fn'],
                start_hits=start_hits,start_total=start_total,start_recall=start_hits/start_total,
                end_hits=end_hits,end_total=end_total,end_recall=end_hits/end_total,
                fp_per_min=60*boundaries['fp']/boundaries['valid_duration_s'],
                extra_fires=sum(r['metric']['oversegmentation']['extra_fires_inside_instances'] for r in rows),
                affected_instances=sum(r['metric']['oversegmentation']['affected_instances'] for r in rows),
                events_total=events['tp']+events['fn'],event_f1=f1(events),
                signed_delay_s=float(np.mean(delays)) if delays else None,
                abs_delay_s=float(np.mean(np.abs(delays))) if delays else None,
                p90_abs_delay_s=float(np.quantile(np.abs(delays),.9)) if delays else None,
                unknown_fires=boundaries['unscored_unknown'])


def evaluate(records, saved, device, output, reference=None):
    config,action,local,state = modules(saved,device)
    model_rows = {arm:[] for arm in POINTS}
    confusion = np.zeros((len(CLASSES),len(CLASSES)),dtype=np.int64)
    for record in records:
        file = record['file']
        stem = Path(file).stem
        x = torch.as_tensor(np.array(record['x'],copy=True),device=device,dtype=torch.float32)
        for arm, parameters in POINTS.items():
            head = local if arm=='B2' else state
            result = rollout(x,action,head,config,**parameters)
            trace = result['trace']
            metric = evaluate_rollout(result,record,period=config.frame_period_s)
            if reference is not None and arm in ('B2','B3-A'):
                source = 'B2' if arm=='B2' else 'B3'
                with np.load(reference/source/(stem+'-trace.npz')) as original:
                    np.testing.assert_array_equal(trace['fire'],original['fire'])
                original = read(reference/source/(stem+'-metrics.json'))
                assert metric['boundary']==original['boundary'] and metric['events']==original['events']
            truth = {m['truth'] for m in metric['boundary']['0.5']['matches']}
            endpoint_counts = {}
            for field in ('start','stop'):
                endpoints = {i[field] for i in record['instances'] if 0<i[field]<len(x)}
                label = 'start' if field=='start' else 'end'
                endpoint_counts.update({label+'_hits':len(endpoints & truth),label+'_total':len(endpoints)})
            model_rows[arm].append(dict(file=file,subject=record['subject'],metric=metric,**endpoint_counts))
            dest = output/arm
            dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/(stem+'-trace.npz'),**trace)
            write_json(dest/(stem+'-events.json'),result['events'])
            write_json(dest/(stem+'-metrics.json'),metric)
            if arm=='B2' and reference is not None:
                # This diagnostic uses GT intervals only for probability pooling.
                p = trace['probabilities']
                for instance in record['instances']:
                    cls = int(p[instance['start']:instance['stop']].mean(0).argmax())
                    confusion[instance['label'],cls] += 1
            print(f'{arm}: scored {file}',flush=True)
    aggregate = {'pooled':{arm:summary(rows) for arm,rows in model_rows.items()},
                 'by_record':{r['file']:{arm:summary([v for v in rows if v['file']==r['file']])
                                        for arm,rows in model_rows.items()} for r in records},
                 'by_subject':{subject:{arm:summary([v for v in rows if v['subject']==subject])
                                        for arm,rows in model_rows.items()}
                               for subject in sorted({r['subject'] for r in records})}}
    return aggregate,confusion


def action_report(confusion, output):
    rows = []
    for k,label in enumerate(CLASSES):
        tp,support,predicted = int(confusion[k,k]),int(confusion[k].sum()),int(confusion[:,k].sum())
        errors = sorted([(CLASSES[j],int(confusion[k,j])) for j in range(len(CLASSES)) if j!=k and confusion[k,j]],
                        key=lambda pair:(-pair[1],pair[0]))
        rows.append(dict(action=label,support=support,correct=tp,precision=tp/predicted if predicted else 0.,
                         recall=tp/support if support else 0.,f1=2*tp/(support+predicted) if support+predicted else 0.,
                         errors=errors))
    with (output/'oracle-confusion.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['truth / predicted',*CLASSES])
        writer.writerows([label,*map(int,confusion[i])] for i,label in enumerate(CLASSES))
    report = dict(classes=CLASSES,confusion=confusion.tolist(),total=int(confusion.sum()),
                  correct=int(np.trace(confusion)),accuracy=float(np.trace(confusion)/confusion.sum()),
                  macro_f1=float(np.mean([r['f1'] for r in rows])),per_class=rows)
    write_json(output/'oracle-action.json',report)
    return report


def encoder(locks):
    path = locked_path(locks['radar_checkpoint'])
    model = SegmentClassifier('rd').eval()
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model'])
    with np.load(locked_path(locks['radar_normalizers'])) as n:
        mean,scale = n['rd_mean'].copy(),n['rd_scale'].copy()
    return FrozenRDStream(model).eval(),mean,scale


def encode_chunk(frontend, rd, mean, scale, carry):
    batch = torch.from_numpy(np.clip((rd-mean)/scale,-8,8))[None]
    result = frontend.forward_chunk(batch,state=carry)
    return result['hidden'][0].cpu().numpy(),result['state']


def feature_preflight(record, locks, raw):
    import h5py
    frontend,mean,scale = encoder(locks)
    with h5py.File(raw/record['file'],'r') as f:
        rd = np.stack([frame_features(f['radar'][i])[1] for i in range(64)])
    result,_ = encode_chunk(frontend,rd,mean,scale,None)
    gap = float(np.abs(result-record['x'][:64]).max())
    if gap > 1e-5:
        raise ValueError(f'feature exporter disagrees with audited Subject 7 cache: {gap}')
    return gap


def export_holdout(file, meta, raw, dest, locks):
    import hashlib
    import h5py
    torch.set_num_threads(4)
    path = raw/file
    before = path.stat()
    if before.st_size!=meta['bytes']:
        raise ValueError(f'unexpected raw file size: {file}')
    frontend,mean,scale = encoder(locks)
    frames = meta['frames']
    output = dest/(Path(file).stem+'-continuous-hidden.npy')
    x = np.lib.format.open_memmap(output,mode='w+',dtype=np.float32,shape=(frames,128))
    carry = None
    digest = hashlib.sha256()
    with h5py.File(path,'r') as hdf:
        data = hdf['radar']
        if data.shape!=(frames,256,256) or data.dtype!=np.dtype('float32'):
            raise ValueError(f'unexpected radar shape or dtype: {file}')
        for start in range(0,frames,64):
            stop = min(frames,start+64)
            raw_frames = data[start:stop]
            digest.update(raw_frames.tobytes())
            rd = np.stack([frame_features(frame)[1] for frame in raw_frames])
            x[start:stop],carry = encode_chunk(frontend,rd,mean,scale,carry)
            if start and start%2048==0:
                print(f'{file}: encoded {stop}/{frames}',flush=True)
    x.flush()
    if not np.isfinite(x).all():
        raise ValueError(f'nonfinite hidden features: {file}')
    after = path.stat()
    if (after.st_size,after.st_mtime_ns)!=(before.st_size,before.st_mtime_ns):
        raise ValueError(f'raw file changed during feature export: {file}')
    return dict(file=file,frames=frames,raw_sha256=digest.hexdigest(),feature_path=str(output),feature_sha256=sha256(output))


def locked_protocol(args, saved, config_path):
    assert saved['calibration']['B2']==POINTS['B2'] and saved['calibration']['B3']==POINTS['B3-A']
    calibration = read(args.fine)
    model_digest = sha256(args.run/'model.pt')
    assert calibration['checkpoint_sha256']==model_digest and calibration['validation_rollouts']==0
    assert calibration['b_at_recall_90']['tau']==1. and calibration['b_at_recall_90']['threshold']==.895
    return dict(checkpoint_sha256=model_digest,settings_sha256=sha256(config_path),
                parameter_selection_sha256=sha256(args.fine),points=POINTS,code_sha256=source_digests(),
                boundary_tolerance_s=.5,event_iou=.5,optimizer_steps=0,teacher_boundary_resets=False,
                test_subjects=['subject8','subject9'],raw_holdout=HOLDOUT,activities_sha256=ACTIVITIES_SHA256,
                threshold_adjusted_on_test=False,checkpoint_selected_on_test=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('confirm','final'))
    for name in ('run','config','fine','raw','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--confirmation',type=Path)
    parser.add_argument('--device',default='cuda')
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(4)
    settings = read(args.config)
    records = load_records(settings)
    saved = load_checkpoint(args.run/'model.pt','cpu')
    config,_,_,_ = modules(saved,'cpu')
    assert record_identity(records,config)==saved['protocol']['record_identity']
    assert {r['file']:r['provenance']['continuous_hidden']['sha256'] for r in records}==saved['protocol']['source_hashes']
    plan = locked_protocol(args,saved,args.config)
    locks = {k:settings['locks'][k] for k in ('radar_checkpoint','radar_normalizers')}
    if args.stage=='confirm':
        write_json(args.output/'locked-protocol.json',plan)
        subject7 = [r for r in records if r['role']=='validation']
        assert [r['file'] for r in subject7]==saved['protocol']['validation_files'] and len(subject7)==4
        preflight = feature_preflight(subject7[0],locks,args.raw)
        aggregate,confusion = evaluate(subject7,saved,args.device,args.output,reference=args.run)
        oracle = action_report(confusion,args.output)
        a,b = (aggregate['pooled'][arm] for arm in ('B3-A','B3-B'))
        assert (oracle['correct'],oracle['total'])==(384,511)
        # A test-stage gate for integrity and advertised operating-mode behavior, not for B3 superiority.
        gate = b['boundary_recall']>a['boundary_recall'] and b['end_recall']>a['end_recall']
        result = dict(stage='subject7_confirmation',**aggregate,oracle=oracle,frontend_prefix_max_abs=preflight,
                      test_signal_values_read=False,original_replays_verified=True,gate_passed=gate)
    else:
        if args.confirmation is None:
            raise ValueError('final test requires the completed confirmation directory')
        prior = read(args.confirmation/'summary.json')
        if not prior['gate_passed'] or not prior['original_replays_verified'] or prior['test_signal_values_read']:
            raise ValueError('confirmation integrity or operating-mode gate failed')
        if plan != read(args.confirmation/'locked-protocol.json'):
            raise ValueError('test protocol differs from the pre-test lock')
        write_json(args.output/'locked-protocol.json',plan)
        activity = args.raw/'labels'/'activities.csv'
        if sha256(activity)!=ACTIVITIES_SHA256:
            raise ValueError('activities.csv checksum mismatch')
        lengths = {r['file']:r['frames'] for r in records} | {k:v['frames'] for k,v in HOLDOUT.items()}
        instances = read_instances(activity,lengths)
        for file,meta in HOLDOUT.items():
            assert sum(r['file']==file for r in instances)==meta['instances']
        dest = args.output/'frozen-features'
        dest.mkdir()
        with ProcessPoolExecutor(max_workers=2,mp_context=get_context('spawn')) as pool:
            pending = [pool.submit(export_holdout,file,meta,args.raw,dest,locks) for file,meta in HOLDOUT.items()]
            provenance = [job.result() for job in pending]
        write_json(args.output/'feature-provenance.json',provenance)
        holdout = [dict(file=r['file'],subject=r['file'].split('_')[0],frames=r['frames'],
                        x=np.load(r['feature_path'],mmap_mode='r'),instances=[i for i in instances if i['file']==r['file']])
                   for r in provenance]
        aggregate,_ = evaluate(holdout,saved,args.device,args.output)
        result = dict(stage='untouched_subject8_9',**aggregate,test_signal_values_read=True,
                      subject7_gate_verified=True,optimizer_steps=0,teacher_boundary_resets=False)
    assert sha256(args.run/'model.pt')==plan['checkpoint_sha256'] and source_digests()==plan['code_sha256']
    write_json(args.output/'summary.json',result)


if __name__=='__main__':
    main()
