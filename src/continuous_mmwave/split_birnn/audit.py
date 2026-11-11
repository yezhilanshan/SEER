"""Immutable MD-only conversion with per-recording temporal/source audit.

Reads all MD values but never radar cubes. No source clock calibration is inferred
from CSV span or spectrum changes. Hospital selection remains unconfirmed.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path
import numpy as np
from .data import COMMANDS, CLASSES, align_commands, build_folds, standardize_frames


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def read_commands(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows or not {'timestamp','command'}.issubset(rows[0]):
        raise ValueError('command CSV requires timestamp and command')
    stamps = [datetime.fromisoformat(r['timestamp']) for r in rows]
    names = [' '.join(r['command'].split()) for r in rows]
    if any(n not in COMMANDS for n in names):
        raise ValueError('unrecognized command')
    offsets = np.array([(s-stamps[0]).total_seconds() for s in stamps])
    return offsets, np.array([COMMANDS[n] for n in names], dtype=np.int64), names


def source_stat(path):
    s = Path(path).stat()
    return {'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns}


def diagnostic_plot(md, y, offsets, path, seed, *, frame_origin_s=0.):
    import importlib.util
    can_plot = importlib.util.find_spec('matplotlib') is not None
    if can_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    rng = np.random.default_rng(seed)
    candidates = offsets[(offsets > frame_origin_s+10) & (offsets < frame_origin_s+(len(md)-1)*.09-10)]
    if len(candidates) == 0:
        return []
    plots = []
    for j, center in enumerate(rng.choice(candidates, min(2,len(candidates)), replace=False)):
        lo = max(0,int((center-5-frame_origin_s)/.09))
        hi = min(len(md),int((center+5-frame_origin_s)/.09)+1)
        times = frame_origin_s+np.arange(lo,hi)*.09
        dest = path.with_name(path.stem+f'-{j}.npz')
        np.savez_compressed(dest,md=md[lo:hi],labels=y[lo:hi],times=times,
                            command_offsets_s=offsets[(offsets>=times[0]) & (offsets<=times[-1])],
                            frame_origin_s=frame_origin_s)
        if not can_plot:
            plots.append(str(dest))
            continue
        fig, axes = plt.subplots(3,1,figsize=(10,6),sharex=True,constrained_layout=True)
        normalized = standardize_frames(md[lo:hi])
        axes[0].imshow(normalized.T, origin='lower',aspect='auto',extent=[times[0],times[-1],0,128],vmin=-2,vmax=3)
        axes[0].set_ylabel('Doppler bin')
        axes[1].plot(times,md[lo:hi].std(1)); axes[1].set_ylabel('MD bin std\n(source units)')
        axes[2].step(times,y[lo:hi],where='post'); axes[2].set_ylabel('Merged label'); axes[2].set_yticks(range(9))
        axes[2].set_xlabel('Approximate seconds from first command (not a calibrated radar clock)')
        for t in offsets[(offsets>=times[0]) & (offsets<=times[-1])]:
            for ax in axes:
                ax.axvline(t,color='red',ls='--',alpha=.6)
        fig.suptitle(path.stem+' | command boundaries, not physical action onset')
        dest = path.with_name(path.stem+f'-{j}.png'); fig.savefig(dest,dpi=130); plt.close(fig)
        plots.append(str(dest))
    return plots


def prepare_data(raw: Path, output: Path, *, make_plots: bool = True, origins: dict | None = None):
    import h5py  # server native dependency; unrelated local tests need not import it
    raw, output = Path(raw).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output/'recordings').mkdir(); (output/'plots').mkdir()
    write_json(output/'status.json', {'state':'running','formal_training_ready':False})
    sources = sorted(raw.glob('subject_*/captured_data/set*/processed_data.h5'))
    if not sources:
        raise ValueError('no final HDF5 recordings found')
    records, sessions, pairs, plots = [], [], [], []
    for h5 in sources:
        subject, session = h5.parents[2].name, h5.parent.name
        csvpath = h5.with_name('timestamp_speech.csv')
        before = {'h5':source_stat(h5), 'csv':source_stat(csvpath)}
        offsets, ids, names = read_commands(csvpath)
        csvsha = hashlib.sha256(csvpath.read_bytes()).hexdigest()
        session_rows = []
        with h5py.File(h5,'r') as handle:
            for sensor, g in sorted(handle['radar'].items()):
                if 'mDoppler' not in g:
                    continue
                md = np.asarray(g['mDoppler'][:])  # preserve native values/dtype before hashing
                if md.ndim != 2 or md.shape[1] != 128 or len(md)==0 or not np.isfinite(md).all():
                    raise ValueError(f'bad MD array: {h5}, {sensor}')
                identity = f'{subject}-{session}-{sensor}'
                origin = float((origins or {}).get(identity,0.))
                labels, timing = align_commands(len(md),offsets,ids,frame_origin_s=origin)
                dest = output/'recordings'/(identity+'.npz')
                times = origin + np.arange(len(md),dtype=np.float64)*.09
                np.savez_compressed(dest,md=md,labels=labels,timestamps=times)
                digest = hashlib.sha256(md.tobytes()).hexdigest()
                with np.load(dest) as saved:
                    if (saved['md'].shape != md.shape or hashlib.sha256(saved['md'].tobytes()).hexdigest()!=digest
                            or not np.array_equal(saved['labels'],labels)):
                        raise ValueError('MD-only saved values do not match source')
                row = {'subject_id':subject,'session_id':session,'sensor_id':sensor,'recording_id':identity,
                       'path':str(dest.relative_to(output)), 'source_h5':str(h5),'source_commands':str(csvpath),
                       'source_csv_sha256':csvsha,'md_values_sha256':digest,
                       'source_md_dtype':str(md.dtype),
                       'npz_sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),
                       'environment_id':'unconfirmed','hospital_status':'unconfirmed','frame_period_s':.09,
                       'frame_count':len(md),'valid_frame_count':timing['valid_frame_count'],
                       'transition_count':timing['transition_count'],'alignment':timing,
                       'h5_source_stat':before['h5'],'h5_hash_scope':'MD_values_only_not_full_cube_file',
                       'h5_metadata':{'root_attrs_present':bool(handle.attrs),'sensor_attrs_present':bool(g.attrs)}}
                records.append(row); session_rows.append(row)
                if make_plots and subject in ('subject_01','subject_02') and session == 'set000':
                    plots.extend(diagnostic_plot(md,labels,offsets,output/'plots'/(identity+'.png'),42,
                                                 frame_origin_s=origin))
        if not session_rows:
            raise ValueError(f'final session has no valid MD recording: {h5}')
        after = {'h5':source_stat(h5),'csv':source_stat(csvpath)}
        if before != after:
            raise ValueError(f'source changed during conversion: {h5}')
        sessions.append({'subject_id':subject,'session_id':session,'command_count':len(names),
                         'csv_span_s':float(offsets[-1]),'sensor_count':len(session_rows),'csv_sha256':csvsha})
        if len(session_rows)==2:
            a,b = session_rows
            ai,bi = np.array(a['alignment']['command_frame_indices']),np.array(b['alignment']['command_frame_indices'])
            pairs.append({'subject_id':subject,'session_id':session,
                          'sensor_frame_count_difference':a['frame_count']-b['frame_count'],
                          'same_command_indices_under_assumed_clock':bool(np.array_equal(ai,bi)),
                          'clock_calibration_proven':False})
        print(json.dumps({'session':f'{subject}/{session}','done':len(sessions),'total':len(sources)}),flush=True)
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    write_json(output/'label_map.json', {'classes':[{'id':i,'name':n} for i,n in enumerate(CLASSES)],'commands':COMMANDS})
    if len({r['subject_id'] for r in records})>=5:
        folds = build_folds(records)
        write_json(output/'candidate_folds.json', {'status':'candidate_not_formal_Hospital_split','folds':folds})
        for f in folds:
            d=output/'candidate_folds'/f"fold{f['fold']}"; d.mkdir(parents=True)
            for role in ('train','validation','test'):
                selected=set(f[role+'_recordings'])
                (d/(role+'.jsonl')).write_text(''.join(json.dumps({**r,'path':str((output/r['path']).resolve())})+'\n'
                                                       for r in records if r['recording_id'] in selected),encoding='utf-8')
    distribution={}
    for key in ('subject_id','sensor_id'):
        distribution[key]={}
        for value in sorted({r[key] for r in records}):
            group=[r for r in records if r[key]==value]
            distribution[key][value]={'recordings':len(group),'frames':sum(r['frame_count'] for r in group),
                                     'class_frame_counts':np.sum([r['alignment']['class_frame_counts'] for r in group],0).tolist(),
                                     'class_run_counts':np.sum([r['alignment']['class_run_counts'] for r in group],0).tolist()}
    result={'session_count':len(sessions),'subject_count':len({r['subject_id'] for r in records}),
            'recording_count':len(records),'sensor_frames':sum(r['frame_count'] for r in records),
            'command_rows':sum(s['command_count'] for s in sessions),'hospital_status':'unconfirmed',
            'formal_training_ready':False,'clock_calibrated':False,'physical_action_onset_available':False,
            'source_values_and_saved_values_equal':True, 'paired_sensor_audit':pairs,
            'class_frame_counts':np.sum([r['alignment']['class_frame_counts'] for r in records],0).tolist(),
            'class_run_counts':np.sum([r['alignment']['class_run_counts'] for r in records],0).tolist(),
            'ignored_frames':sum(r['alignment']['ignored_frame_count'] for r in records),
            'commands_at_or_after_end':sum(r['alignment']['commands_at_or_after_end'] for r in records),
            'zero_transition_recordings':[r['recording_id'] for r in records if not r['transition_count']],
            'distribution':distribution,'sessions':sessions,'diagnostic_plots':plots,
            'unknowns':['Hospital mapping','radar first-frame absolute timestamp','actual frame clocks','physical action onset'],
            'tail_policy':'last command extrapolated to last radar frame; validity not established by CSV'}
    write_json(output/'audit.json',result)
    write_json(output/'status.json', {'state':'completed_approximate_protocol','formal_training_ready':False})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-plots',action='store_true'); p.add_argument('--origins',type=Path)
    args=p.parse_args()
    origins=json.loads(args.origins.read_text()) if args.origins else None
    result=prepare_data(args.raw,args.output,make_plots=not args.no_plots,origins=origins)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sessions','paired_sensor_audit','distribution')},ensure_ascii=False))


if __name__=='__main__':
    main()
