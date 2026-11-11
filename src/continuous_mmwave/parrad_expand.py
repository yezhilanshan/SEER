"""Snapshot complete downloads, reuse canonical data and convert new PARrad sessions."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def file_stamp(path):
    s=Path(path).stat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev)


def unchanged(path,before):
    if file_stamp(path)!=before:raise ValueError(f'source changed during conversion: {path}')


def new_root(path):Path(path).mkdir(parents=True,exist_ok=False)


def dump(path,value):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(path)


def snapshot(raw,expected_sizes):
    raw=Path(raw);ready=[];excluded=[]
    for key,size in sorted(expected_sizes.items()):
        subject,session=key.split('/');d=raw/subject/'captured_data'/session
        h5,csv=d/'processed_data.h5',d/'timestamp_speech.csv';reason=None
        if any(d.glob('*.part')):reason='partial_download'
        elif not h5.is_file() or not csv.is_file():reason='missing_files'
        elif h5.stat().st_size!=size:reason='h5_size_mismatch'
        elif not all(p.with_name(p.name+'.sha256').is_file() for p in (h5,csv)):reason='missing_completion_digest'
        if reason:excluded.append(dict(key=key,reason=reason));continue
        digests={p.name:p.with_name(p.name+'.sha256').read_text().split()[0] for p in (h5,csv)}
        if not all(re.fullmatch('[0-9a-f]{64}',v) for v in digests.values()):reason='invalid_completion_digest'
        elif sha(csv)!=digests[csv.name]:reason='csv_sha256_mismatch'
        if reason:excluded.append(dict(key=key,reason=reason));continue
        ready.append(dict(key=key,subject_id=subject,session_id=session,h5=str(h5.resolve()),commands=str(csv.resolve()),
            h5_stamp=file_stamp(h5),csv_stamp=file_stamp(csv),h5_sha256=digests[h5.name],csv_sha256=digests[csv.name],
            h5_sha256_scope='download_sidecar_only_not_rehashed',expected_h5_bytes=size))
    return dict(ready=ready,excluded=excluded,policy='fixed complete-file snapshot; no .part; exact probed byte count; CSV SHA verified; H5 sidecar recorded, RD/MD verified separately')


def rebase_row(row,source_manifest,destination_manifest):
    result=dict(row);p=Path(row['path']);p=p if p.is_absolute() else Path(source_manifest).parent/p
    result['path']=Path(os.path.relpath(p,Path(destination_manifest).parent)).as_posix()
    result['recording_id']=row.get('recording_id',str(row['path']).replace('\\','/'))
    return result


def write_manifest(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows),encoding='utf-8')


def write_views(root,rows):
    root=Path(root);master=root/'manifest.jsonl';write_manifest(master,rows)
    for subject in sorted({r['subject_id'] for r in rows}):
        path=root/'subjects'/f'{subject}.jsonl';write_manifest(path,[rebase_row(r,master,path) for r in rows if r['subject_id']==subject])
    # Compatibility view only; not a newly reserved untouched research test split.
    if {'subject_05','subject_06'}.issubset({r['subject_id'] for r in rows}):
        for split in ('train','validation','test'):
            path=root/'splits/legacy_05_06'/f'{split}.jsonl'
            selected=[r for r in rows if ('validation' if r['subject_id']=='subject_05' else 'test' if r['subject_id']=='subject_06' else 'train')==split]
            write_manifest(path,[rebase_row(r,master,path) for r in selected])


def verify_recording(row,manifest,source,label_to_id):
    """Read all saved values; independently reconstruct labels and RD/MD digests."""
    import h5py
    import numpy as np
    from .parrad_convert import read_command_annotations
    p=Path(row['path']);p=p if p.is_absolute() else manifest.parent/p
    annotations=read_command_annotations(source['commands']);origin=annotations[0].timestamp
    offsets=np.array([(r.timestamp-origin).total_seconds() for r in annotations]);ids=np.array([label_to_id[r.command] for r in annotations],dtype=np.int64)
    with np.load(p,allow_pickle=False) as arrays:
        rd,md,labels,times=(arrays[k] for k in ('rd','md','labels','timestamps'));n=len(times)
        assert rd.shape==(n,1,93,128) and md.shape==(n,128) and labels.shape==(n,)
        assert rd.dtype==md.dtype==np.float32 and labels.dtype==np.int64 and times.dtype==np.float64
        assert np.isfinite(rd).all() and np.isfinite(md).all() and np.isfinite(times).all()
        np.testing.assert_array_equal(times,np.arange(n,dtype=np.float64)*.09)
        active=np.searchsorted(offsets,times,side='right')-1;expected=np.full(n,-100,np.int64);valid=active>=0;expected[valid]=ids[active[valid]]
        np.testing.assert_array_equal(labels,expected)
        values={}
        with h5py.File(source['h5'],'r') as h:
            group=h['radar'][row['sensor_id']]
            if group['rdn'].shape!=(n,93,128) or group['mDoppler'].shape!=(n,128):
                raise ValueError(f'raw source shape differs from saved arrays: {p}')
            for key,dataset,saved in [('rd','rdn',rd[:,0]),('md','mDoppler',md)]:
                digest=hashlib.sha256()
                for start in range(0,n,256):digest.update(np.asarray(group[dataset][start:start+256],dtype=np.float32).tobytes())
                values[key]=hashlib.sha256(saved.tobytes()).hexdigest();assert values[key]==digest.hexdigest(),f'raw {key} differs: {p}'
        return dict(path=str(p.resolve()),sha256=sha(p),stamp=file_stamp(p),frames=n,duration_seconds=n*.09,
            class_frames=np.bincount(labels[labels>=0],minlength=len(label_to_id)).tolist(),ignored_frames=int((labels<0).sum()),
            raw_RD_MD_sha256=values,full_arrays_raw_equal=True,last_command_beyond_recording=bool(offsets[-1]>=n*.09))


def main():
    parser=argparse.ArgumentParser()
    for name in ('raw','existing','expected-sizes','phrases','output'):parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args();root=args.output.resolve();new_root(root)
    started=datetime.now().astimezone().isoformat();completed=[]
    def status(state,**more):
        payload=dict(state=state,started=started,completed_sessions=len(completed),**more);dump(root/'status.json',payload);print(json.dumps(payload),flush=True)
    try:
        import numpy as np
        from .parrad_convert import convert_recording,read_command_annotations,_sensor_paths
        from .radar_text.data import OfflineCaptionDataset,load_phrases,caption_from_actions
        from .radar_text.sliding_baselines import complete_window
        import h5py
        status('snapshot');expected=json.loads(args.expected_sizes.read_text());plan=snapshot(args.raw,expected);dump(root/'download-snapshot.json',plan)
        assert plan['ready'],'no complete sessions'
        for p,name in ((args.existing/'label_map.json','label_map.json'),(args.phrases,'action_phrases.json'),(args.expected_sizes,'expected-h5-bytes.json')):shutil.copyfile(p,root/name)
        vocabulary=json.loads((root/'label_map.json').read_text());label_to_id={r['command']:r['id'] for r in vocabulary['classes']}
        old_manifest=args.existing/'manifest.jsonl';old_rows=[json.loads(line) for line in old_manifest.read_text().splitlines() if line.strip()]
        original_metadata={str(p.resolve()):sha(p) for p in args.existing.glob('*.json*')};old_by_session={}
        for row in old_rows:old_by_session.setdefault((row['subject_id'],row['session_id']),[]).append(row)
        for source in plan['ready']:
            commands=read_command_annotations(source['commands']);unknown=sorted({r.command for r in commands}-label_to_id.keys())
            if unknown:raise ValueError(f'unknown commands in {source["key"]}: {unknown}')
            with h5py.File(source['h5'],'r') as h:source['sensors']=[s[0] for s in _sensor_paths(h)]
        dump(root/'conversion-plan.json',plan);master=root/'manifest.jsonl';rows=[];records=[];reuse=0;new=0
        for source in plan['ready']:
            unchanged(source['h5'],source['h5_stamp']);unchanged(source['commands'],source['csv_stamp'])
            key=(source['subject_id'],source['session_id']);previous=old_by_session.get(key)
            status('convert_or_reuse',session=source['key'],mode='reuse' if previous else 'new')
            if previous:
                assert sorted(r['sensor_id'] for r in previous)==source['sensors']
                rr=[rebase_row(r,old_manifest,master) for r in previous];reuse+=len(rr)
            else:
                target=root/'converted'/source['subject_id']/source['session_id'];target.mkdir(parents=True)
                shutil.copyfile(root/'label_map.json',target/'label_map.json')
                manifest=convert_recording(source['h5'],source['commands'],target,subject_id=source['subject_id'],session_id=source['session_id'],environment_id='parrad',frame_period_s=.09,md_only=False)
                rr=[]
                for line in manifest.read_text().splitlines():
                    row=json.loads(line);row['recording_id']='recordings/'+Path(row['path']).name;rr.append(rebase_row(row,manifest,master))
                new+=len(rr)
            for row in rr:
                if previous:
                    assert Path(row['source_h5']).resolve()==Path(source['h5']).resolve() and Path(row['source_commands']).resolve()==Path(source['commands']).resolve()
                row['recording_start_rule']='first_command_as_legacy_converter';row['source_h5_sha256_scope']=source['h5_sha256_scope']
                record=verify_recording(row,master,source,label_to_id);record.update(subject_id=row['subject_id'],session_id=row['session_id'],sensor_id=row['sensor_id'],reused=bool(previous))
                records.append(record);rows.append(row)
            unchanged(source['h5'],source['h5_stamp']);unchanged(source['commands'],source['csv_stamp'])
            completed.append(source['key']);dump(root/'recording-audit.json',records)
        write_views(root,rows);status('index_caption_windows',recordings=len(rows))
        data=OfflineCaptionDataset(master,load_phrases(root/'action_phrases.json'));window_counts=Counter();index_path=root/'caption-windows-12s-stride6.jsonl'
        with index_path.open('w',encoding='utf-8') as f:
            for rid,a,b,sid,actions in data.windows:
                meta=data.recordings.rows[rid];n=b-a;window_counts[meta['subject_id']]+=1
                f.write(json.dumps(dict(sample_id=sid,recording_id=meta['recording_id'],subject_id=meta['subject_id'],session_id=meta['session_id'],sensor_id=meta['sensor_id'],
                    start_frame=a,stop_frame=b,actions=actions,caption=caption_from_actions(actions,data.phrases),full_window=complete_window(n*.09,.09)),ensure_ascii=False)+'\n')
        # Load real model-ready batches, independent of the conversion audit.
        from .radar_text.data import collate_windows
        batch=collate_windows([data[0],data[-1]]);assert batch['rd'].shape[2:]==(1,93,128) and batch['md'].shape[-1]==128
        for source in plan['ready']:
            unchanged(source['h5'],source['h5_stamp']);unchanged(source['commands'],source['csv_stamp']);assert sha(source['commands'])==source['csv_sha256']
        assert all(sha(Path(p))==digest for p,digest in original_metadata.items())
        assert all(file_stamp(r['path'])==r['stamp'] for r in records)
        assert all(sha(r['path'])==r['sha256'] for r in records if r['reused'])
        subjects=sorted({r['subject_id'] for r in rows});by_subject=[]
        for subject in subjects:
            available=[s for s in completed if s.startswith(subject+'/')];expected_sessions=[s for s in expected if s.startswith(subject+'/')]
            rr=[r for r in records if r['subject_id']==subject]
            by_subject.append(dict(subject_id=subject,ready_sessions=len(available),expected_sessions=len(expected_sessions),all_sessions_complete=len(available)==len(expected_sessions),
                recordings=len(rr),caption_windows=window_counts[subject],sensor_frames=sum(r['frames'] for r in rr),sensor_hours=sum(r['duration_seconds'] for r in rr)/3600))
        unique_seconds=sum(max(r['duration_seconds'] for r in records if r['subject_id']==s['subject_id'] and r['session_id']==s['session_id']) for s in plan['ready'])
        summary=dict(subjects=len(subjects),sessions=len(completed),recordings=len(rows),reused_recordings=reuse,new_recordings=new,
            sensor_frames=sum(r['frames'] for r in records),sensor_hours=sum(r['duration_seconds'] for r in records)/3600,session_hours=unique_seconds/3600,
            class_frames=np.sum([r['class_frames'] for r in records],axis=0).tolist(),caption_windows=len(data),window_skipped=data.skipped,
            by_subject=by_subject,legacy_split='compatibility only: all other ready subjects train,05validation,06test; test06 previously observed',
            environment='parrad; source files do not establish Homelab/Hospital',label_policy='command_active_until_next_command; .09s generated timestamps; first command origin; no true motion boundary claim',
            verified='every RD/MD element digest equal to raw datasets; labels independently rebuilt; all real arrays finite; source stamps stable; old38 data SHA unchanged',
            h5_checksum_limit='large cube-inclusive H5 file hashes recorded from completed-download sidecars, not independently rehashed; probed size and full required RD/MD values checked',
            batch_shapes={k:list(batch[k].shape) for k in ('rd','md','lengths','labels')},training_started=False)
        dump(root/'summary.json',summary)
        artifacts={p.relative_to(root).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p)) for p in root.rglob('*') if p.is_file() and p.name!='status.json'}
        dump(root/'artifact-manifest.json',artifacts);status('complete',finished=datetime.now().astimezone().isoformat(),subjects=len(subjects),sessions=len(completed),recordings=len(rows),new_recordings=new)
    except BaseException as error:status('failed',error=str(error));raise


if __name__=='__main__':main()
