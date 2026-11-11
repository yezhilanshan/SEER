"""Independent artifact, legacy-data, window-index and subject-view readback."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import h5py

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser()
    for k in ('data','code','previous-provenance','destination'):p.add_argument('--'+k,required=True,type=Path)
    args=p.parse_args();root=args.data;load=lambda p:json.loads(p.read_text());read=lambda p:[json.loads(s) for s in p.read_text().splitlines() if s.strip()]
    assert load(root/'status.json')['state']=='complete'
    summary=load(root/'summary.json');audit=load(root/'recording-audit.json');rows=read(root/'manifest.jsonl')
    manifest=load(root/'artifact-manifest.json');hashes={}
    for name,entry in manifest.items():
        path=root/name;assert path.stat().st_size==entry['bytes'];actual=sha(path);assert actual==entry['sha256'];hashes[str(path.resolve())]=actual
    previous=load(args.previous_provenance)['original_input_sha256'];legacy=0
    for record in audit:
        actual=hashes.get(record['path']) or sha(Path(record['path']));assert actual==record['sha256']
        if record['reused']:assert actual==previous[record['path']];legacy+=1
    assert legacy==summary['reused_recordings']
    sources=load(args.code/'source-manifest.json');assert all(sha(args.code/name)==digest for name,digest in sources.items())
    assert len(rows)==summary['recordings']==len(audit)
    assert len({(r['subject_id'],r['session_id'],r['sensor_id']) for r in rows})==len(rows)
    paths={};class_frames=np.zeros(14,np.int64);sensor_frames=0;expected_windows=[];subjects=Counter();full=0
    source_by_session={s['key']:s for s in load(root/'conversion-plan.json')['ready']}
    phrases={int(k):v for k,v in load(root/'action_phrases.json').items()}
    for row in rows:
        path=(root/row['path']).resolve();assert path.is_file();paths[(row['subject_id'],row['session_id'],row['sensor_id'])]=path
        with np.load(path,allow_pickle=False) as arrays:labels=arrays['labels'];times=arrays['timestamps']
        n=len(labels);sensor_frames+=n;assert len(times)==n and np.array_equal(times,np.arange(n)*.09)
        source=source_by_session[row['subject_id']+'/'+row['session_id']]
        with h5py.File(source['h5'],'r') as raw:
            group=raw['radar'][row['sensor_id']]
            assert group['rdn'].shape==(n,93,128) and group['mDoppler'].shape==(n,128),f'raw source shape differs: {path}'
        assert set(np.unique(labels))<=set(range(14))|{-100};class_frames+=np.bincount(labels[labels>=0],minlength=14)
        identity=hashlib.sha256(row['recording_id'].encode()).hexdigest()[:12]
        for begin in np.arange(times[0],times[-1]+.09-1e-7,6.):
            a=int(np.searchsorted(times,begin-1e-7));b=int(np.searchsorted(times,begin+12.-1e-7))
            if b<=a or times[b-1]-times[a]+.09<8.-1e-6 or np.any(labels[a:b]<0):continue
            seq=[]
            for label in labels[a:b].tolist():
                if not seq or seq[-1]!=label:seq.append(label)
            if len(seq)>3:continue
            sid=f'{row["subject_id"]}/{row["session_id"]}/{row["sensor_id"]}/{identity}:{a}:{b}'
            words=[phrases[k].strip().rstrip('.') for k in seq]
            body=words[0] if len(words)==1 else ', then '.join(words) if len(words)==2 else ', then '.join(words[:-1])+', and then '+words[-1]
            complete=(b-a)*.09+.09>=12.-1e-6
            expected_windows.append((sid,a,b,seq,'The person '+body+'.',complete));subjects[row['subject_id']]+=1;full+=complete
    index=read(root/'caption-windows-12s-stride6.jsonl');assert len(index)==len(expected_windows)==summary['caption_windows']
    assert len({r['sample_id'] for r in index})==len(index)
    for row,expected in zip(index,expected_windows):assert (row['sample_id'],row['start_frame'],row['stop_frame'],row['actions'],row['caption'],row['full_window'])==expected
    assert sensor_frames==summary['sensor_frames'] and class_frames.tolist()==summary['class_frames']
    for source in source_by_session.values():
        for field,stamp in (('h5','h5_stamp'),('commands','csv_stamp')):
            st=Path(source[field]).stat();actual=dict(bytes=st.st_size,mtime_ns=st.st_mtime_ns,inode=st.st_ino,device=st.st_dev)
            assert actual==source[stamp],f'raw source changed: {source[field]}'
        assert sha(source['commands'])==source['csv_sha256']
    for subject in sorted(subjects):
        subset=read(root/'subjects'/f'{subject}.jsonl');original=[r for r in rows if r['subject_id']==subject]
        assert len(subset)==len(original)
        for r,o in zip(subset,original):assert (root/'subjects'/r['path']).resolve()==(root/o['path']).resolve() and r['recording_id']==o['recording_id']
    split_subjects={};split_total=0
    for name in ('train','validation','test'):
        sub=root/'splits/legacy_05_06';rr=read(sub/f'{name}.jsonl');split_subjects[name]=sorted({r['subject_id'] for r in rr});split_total+=len(rr)
        for r in rr:assert (sub/r['path']).resolve()==paths[(r['subject_id'],r['session_id'],r['sensor_id'])]
    assert split_total==len(rows)
    assert not any(set(split_subjects[a])&set(split_subjects[b]) for a,b in (('train','validation'),('train','test'),('validation','test')))
    # References from historical P1 must survive rebasing and expansion exactly.
    old_root=args.previous_provenance.parent
    for split,subject in (('validation','subject_05'),('test','subject_06')):
        old=load(old_root/f'windows-{split}.json');current={r['sample_id']:r for r in index if r['subject_id']==subject}
        assert set(current)=={r['sample_id'] for r in old}
        for r in old:assert current[r['sample_id']]['actions']==r['actions'] and current[r['sample_id']]['full_window']==r['full_window']
    result=dict(passed=True,artifact_files_verified=len(manifest),source_files_verified=len(sources),legacy_SHA_matches_historical=legacy,
        sensor_recordings=len(rows),raw_source_shapes_checked=len(rows),raw_session_stamps_checked=len(source_by_session),windows_independently_rebuilt=len(index),strict_full_windows=int(full),legacy_validation_test_IDs_preserved=True,
        split_subjects=split_subjects,training_started=False,H5_cube_file_SHA_independently_rehashed=False)
    args.destination.write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
