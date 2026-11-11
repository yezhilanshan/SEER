import hashlib
import json
from pathlib import Path
import pytest


def session(root,subject='subject_01',name='set000',partial=False):
    d=root/subject/'captured_data'/name;d.mkdir(parents=True)
    for filename,content in [('processed_data.h5',b'test-hdf'),('timestamp_speech.csv',b'timestamp,command\n2019-01-01T00:00:00,walk to bed\n')]:
        p=d/(filename+'.part' if partial and filename.endswith('.h5') else filename);p.write_bytes(content)
        if not p.name.endswith('.part'):p.with_name(p.name+'.sha256').write_text(hashlib.sha256(content).hexdigest()+'  '+filename+'\n')
    return d


def test_snapshot_excludes_partial_missing_and_wrong_length(tmp_path):
    from continuous_mmwave.parrad_expand import snapshot
    session(tmp_path);session(tmp_path,'subject_02',partial=True);session(tmp_path,'subject_03')
    plan=snapshot(tmp_path,{'subject_01/set000':8,'subject_02/set000':8,'subject_03/set000':99,'subject_04/set000':8})
    assert [r['key'] for r in plan['ready']]==['subject_01/set000']
    reasons={r['key']:r['reason'] for r in plan['excluded']}
    assert reasons['subject_02/set000']=='partial_download' and reasons['subject_03/set000']=='h5_size_mismatch' and reasons['subject_04/set000']=='missing_files'
    assert plan['ready'][0]['h5_sha256_scope']=='download_sidecar_only_not_rehashed'


def test_snapshot_rejects_bad_csv_digest_and_missing_completion_digest(tmp_path):
    from continuous_mmwave.parrad_expand import snapshot
    a=session(tmp_path);b=session(tmp_path,'subject_02')
    (a/'timestamp_speech.csv').write_text('changed')
    (b/'processed_data.h5.sha256').unlink()
    plan=snapshot(tmp_path,{'subject_01/set000':8,'subject_02/set000':8})
    assert not plan['ready']
    assert {x['reason'] for x in plan['excluded']}=={'csv_sha256_mismatch','missing_completion_digest'}


def test_manifest_rebase_keeps_old_window_identity_and_paths(tmp_path):
    from continuous_mmwave.parrad_expand import rebase_row
    old=tmp_path/'old';new=tmp_path/'new';old.mkdir();new.mkdir()
    row=dict(path='recordings/subject_01-set000-radar.npz',subject_id='subject_01',session_id='set000',sensor_id='radar',environment_id='parrad')
    result=rebase_row(row,old/'manifest.jsonl',new/'manifest.jsonl')
    assert result['recording_id']==row['path'] and result['path']=='../old/'+row['path'] and 'recording_id' not in row
    row['recording_id']='explicit-original-id';assert rebase_row(row,old/'manifest.jsonl',new/'manifest.jsonl')['recording_id']=='explicit-original-id'


def test_stat_change_fails_closed_and_output_never_overwritten(tmp_path):
    from continuous_mmwave.parrad_expand import file_stamp,unchanged,new_root
    p=tmp_path/'source';p.write_bytes(b'abc');before=file_stamp(p);p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):unchanged(p,before)
    out=tmp_path/'output';out.mkdir();(out/'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):new_root(out)
    assert list(out.iterdir())==[out/'sentinel']


def test_subject_views_partition_without_sensor_or_session_leakage(tmp_path):
    from continuous_mmwave.parrad_expand import write_views
    rows=[dict(path=f'recordings/{s}-{n}.npz',subject_id=s,session_id='set000',sensor_id=n,environment_id='parrad')
          for s in ('subject_01','subject_05','subject_06') for n in ('14','68')]
    write_views(tmp_path,rows)
    read=lambda p:[json.loads(s) for s in p.read_text().splitlines()]
    groups={name:read(tmp_path/'splits/legacy_05_06'/f'{name}.jsonl') for name in ('train','validation','test')}
    assert [{r['subject_id'] for r in groups[n]} for n in groups]==[{'subject_01'},{'subject_05'},{'subject_06'}]
    for name,rr in groups.items():
        assert len(rr)==2
        for r in rr:assert (tmp_path/'splits/legacy_05_06'/r['path']).resolve().parent==tmp_path/'recordings'
    assert len(read(tmp_path/'subjects/subject_01.jsonl'))==2


def test_real_conversion_reuse_and_loader_end_to_end(tmp_path,monkeypatch):
    h5py=pytest.importorskip('h5py');import numpy as np
    from continuous_mmwave.parrad_convert import convert_recording
    from continuous_mmwave.parrad_expand import main
    from continuous_mmwave.radar_text.data import OfflineCaptionDataset
    raw=tmp_path/'raw';sizes={}
    for subject in ('subject_01','subject_05'):
        d=raw/subject/'captured_data/set000';d.mkdir(parents=True)
        with h5py.File(d/'processed_data.h5','w') as h:
            g=h.create_group('radar/radar');g['rdn']=np.ones((160,93,128),np.float32);g['mDoppler']=np.ones((160,128),np.float32)
        (d/'timestamp_speech.csv').write_text('timestamp,command\n2019-01-01T00:00:00,walk to bed\n2019-01-01T00:00:01,sit down on bed\n')
        for filename in ('processed_data.h5','timestamp_speech.csv'):
            p=d/filename;p.with_name(filename+'.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+filename)
        sizes[subject+'/set000']=(d/'processed_data.h5').stat().st_size
    old=tmp_path/'old';source=raw/'subject_05/captured_data/set000'
    convert_recording(source/'processed_data.h5',source/'timestamp_speech.csv',old,subject_id='subject_05',session_id='set000',environment_id='parrad')
    vocabulary=json.loads((old/'label_map.json').read_text());phrases={r['id']:r['command'] for r in vocabulary['classes']}
    expected=tmp_path/'sizes.json';expected.write_text(json.dumps(sizes));text=tmp_path/'phrases.json';text.write_text(json.dumps(phrases));out=tmp_path/'output'
    original_ids=[w[3] for w in OfflineCaptionDataset(old/'manifest.jsonl',phrases).windows]
    monkeypatch.setattr('sys.argv',['expand','--raw',str(raw),'--existing',str(old),'--expected-sizes',str(expected),'--phrases',str(text),'--output',str(out)])
    main();summary=json.loads((out/'summary.json').read_text())
    assert summary['new_recordings']==summary['reused_recordings']==1 and summary['recordings']==2 and not summary['training_started']
    data=OfflineCaptionDataset(out/'manifest.jsonl',phrases)
    assert [w[3] for w in data.windows if w[3].startswith('subject_05/')]==original_ids
    assert json.loads((out/'status.json').read_text())['state']=='complete'
    assert all(r['full_arrays_raw_equal'] for r in json.loads((out/'recording-audit.json').read_text()))


def test_recording_audit_rejects_matching_but_truncated_source_prefix(tmp_path):
    h5py=pytest.importorskip('h5py');import numpy as np
    from continuous_mmwave.parrad_expand import verify_recording
    h5=tmp_path/'source.h5';csv=tmp_path/'commands.csv';saved=tmp_path/'saved.npz'
    with h5py.File(h5,'w') as h:
        g=h.create_group('radar/radar');g['rdn']=np.ones((257,93,128),np.float32);g['mDoppler']=np.ones((257,128),np.float32)
    csv.write_text('timestamp,command\n2019-01-01T00:00:00,walk to bed\n')
    np.savez(saved,rd=np.ones((256,1,93,128),np.float32),md=np.ones((256,128),np.float32),labels=np.zeros(256,np.int64),timestamps=np.arange(256)*.09)
    with pytest.raises(ValueError,match='source shape'):
        verify_recording(dict(path=str(saved),sensor_id='radar'),tmp_path/'manifest.jsonl',dict(h5=str(h5),commands=str(csv)),{'walk to bed':0})
