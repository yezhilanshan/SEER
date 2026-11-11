"""Catches dropped odd frames, event-dependent writes, and noncausal consolidation."""
import copy
import numpy as np
import pytest
import torch


def test_time_memory_is_bounded_weighted_and_restorable():
    from continuous_mmwave.latent_memory.frontend import ContinuousMemoryStream
    stream = ContinuousMemoryStream(capacities=(2,2,2), group_size=2)
    cells = torch.arange(38.).view(38,1,1).expand(-1,4,64).clone()
    stream.push(cells)
    memory = stream.snapshot()
    assert all(len(level) <= 2 for level in memory.levels)
    items = memory.candidates(2)
    assert items[0].start_frame == 0 and items[-1].end_frame == 38
    assert sum(float(x.weights.sum()) for x in items) == 38
    assert sum(float((x.values*x.weights[:,None,None]).sum()) for x in items) == 703*256
    restored = type(memory).restore(memory.export())
    assert [(x.start_frame,x.end_frame) for x in restored.candidates(2)] == [(x.start_frame,x.end_frame) for x in items]
    for group in range(3):
        assert all(x.level <= group for x in memory.candidates(group))
    assert memory.candidates(0) != memory.candidates(2)


def test_odd_query_is_a_snapshot_and_chunking_does_not_change_stream():
    from continuous_mmwave.latent_memory.frontend import ContinuousMemoryStream
    torch.manual_seed(7); cells=torch.randn(19,4,64)
    a=ContinuousMemoryStream(capacities=(2,2,2),group_size=2)
    a.push(cells[:3]); snapshot=a.snapshot()
    assert snapshot.candidates(2)[-1].end_frame == 3
    assert a.buffer.pending.shape == (1,4,64)
    a.push(cells[3:8]); a.snapshot(); a.push(cells[8:])
    b=ContinuousMemoryStream(capacities=(2,2,2),group_size=2); b.push(cells)
    for x,y in zip(a.snapshot().candidates(2),b.snapshot().candidates(2)):
        assert (x.start_frame,x.end_frame)==(y.start_frame,y.end_frame)
        torch.testing.assert_close(x.values,y.values)
    assert snapshot.candidates(2)[-1].end_frame == 3
    changed=cells.clone(); changed[3:]*=100
    c=ContinuousMemoryStream(); c.push(changed[:3])
    torch.testing.assert_close(snapshot.candidates(2)[0].values,c.snapshot().candidates(2)[0].values)


def test_stream_rejects_invalid_frames_without_consuming_them():
    from continuous_mmwave.latent_memory.frontend import ContinuousMemoryStream
    stream=ContinuousMemoryStream()
    with pytest.raises(ValueError): stream.push(torch.full((2,4,64),float('nan')))
    assert stream.observed_frames == 0
    with pytest.raises(ValueError): stream.push(torch.ones(2,64))


def test_stream_owns_pending_evidence_when_acquisition_buffer_is_reused():
    from continuous_mmwave.latent_memory.frontend import ContinuousMemoryStream
    stream=ContinuousMemoryStream(); values=torch.ones(1,4,64)
    stream.push(values); values.fill_(99)
    assert float(stream.snapshot().candidates(2)[0].values[0,0,0])==1.
    stream.push(torch.full((1,4,64),2.))
    assert stream.snapshot().candidates(2)[0].values[:,0,0].tolist()==[1.,2.]


def test_fixed_queries_and_supervision_keep_repeats_and_no_future_end():
    from continuous_mmwave.latent_memory.data import query_cutoffs, prefix_reference
    assert query_cutoffs(11,4) == [4,8,11]
    intervals=[dict(start=0,stop=2,label=0),dict(start=3,stop=6,label=0),dict(start=8,stop=10,label=1)]
    reference=prefix_reference(intervals,5)
    assert reference['sequence'] == [0,0]
    assert reference['ongoing'] == [False,True]
    assert reference['known_frames'] == 4 and reference['unknown_frames'] == 1
    assert reference['text'].lower().count('stood up') == 2
    alternate=copy.deepcopy(intervals); alternate[1]['stop']=7; alternate[2]['label']=11
    assert prefix_reference(alternate,5)==reference


def test_manifest_verification_rejects_test_subjects_and_changed_cache(tmp_path):
    import json,hashlib
    from continuous_mmwave.latent_memory.data import load_records
    path=tmp_path/'cells.npy'; np.save(path,np.ones((5,4,64),np.float32))
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    row=dict(file='subject1_01.hdf5',role='train',subject='subject1',frames=5,cells_path=str(path),
             artifacts={'cells.npy':{'sha256':digest}},instances=[])
    manifest=tmp_path/'manifest.json'; manifest.write_text(json.dumps([row]))
    assert len(load_records(manifest))==1
    np.save(path,np.zeros((5,4,64),np.float32))
    with pytest.raises(ValueError,match='hash'): load_records(manifest)
    row['role']='test'; row['subject']='subject8'; manifest.write_text(json.dumps([row]))
    with pytest.raises(ValueError,match='role|subject|test'): load_records(manifest)
