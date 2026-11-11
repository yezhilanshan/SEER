from __future__ import annotations
import importlib
import json
import numpy as np
import pytest
import torch


def test_diagnostic_windows_use_explicit_recording_origin(tmp_path):
    a=importlib.import_module('continuous_mmwave.split_birnn.audit')
    md=np.arange(600*128,dtype=np.float32).reshape(600,128)
    a.diagnostic_plot(md,np.zeros(600,dtype=np.int64),np.array([0.,15.,25.,40.]),
                      tmp_path/'diagnostic.png',42,frame_origin_s=-10.)
    windows=list(tmp_path.glob('diagnostic-*.npz'))
    assert len(windows)==2
    for p in windows:
        with np.load(p) as d:
            # Raw value encodes its original frame index; expected clock is independent.
            frame_index=d['md'][:,0].astype(np.float64)/128
            np.testing.assert_allclose(d['times'],-10+frame_index*.09,atol=1e-10)
            assert ((d['command_offsets_s']>=d['times'][0]) & (d['command_offsets_s']<=d['times'][-1])).all()


def test_real_optimization_probe_rejects_unsupervised_batch_without_mutation():
    p=importlib.import_module('continuous_mmwave.split_birnn.probe')
    m=importlib.import_module('continuous_mmwave.split_birnn.model').SplitBiGRU().train()
    opt=torch.optim.Adam(m.parameters(),lr=.001)
    before={k:v.clone() for k,v in m.state_dict().items()}
    with pytest.raises(ValueError,match='supervised'):
        p.step(m,opt,torch.randn(2,12,128),torch.full((2,12),-100,dtype=torch.long))
    assert all(torch.equal(before[k],v) for k,v in m.state_dict().items())
    assert not opt.state


def test_csv_boundary_outside_recording_is_not_mistaken_for_extrapolated_tail():
    d=importlib.import_module('continuous_mmwave.split_birnn.data')
    _,a=d.align_commands(4,np.array([0.,.9]),np.array([0,1]))
    assert a['last_command_tail_is_extrapolated'] is False
    assert a['unconfirmed_extrapolated_tail_s']==0
    assert a['csv_end_to_recording_end_signed_s']==pytest.approx(-.54)


def write_source(tmp_path, *, dtype='float64', with_md=True):
    h5py=pytest.importorskip('h5py')
    s=tmp_path/'raw'/'subject_01'/'captured_data'/'set000';s.mkdir(parents=True)
    md=np.tile(np.array([1.+2**-30]*128,dtype=dtype),(12,1))
    with h5py.File(s/'processed_data.h5','w') as f:
        f.create_group('radar/TI_xWR14xx')
        if with_md: f.create_dataset('radar/TI_xWR14xx/mDoppler',data=md)
    (s/'timestamp_speech.csv').write_text('timestamp,command\n2020-01-01T00:00:00,walk to bed\n',encoding='utf-8')
    return md


def test_hdf5_native_precision_is_not_lost_before_value_equality_check(tmp_path):
    md=write_source(tmp_path)
    a=importlib.import_module('continuous_mmwave.split_birnn.audit')
    out=tmp_path/'out';a.prepare_data(tmp_path/'raw',out,make_plots=False)
    r=json.loads((out/'manifest.jsonl').read_text().strip())
    with np.load(out/r['path']) as d:
        assert d['md'].dtype==md.dtype
        assert np.array_equal(d['md'],md)
    assert r['source_md_dtype']=='float64'


def test_final_session_without_md_fails_instead_of_completing_empty_dataset(tmp_path):
    write_source(tmp_path,with_md=False)
    a=importlib.import_module('continuous_mmwave.split_birnn.audit')
    with pytest.raises(ValueError,match='MD'):
        a.prepare_data(tmp_path/'raw',tmp_path/'out',make_plots=False)
    status=json.loads((tmp_path/'out'/'status.json').read_text())
    assert status['formal_training_ready'] is False and not status['state'].startswith('completed')
