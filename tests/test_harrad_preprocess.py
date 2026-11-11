import ast
from pathlib import Path
import numpy as np
import pytest


def test_fft_matches_published_official_function_and_does_not_change_raw():
    from continuous_mmwave.harrad.preprocess import frame_features
    source=Path('outputs/harrad-format-audit-20261016-v1/process.py').read_text()
    tree=ast.parse(source)
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='range_doppler')
    space={'np':np}; exec(compile(ast.Module(body=[fn],type_ignores=[]),'official','exec'),space)
    raw=np.random.default_rng(3).normal(size=(256,256)).astype(np.float32)
    before=raw.copy(); md,rd=frame_features(raw)
    official_db=20*np.log10(space['range_doppler'](raw))
    np.testing.assert_allclose(md,official_db.sum(0).astype(np.float32),atol=1e-4,rtol=1e-6)
    np.testing.assert_allclose(rd,official_db.reshape(40,4,64,4).mean((1,3)).astype(np.float32),atol=1e-5,rtol=1e-6)
    np.testing.assert_array_equal(raw,before)


def test_zero_frame_finite_and_sequence_sample_excluded():
    from continuous_mmwave.harrad.preprocess import frame_features
    x=np.zeros((256,256),np.float32); md,rd=frame_features(x)
    assert np.isfinite(md).all() and np.isfinite(rd).all()
    x[:,0]=1e9
    changed=frame_features(x)
    np.testing.assert_array_equal(md,changed[0]); np.testing.assert_array_equal(rd,changed[1])


def test_conversion_test_record_never_opens_and_source_readonly(tmp_path):
    h5py=pytest.importorskip('h5py')
    from continuous_mmwave.harrad.preprocess import convert_record
    source=tmp_path/'subject1_01.hdf5'
    with h5py.File(source,'w') as f: f['radar']=np.zeros((4,256,256),np.float32)
    before=source.read_bytes()
    meta=convert_record({'source':str(source),'file':source.name,'role':'train','subject':'subject1','frames':4,'instances':[{'start':1,'stop':3,'label':0,'instance_id':0}]},str(tmp_path/'canonical'))
    assert source.read_bytes()==before
    assert np.load(meta['labels_path']).tolist()==[-100,0,0,-100]
    assert np.load(meta['features']['rd']).shape==(4,40,64)
    with pytest.raises(ValueError,match='test'):
        convert_record({'source':str(tmp_path/'missing.hdf5'),'role':'test'},str(tmp_path/'not_created'))
    assert not (tmp_path/'not_created').exists()
