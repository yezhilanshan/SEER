from __future__ import annotations
import importlib
import numpy as np
import pytest


def test_sampler_uses_one_length_and_keeps_real_unlabeled_prefix(tmp_path):
    probe = importlib.import_module('continuous_mmwave.split_birnn.probe')
    md = np.random.default_rng(1).normal(size=(15,128)).astype('float32')
    labels=np.array([-100]*3+[0]*12,dtype=np.int64)
    p=tmp_path/'r.npz'; np.savez(p,md=md,labels=labels)
    rows=[{'path':str(p),'recording_id':'r'}]
    x,y,origins=probe.sample_batch(rows, np.random.default_rng(3), batch_size=2, length=15)
    assert x.shape == (2,15,128) and y.shape == (2,15)
    assert y[0].tolist() == labels.tolist()
    assert origins == [{'recording_id':'r','start_frame':0,'frames':15}]*2
    with pytest.raises(ValueError):
        probe.sample_batch(rows,np.random.default_rng(3),batch_size=1,length=16)
