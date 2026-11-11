import numpy as np
import pytest


def test_event_dataset_refuses_test_and_preserves_odd_full_segment(tmp_path):
    from continuous_mmwave.harrad.event_data import EventDataset,event_collate
    x=np.random.default_rng(2).normal(size=(11,4,64)).astype(np.float32)
    p=tmp_path/'cells.npy'; np.save(p,x)
    rows=[{'role':'train','subject':'subject1','uid':'a','start':1,'stop':10,'label':0,'instance_id':1,'cells_path':str(p)}]
    ds=EventDataset(rows); b=event_collate([ds[0]])
    assert b['cells'].shape==(1,9,4,64) and b['mask'].sum()==9
    np.testing.assert_array_equal(b['cells'][0].numpy(),x[1:10])
    with pytest.raises(ValueError,match='test'): EventDataset([{**rows[0],'role':'test','cells_path':'not-opened'}])


def test_event_rows_do_not_include_gru_or_future_features():
    from continuous_mmwave.harrad.event_data import event_rows
    records=[{'role':'train','file':'a','subject':'s','cells_path':'cells.npy','oracle_path':'future.npy',
              'instances':[{'start':2,'stop':5,'label':1,'instance_id':4}]}]
    r=event_rows(records)[0]
    assert r['cells_path']=='cells.npy' and 'oracle_path' not in r and 'hidden_path' not in r
