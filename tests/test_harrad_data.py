import json
from pathlib import Path

import numpy as np
import pytest


def test_instances_keep_adjacent_same_class_separate_and_gaps_unknown(tmp_path):
    from continuous_mmwave.harrad.data import read_instances, frame_annotations, subject_role
    csv = tmp_path / 'activities.csv'
    csv.write_text('subject1_01.hdf5,1,3,standing_up,subject1\nsubject1_01.hdf5,3,5,standing_up,subject1\n')
    rows = read_instances(csv, {'subject1_01.hdf5': 7})
    labels, ids = frame_annotations(rows, 'subject1_01.hdf5', 7)
    assert labels.tolist() == [-100, 0, 0, 0, 0, -100, -100]
    assert ids.tolist() == [-1, 0, 0, 1, 1, -1, -1]
    assert subject_role('subject1', {'train':['subject1'], 'validation':['subject2'], 'test':['subject3']}) == 'train'
    with pytest.raises(ValueError, match='split'):
        subject_role('subject1', {'train':['subject1'], 'validation':['subject1'], 'test':['subject3']})


@pytest.mark.parametrize('row', ['subject1_01.hdf5,1,8,standing_up,subject1', 'subject1_01.hdf5,1,3,no_such_class,subject1', '../subject1_01.hdf5,1,3,standing_up,subject1', 'subject1_01.hdf5,1,3,standing_up,subject2'])
def test_bad_annotations_rejected(tmp_path, row):
    from continuous_mmwave.harrad.data import read_instances
    p=tmp_path/'a.csv'; p.write_text(row+'\n')
    with pytest.raises(ValueError): read_instances(p, {'subject1_01.hdf5':7})


def test_normalization_excludes_unlabeled_and_other_roles(tmp_path):
    from continuous_mmwave.harrad.data import fit_normalizers
    records=[]
    for role, x in [('train',np.array([[1.,3.],[3.,7.],[1000.,1000.]],np.float32)),('validation',np.full((3,2),1e6,np.float32))]:
        p=tmp_path/role; p.mkdir()
        np.save(p/'md.npy',x); np.save(p/'rd.npy',x.reshape(3,1,2)); np.save(p/'labels.npy',np.array([0,1,-100]))
        records.append({'role':role,'features':{'md':str(p/'md.npy'),'rd':str(p/'rd.npy')},'labels_path':str(p/'labels.npy')})
    stats=fit_normalizers(records)
    np.testing.assert_allclose(stats['md_mean'],[2,5])
    np.testing.assert_allclose(stats['md_scale'],[1,2])
    assert int(stats['fit_frames'])==2


def test_dataset_refuses_test_arrays_and_keeps_full_instance(tmp_path):
    from continuous_mmwave.harrad.data import ClipDataset, collate_clips
    np.save(tmp_path/'md.npy',np.ones((169,256),np.float32))
    rows=[{'role':'train','uid':'u','subject':'subject1','start':0,'stop':169,'label':0,'instance_id':0,'feature_path':str(tmp_path/'md.npy')}]
    ds=ClipDataset(rows, {'mean':np.zeros(256),'scale':np.ones(256)}, 'md')
    batch=collate_clips([ds[0]])
    assert batch['x'].shape==(1,169,256) and batch['mask'].sum()==169
    rows[0]['role']='test'; rows[0]['feature_path']=str(tmp_path/'must_not_open.npy')
    with pytest.raises(ValueError,match='test'): ClipDataset(rows,{'mean':np.zeros(256),'scale':np.ones(256)},'md')
