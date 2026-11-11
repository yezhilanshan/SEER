import json
import numpy as np
import pytest
import torch
from romsl_mvp.romsl.targets import compute_targets
from romsl_mvp.romsl.model import masked_huber


def test_proxy_requires_explicit_opt_in():
    pose=np.ones((10,17,3))
    pose[...,0]+=np.arange(10)[:,None]*.1
    joints={'left_hip':4,'right_hip':1,'groups':{
        'left_arm':[11,12,13],'right_arm':[14,15,16],'left_leg':[4,5,6],'right_leg':[1,2,3]}}
    g={'status':'PROXY_GEOMETRY_ONLY','verified':False,'R':np.eye(3),'t':np.zeros(3),
       'radar_origin_world_m':np.zeros(3),'translation_unit':'m','source':'synthetic proxy'}
    with pytest.raises(ValueError,match='BLOCKED_GEOMETRY'):
        compute_targets(pose,joints,.1,radial=True,geometry=g)
    result=compute_targets(pose,joints,.1,radial=True,geometry=g,allow_proxy=True)
    assert result['C_mask'][3:].all()
    assert g['verified'] is False


def test_D_weighted_loss_uses_correct_denominator():
    prediction=torch.tensor([[[1.],[3.]]],requires_grad=True)
    target=torch.zeros_like(prediction)
    mask=torch.ones_like(prediction,dtype=torch.bool)
    frame=torch.ones(1,2,dtype=torch.bool)
    weighted=masked_huber(prediction,target,mask,frame,torch.tensor([[.25,1.]]))
    assert weighted.item()==pytest.approx(2.1)
    torch.testing.assert_close(masked_huber(prediction,target,mask,frame,torch.ones(1,2)),
                               masked_huber(prediction,target,mask,frame))


def test_proxy_dataset_uses_C_stats_and_D_weights(tmp_path):
    from romsl_mvp.romsl.proxy_experiment import ProxyWindows
    for name in ['artifacts','cache','proxy_targets']:
        (tmp_path/name).mkdir()
    row={'cache':'E01_S01_A03.npz','start':0,'end':16,'recording_id':'E01/S01/A03','action_id':'A03'}
    (tmp_path/'artifacts/manifest_train.jsonl').write_text(json.dumps(row)+'\n')
    np.savez(tmp_path/'artifacts/target_stats.npz',B_mean=np.zeros(5),B_std=np.ones(5))
    np.savez(tmp_path/'artifacts/proxy_stats.npz',mean=np.ones(5),std=np.full(5,2))
    np.savez(tmp_path/'cache'/row['cache'],points=np.ones((16,4,5),dtype=np.float32),
             point_mask=np.ones((16,4),dtype=bool),B=np.zeros((16,5),dtype=np.float32),
             B_mask=np.ones((16,5),dtype=bool))
    np.savez(tmp_path/'proxy_targets'/row['cache'],C=np.full((16,5),3,dtype=np.float32),
             C_mask=np.ones((16,5),dtype=bool),echo_weight=np.full(16,.625,dtype=np.float32))
    c=ProxyWindows(tmp_path,'train','C_proxy')[0]
    d=ProxyWindows(tmp_path,'train','D_proxy')[0]
    assert len(c)==4 and len(d)==5
    torch.testing.assert_close(c[2],torch.ones(16,5))
    torch.testing.assert_close(d[2],c[2])
    torch.testing.assert_close(d[4],torch.full((16,),.625))
