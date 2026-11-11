import numpy as np
import torch

from continuous_mmwave.harrad.event_prediction import PredictionModel, future_cutoffs, sample_indices


def test_horizon_is_exact_raw_frame_offset_and_never_crosses_record():
    current=np.array([2,4,9,15])
    np.testing.assert_array_equal(future_cutoffs(current),[[5,11,17],[7,13,19],[12,18,24],[18,24,30]])
    np.testing.assert_array_equal(sample_indices(current,24),[0,1,2])


def test_input_projection_starts_as_fixed_pool_and_does_not_change_target():
    model=PredictionModel('event_free',8)
    radar=torch.randn(2,4,256)
    target=radar.mean(1).detach().clone()
    torch.testing.assert_close(model.motion(radar.flatten(1)),target)
    loss=model.motion(radar.flatten(1)).square().mean()
    loss.backward()
    assert model.motion.weight.grad.abs().max()>0
    assert not target.requires_grad


def test_future_prefix_padding_cannot_change_history_prediction():
    model=PredictionModel('gru',8).eval()
    with torch.no_grad():
        for head in model.heads:
            head.head[-1].weight.normal_(std=.02)
    radar=torch.randn(2,4,256)
    event=torch.randn(2,8,896)
    prefix=torch.randn(2,8,4,256)
    mask=torch.ones(2,8,dtype=torch.bool)
    mask[:,5:]=False
    before=model(radar,event,prefix,mask)[0]
    prefix[:,5:]*=100
    after=model(radar,event,prefix,mask)[0]
    torch.testing.assert_close(before,after,atol=0,rtol=0)
