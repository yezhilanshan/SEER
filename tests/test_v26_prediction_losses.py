import importlib
import importlib.util
import math
import pytest
import torch


def api():
    assert importlib.util.find_spec('continuous_mmwave.v26.losses') is not None, 'prediction loss missing'
    return importlib.import_module('continuous_mmwave.v26.losses')


def test_gaussian_nll_scale_and_recording_reduction():
    module = api()
    zero = torch.zeros(3,4)
    loss = module.gaussian_nll(zero,zero,zero)
    torch.testing.assert_close(loss,torch.full((3,),.5*math.log(2*math.pi)))
    torch.testing.assert_close(module.gaussian_nll(zero+1,zero,zero),loss+.5)
    value = module.reduce_prediction_losses(torch.tensor([2.,2.,10.]),torch.tensor([0,0,1]),torch.ones(3,dtype=torch.bool))
    assert value.item() == 6
    with pytest.raises(ValueError, match='valid'):
        module.reduce_prediction_losses(loss,torch.zeros(3,dtype=torch.long),torch.zeros(3,dtype=torch.bool))


def test_invalid_targets_do_not_silently_enter_loss():
    module = api()
    with pytest.raises(ValueError,match='finite'):
        module.gaussian_nll(torch.zeros(1,2),torch.zeros(1,2),torch.full((1,2),float('nan')))


def test_persistence_variance_is_recording_equal_and_duplication_invariant():
    module=api()
    # A residual^2=1, B residual^2=9 -> optimum variance=5, not 109/101.
    try:
        one=module.fit_persistence_variance(torch.zeros(2,1),torch.tensor([[1.],[3.]]),torch.tensor([0,1]))
        duplicated=module.fit_persistence_variance(torch.zeros(101,1),torch.cat((torch.ones(100,1),torch.tensor([[3.]]))),torch.tensor([0]*100+[1]))
    except TypeError:
        pytest.fail('recording-equal variance interface missing')
    torch.testing.assert_close(one,torch.tensor([math.log(5.)]))
    torch.testing.assert_close(duplicated,one)
