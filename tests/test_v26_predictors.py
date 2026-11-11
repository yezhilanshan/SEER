import importlib
import importlib.util
import torch


def api():
    assert importlib.util.find_spec('continuous_mmwave.v26.predictors') is not None, 'prediction arms missing'
    return importlib.import_module('continuous_mmwave.v26.predictors')


def test_all_prediction_arms_have_causal_prefix_and_bounded_variance():
    module = api()
    config = dict(motion_dim=8,radar_dim=8,event_dim=16,hidden=8,head_width=12,logvar_min=-6.,logvar_max=2.)
    motion = torch.randn(2,8)
    prefix = torch.randn(2,5,4,8)
    mask = torch.tensor([[1,1,1,0,0],[1,1,1,1,1]],dtype=torch.bool)
    event = torch.randn(2,8,16)
    for kind in ('event_free','gru','tcn','event'):
        model=module.make_predictor(kind,config).eval()
        mean,var=model(motion,prefix,event,mask)
        assert mean.shape == var.shape == (2,8)
        assert torch.isfinite(mean).all() and (var >= -6).all() and (var <= 2).all()
        changed=prefix.clone(); changed[0,3:] += 100
        torch.testing.assert_close(model(motion,changed,event,mask)[0][0],mean[0],rtol=0,atol=0)
        (mean.square().mean()+var.square().mean()).backward()
        assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    free=module.make_predictor('event_free',config)
    torch.testing.assert_close(free(motion,prefix,event,mask)[0],free(motion,prefix,event+torch.randn_like(event),mask)[0])
    event_model=module.make_predictor('event',config)
    assert not torch.equal(event_model(motion,prefix,event,mask)[0],event_model(motion,prefix,event+torch.randn_like(event),mask)[0])


def test_architecture_selection_matches_actual_used_parameter_budget():
    module = api()
    for kind in ('event_free','gru','tcn','event'):
        config = module.select_architecture(kind,dict(motion_dim=256,radar_dim=256,event_dim=896,parameter_budget=400000))
        count=sum(p.numel() for p in module.make_predictor(kind,config).parameters())
        assert abs(count-400000)/400000 <= .1
        assert config['parameter_count'] == count
