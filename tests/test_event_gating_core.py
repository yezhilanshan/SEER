import math

import pytest
import torch


def test_cif_splits_mass_across_tokens_without_losing_remainder():
    from continuous_mmwave.event_gating.cif import cif_integrate
    x = torch.tensor([[2.], [6.], [10.]])
    out = cif_integrate(x, torch.tensor([.6, .8, .3]))
    torch.testing.assert_close(out.tokens, torch.tensor([[3.6]]))
    assert out.residual_mass.item() == pytest.approx(.7)
    assert out.residual_sum.item() == pytest.approx(5.4)
    assert out.raw_mass.item() == pytest.approx(1.7)


def test_cif_scaling_supports_multiple_tokens_per_frame_and_has_gradients():
    from continuous_mmwave.event_gating.cif import cif_integrate
    x = torch.tensor([[2.], [8.]], requires_grad=True)
    alpha = torch.tensor([.2, .3], requires_grad=True)
    out = cif_integrate(x, alpha, target_count=3)
    assert len(out.tokens) == 3
    assert out.raw_mass.item() == pytest.approx(.5)
    assert out.residual_mass.item() == pytest.approx(0, abs=1e-5)
    out.tokens.square().sum().backward()
    assert x.grad.abs().sum() > 0
    assert alpha.grad.abs().sum() > 0


def test_lif_integrates_in_seconds_instead_of_per_frame():
    from continuous_mmwave.event_gating.lif import LIFController
    gate = LIFController(tau=1.5, theta=10, beta=0)
    for dt, n in [(.1, 20), (.25, 8)]:
        state = None
        for _ in range(n):
            step = gate.step(torch.tensor(.4), dt, state)
            state = step.state
        assert state.voltage.item() == pytest.approx(.6 * (1-math.exp(-2/1.5)), rel=1e-5)


def test_lif_zero_drive_and_surrogate_gradient_and_adaptation():
    from continuous_mmwave.event_gating.lif import LIFController
    gate = LIFController(tau=1.5, beta=1)
    state = None
    for _ in range(50):
        step = gate.step(torch.tensor(0.), .1, state)
        state = step.state
    assert state.voltage.item() == 0
    u = torch.tensor(4., requires_grad=True)
    step = gate.step(u, .5)
    assert step.hard_spike.item() == 1
    step.spike.backward()
    assert u.grad.item() > 0
    following = gate.step(torch.tensor(0.), .1, step.state)
    assert following.threshold.item() > 1
    assert step.state.voltage.item() >= 0


@pytest.mark.parametrize('dt', [0., -1., float('nan')])
def test_lif_rejects_invalid_time_step(dt):
    from continuous_mmwave.event_gating.lif import LIFController
    with pytest.raises(ValueError):
        LIFController().step(torch.tensor(1.), dt)


def test_cif_long_fp32_sequences_close_scaled_mass_exactly_without_losing_gradients():
    from continuous_mmwave.event_gating.cif import cif_integrate
    torch.manual_seed(1)
    for count in (1,2,3):
        alpha=torch.sigmoid(torch.randn(900)-2).requires_grad_()
        x=torch.linspace(0,1,900).view(-1,1).requires_grad_()
        out=cif_integrate(x,alpha,count)
        assert len(out.tokens)==count
        assert out.residual_mass.item()==pytest.approx(0,abs=1e-9)
        expected=(alpha.double()*count/alpha.double().sum()*x[:,0].double()).sum()
        torch.testing.assert_close(out.tokens.double().sum(),expected,atol=1e-6,rtol=1e-6)
        out.tokens.square().sum().backward()
        assert torch.isfinite(alpha.grad).all() and alpha.grad.abs().sum()>0
