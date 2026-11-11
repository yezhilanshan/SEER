import pytest
import torch


def stepper():
    import continuous_mmwave.parrad_temporal_qa as r
    assert hasattr(r,'finite_optimizer_step'), 'AMP overflow must retry the same batch without an optimizer update'
    return r.finite_optimizer_step


def test_actual_half_overflow_retries_without_changing_weights_rng_or_update_count():
    step=stepper(); p=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.AdamW([p],lr=.001,weight_decay=0.)
    scaler=torch.amp.GradScaler('cpu',init_scale=256.)
    torch.manual_seed(42); torch.rand(()); want_next=torch.rand(())
    torch.manual_seed(42); rejected=[]
    def backward():
        assert float(p.detach())==1. and not optimizer.state
        torch.rand(())
        loss=(p.half()*100).square().float()
        scaler.scale(loss).backward()
        return float(loss.detach())
    result=step([p],optimizer,scaler,backward,torch.device('cpu'),retry_sink=rejected.append)
    assert result['loss']==10000. and result['gradient_norm']==20000.
    assert result['gradient_backoffs']==7 and result['loss_scale']==2.
    assert len(rejected)==7 and all(r['reason']=='nonfinite-gradient' for r in rejected)
    assert torch.equal(torch.rand(()),want_next)
    assert {int(v['step']) for v in optimizer.state.values()}=={1}
    assert torch.isfinite(p) and p<1.


def test_exhausted_overflow_never_updates_or_leaves_invalid_gradients():
    step=stepper(); p=torch.nn.Parameter(torch.tensor(1.))
    optimizer=torch.optim.AdamW([p]); scaler=torch.amp.GradScaler('cpu',init_scale=256.)
    def backward():
        loss=(p.half()*100).square().float(); scaler.scale(loss).backward(); return float(loss.detach())
    with pytest.raises(RuntimeError,match='nonfinite'):
        step([p],optimizer,scaler,backward,torch.device('cpu'),max_backoffs=0)
    assert float(p.detach())==1. and not optimizer.state and p.grad is None
