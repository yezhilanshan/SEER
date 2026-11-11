import pytest
from continuous_mmwave.v25 import data
from continuous_mmwave.v25.stage3 import V25Stage3
import torch


def test_visible_pair_label_does_not_copy_future_transition_backwards():
    from continuous_mmwave.v25.collect import visible_pair_label
    intervals=[(0.,1.,0),(1.,2.,1),(2.,3.,0)]
    assert visible_pair_label(intervals,reference_time=.72,current_time=.90)==0
    assert visible_pair_label(intervals,reference_time=.72,current_time=1.18) is None
    assert visible_pair_label(intervals,reference_time=.72,current_time=1.72)==1
    assert visible_pair_label(intervals,reference_time=.72,current_time=2.18)==0


def test_oracle_phase_is_unscaled_bce_and_only_semantic_trainable():
    stage=V25Stage3(radar_dim=8,event_dim=16,semantic_dim=8)
    stage.set_phase('oracle')
    row=dict(active=torch.randn(4,8),candidate=torch.randn(4,8),target=torch.tensor([0.,1.,0.,1.]),
             age_bucket=torch.tensor([0,0,1,1]))
    result=stage.oracle_step(row)
    expected=torch.nn.functional.binary_cross_entropy_with_logits(stage.heads.semantic(row['active'],row['candidate']),row['target'])
    torch.testing.assert_close(result['loss'],expected)
    result['loss'].backward()
    assert all(p.grad is None for p in stage.heads.local.parameters())
    assert any(p.grad is not None for p in stage.heads.semantic.parameters())
