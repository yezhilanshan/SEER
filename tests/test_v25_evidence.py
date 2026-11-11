import math

import pytest
import torch

from continuous_mmwave.v25.models import AgeFreeStateGate, SemanticDifference, EvidenceProjection, V25Heads
from continuous_mmwave.v25.evidence import EvidenceConfig, SequentialEvidence, Episode, episode_label
from continuous_mmwave.v25.training import semantic_loss, evidence_loss, age_bias_probe, age_matched_weights


def test_gate_tanh_residual_and_no_age_input():
    gate = AgeFreeStateGate(4, initial_update=.1)
    before = torch.randn(2, 3, 4)
    proposal = torch.randn_like(before)
    out, alpha = gate(before, proposal, 0.)
    other, _ = gate(before, proposal, 1.)
    assert torch.equal(out, other)
    assert gate.network[0].in_features == 8
    torch.testing.assert_close(out, before + .1 * (proposal - before))
    out.sum().backward()
    assert gate.network[-1].weight.grad.abs().sum() > 0


def test_heads_have_no_verify_or_age_features():
    heads = V25Heads(radar_dim=8, event_dim=4, semantic_dim=128, evidence_dim=4)
    assert not hasattr(heads, 'verify')
    assert heads.semantic.network[0].in_features == 513
    assert heads.radar.classifier[0].in_features == 4 * 128 + 1
    logits = heads.semantic(torch.randn(2, 128), torch.randn(2, 128))
    assert logits.shape == (2,)
    evidence = heads.evidence(torch.randn(2, 8, 4))
    assert evidence.shape == (2, 4)
    assert ((evidence >= 0) & (evidence <= 1)).all()


def test_accumulation_exact_formula_and_warmup():
    config = EvidenceConfig(rho=.9, tau_stability=.5, tau_difference=.6,
                            tau_evidence=.2, evidence_weight=2., h_plus=1., h_minus=-1., warmup_s=.5)
    state = SequentialEvidence(config)
    out = state.update(.9, .75, .4, .2)
    expected = .5 * (math.log(.9/.1) - math.log(.6/.4) + 2*(.4-.2))
    assert out.score == pytest.approx(expected, abs=1e-5)
    assert out.decision == 'none'
    out = state.update(.9, 1., .4, .5)
    assert out.score == pytest.approx(.9*expected + 2*expected)
    assert out.decision == 'confirm'


def test_timeout_rejects_even_high_diff_if_unreliable():
    state = SequentialEvidence(EvidenceConfig(timeout_s=1., warmup_s=.5))
    assert state.update(.999, 0., 0., 1.01).decision == 'timeout_reject'
    with pytest.raises(ValueError):
        state.update(float('nan'), 1., 0., 2.)


@pytest.mark.parametrize('kwargs', [{'rho':0}, {'tau_stability':1}, {'tau_difference':0},
                                    {'h_minus':1}, {'warmup_s':2,'timeout_s':1}])
def test_invalid_accumulator_config(kwargs):
    with pytest.raises(ValueError):
        EvidenceConfig(**kwargs)


def test_episode_lifecycle_label_rejects_transient_return():
    ep = Episode('c1', 'a0', 1.)
    ep.end_time = 2.
    assert episode_label(ep, [(1.,1.2,'B'), (1.2,2.,'A')], 'A', min_stable_s=.5) == 0
    assert episode_label(ep, [(1.,2.,'B')], 'A', min_stable_s=.5) == 1
    assert episode_label(ep, [(0.,1.,'A')], 'A', min_stable_s=.5) is None


def test_oracle_loss_not_concatenated_and_evidence_missing_mask():
    closed = torch.tensor([.2,-.4], requires_grad=True)
    oracle = torch.tensor([2.], requires_grad=True)
    loss = semantic_loss(closed, torch.tensor([1.,0.]), oracle, torch.tensor([0.]), oracle_weight=.3)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(closed, torch.tensor([1.,0.]))
    expected += .3 * torch.nn.functional.binary_cross_entropy_with_logits(oracle, torch.tensor([0.]))
    torch.testing.assert_close(loss, expected)
    loss.backward()
    pred = torch.tensor([[.2,.6]], requires_grad=True)
    e_loss = evidence_loss(pred, torch.tensor([[.4,float('nan')]]), torch.tensor([[True,False]]))
    e_loss.backward()
    assert pred.grad[0,1] == 0


def test_age_weights_balance_each_age_and_class_and_probe_flags_bias():
    ages = torch.tensor([1,1,1,2,2])
    labels = torch.tensor([0,0,1,0,1])
    weights = age_matched_weights(ages, labels)
    for age in (1,2):
        for label in (0,1):
            assert weights[(ages == age) & (labels == label)].sum() == pytest.approx(.25)
    result = age_bias_probe([.9,.8,.2,.1], [0,0,0,0], [1,2,3,4], early_max_age=2)
    assert result['same_early_fpr'] == 1.
    assert result['same_age_correlation'] < -.9
    assert result['early_auroc'] is None


@pytest.mark.parametrize('dtype',[torch.float16,torch.bfloat16])
def test_native_event_dtype_is_accepted_by_fp32_evidence_head(dtype):
    tokens = torch.randn(2,8,16,dtype=dtype,requires_grad=True)
    head = EvidenceProjection(16,4)
    output = head(tokens)
    assert output.dtype == torch.float32
    output.sum().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
