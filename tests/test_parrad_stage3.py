from __future__ import annotations

import math

import pytest
import torch

from continuous_mmwave.models.transition import (
    LocalChangeHead,
    MonotonicVerifyHead,
    PrototypeState,
    RadarTransitionHead,
    SemanticDifferenceHead,
    candidate_stability,
    update_prototype,
)
from continuous_mmwave.parrad_stage3_data import (
    PhysicalBatch,
    PhysicalSequenceBatch,
    SemanticPairBatch,
    VerifyBatch,
    VerifyFeatureBatch,
)
from continuous_mmwave.stages.parrad_stage3 import PARradStage3Module


def test_local_change_head_shapes_masks_and_rejects_all_masked() -> None:
    head = LocalChangeHead(radar_dim=16, hidden_dim=8)
    previous = torch.randn(2, 4, 16)
    current = torch.randn(2, 4, 16)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    assert head(previous, current, mask).shape == (2,)
    with pytest.raises(ValueError, match="valid radar token"):
        head(previous[:1], current[:1], torch.zeros(1, 4, dtype=torch.bool))


def test_local_change_masked_tokens_do_not_change_logit() -> None:
    torch.manual_seed(3)
    head = LocalChangeHead(radar_dim=4, hidden_dim=8).eval()
    previous = torch.randn(1, 4, 4)
    current = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, True, False, False]])
    changed = current.clone()
    changed[:, 2:] = 10_000

    torch.testing.assert_close(
        head(previous, current, mask),
        head(previous, changed, mask),
    )


def test_prototype_state_uses_physical_time_constant() -> None:
    state = PrototypeState.initialize(torch.tensor([[1.0, 0.0]]))

    updated = update_prototype(state, torch.tensor([[0.0, 1.0]]), 0.18, 2.0)

    assert updated.alpha == pytest.approx(math.exp(-0.18 / 2.0))
    assert updated.count == 2
    assert updated.age_seconds == pytest.approx(0.18)


def test_radar_transition_is_finite_for_zero_variance() -> None:
    head = RadarTransitionHead(radar_dim=16, prototype_dim=8)
    tokens = torch.ones(2, 4, 16)

    logits, state = head(tokens, torch.ones(2, 4, dtype=torch.bool), None, 0.18)

    assert logits.shape == (2,)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(state.variance).all()


def test_candidate_stability_requires_three_keys() -> None:
    assert candidate_stability(torch.randn(2, 8), tau=0.1).item() == 0
    assert candidate_stability(torch.randn(3, 8), tau=0.1).ndim == 0
    assert candidate_stability(torch.randn(4, 3, 8), tau=0.1).shape == (4,)


def test_semantic_difference_head_returns_one_logit_per_pair() -> None:
    head = SemanticDifferenceHead(semantic_dim=8, hidden_dim=16)
    active, candidate = torch.randn(3, 8), torch.randn(3, 8)

    output = head(active, candidate, torch.tensor([0.2, 0.5, 1.0]))

    assert output.shape == (3,)
    assert torch.isfinite(output).all()


def test_verify_probability_is_monotonic_in_each_input() -> None:
    head = MonotonicVerifyHead()
    base = torch.tensor([[0.2, 0.3, 0.4]])

    for column in range(3):
        higher = base.clone()
        higher[:, column] += 0.2
        assert head(higher).item() >= head(base).item()
    assert torch.all(head.positive_weights() >= 0)


@pytest.mark.parametrize(
    ("phase", "trainable_prefixes"),
    [
        ("physical", ("local_change", "radar_transition")),
        ("semantic", ("semantic_difference",)),
        ("verify", ("verify",)),
    ],
)
def test_stage3_phase_exposes_only_expected_parameters(
    phase: str, trainable_prefixes: tuple[str, ...]
) -> None:
    module = PARradStage3Module(radar_dim=8, semantic_dim=4, prototype_dim=4)

    module.set_phase(phase)

    trainable = {name for name, parameter in module.named_parameters() if parameter.requires_grad}
    assert trainable
    assert all(name.startswith(trainable_prefixes) for name in trainable)


def test_stage3_losses_are_logits_based_finite_and_weighted() -> None:
    torch.manual_seed(7)
    module = PARradStage3Module(radar_dim=8, semantic_dim=4, prototype_dim=4)
    physical = PhysicalBatch(
        previous=torch.randn(2, 4, 8),
        current=torch.randn(2, 4, 8),
        mask=torch.ones(2, 4, dtype=torch.bool),
        local_target=torch.tensor([0.0, 1.0]),
        radar_target=torch.tensor([0.0, 1.0]),
        reset=torch.tensor([False, True]),
    )
    semantic = SemanticPairBatch(
        active=torch.randn(2, 4),
        candidate=torch.randn(2, 4),
        maturity=torch.tensor([0.5, 1.0]),
        target=torch.tensor([0.0, 1.0]),
    )
    verify = VerifyBatch(
        active=torch.randn(2, 4),
        key_history=torch.randn(2, 3, 4),
        maturity=torch.tensor([0.5, 1.0]),
        target=torch.tensor([0.0, 1.0]),
        weight=torch.tensor([1.0, 3.0]),
    )

    physical_metrics = module.physical_step(physical)
    semantic_metrics = module.semantic_step(semantic)
    verify_metrics = module.verify_step(verify)

    assert set(physical_metrics) >= {"loss", "local_loss", "radar_loss"}
    assert set(semantic_metrics) >= {"loss", "logits"}
    assert set(verify_metrics) >= {"loss", "logits", "stability"}
    assert all(
        torch.isfinite(metrics["loss"])
        for metrics in (physical_metrics, semantic_metrics, verify_metrics)
    )
    assert semantic_metrics["logits"].shape == verify_metrics["logits"].shape == (2,)
    module.semantic_difference.eval()
    expected_difference = torch.sigmoid(
        module.semantic_difference(verify.active, verify.key_history[:, -1], verify.maturity)
    )
    torch.testing.assert_close(
        verify_metrics["difference_probability"], expected_difference.detach()
    )


def test_verify_step_accepts_precomputed_closed_loop_features() -> None:
    module = PARradStage3Module(radar_dim=8, semantic_dim=4, prototype_dim=4)
    batch = VerifyFeatureBatch(
        inputs=torch.tensor([[0.8, 0.6, 0.5], [0.2, 0.1, 1.0]]),
        target=torch.tensor([1.0, 0.0]),
        weight=torch.tensor([1.0, 2.0]),
    )

    metrics = module.verify_step(batch)

    assert metrics["logits"].shape == (2,)
    torch.testing.assert_close(metrics["verify_inputs"], batch.inputs)
    assert torch.isfinite(metrics["loss"])


def test_physical_sequence_step_carries_prototype_state_across_updates() -> None:
    torch.manual_seed(11)
    module = PARradStage3Module(radar_dim=8, semantic_dim=4, prototype_dim=4)
    batch = PhysicalSequenceBatch(
        tokens=torch.randn(1, 4, 4, 8),
        mask=torch.ones(1, 4, 4, dtype=torch.bool),
        local_target=torch.tensor([[0.0, 0.5, 1.0]]),
        radar_target=torch.tensor([[0.0, 0.5, 1.0]]),
        valid=torch.ones(1, 3, dtype=torch.bool),
        weight=torch.ones(1, 3),
    )

    metrics = module.physical_step(batch)

    assert metrics["local_logits"].shape == metrics["radar_logits"].shape == (1, 3)
    assert metrics["radar_state_count"] == 4
    assert torch.isfinite(metrics["loss"])
