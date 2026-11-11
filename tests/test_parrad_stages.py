from __future__ import annotations

import importlib

import pytest
import torch

from continuous_mmwave.config import ModelConfig
from continuous_mmwave.data_parrad import PARradBatch
from continuous_mmwave.models.event_former import StreamingSemanticEventFormer, TinyLLMAdapter
from continuous_mmwave.models.parrad_encoder import PARradCausalMDGRUEncoder, PARradRadarEncoder
from continuous_mmwave.models.semantic import SemanticProjection
from continuous_mmwave.parrad_config import PARradConfig


def tiny_config() -> PARradConfig:
    return PARradConfig(
        rd_height=16,
        rd_width=16,
        md_bins=16,
        radar_dim=16,
        motion_dim=8,
        fusion_heads=4,
        sensor_vocab_size=2,
        dropout=0.0,
    )


def batch() -> PARradBatch:
    return PARradBatch(
        rd=torch.randn(2, 2, 1, 16, 16),
        md=torch.randn(2, 2, 16),
        labels=torch.tensor([[0, 1], [1, 0]]),
        timestamps=torch.tensor([[0.0, 0.09], [0.0, 0.09]], dtype=torch.float64),
        frame_mask=torch.ones(2, 2, dtype=torch.bool),
        sensor_index=torch.tensor([0, 1]),
        metadata=(
            {"subject_id": "s1", "session_id": "a", "sensor_id": "60ghz", "environment_id": "test"},
            {"subject_id": "s2", "session_id": "b", "sensor_id": "77ghz", "environment_id": "test"},
        ),
    )


def stage2_batch() -> PARradBatch:
    sample = batch()
    return PARradBatch(
        sample.rd,
        sample.md,
        torch.tensor([[0, 0], [1, 1]]),
        sample.timestamps,
        sample.frame_mask,
        sample.sensor_index,
        sample.metadata,
    )


def md_gru_config() -> PARradConfig:
    return PARradConfig(
        encoder_type="causal_md_gru",
        rd_height=16,
        rd_width=16,
        md_bins=16,
        radar_dim=16,
        motion_dim=8,
        fusion_heads=4,
        sensor_vocab_size=2,
        md_gru_dim=16,
        dropout=0.0,
    )


def test_parrad_stage1_loss_ignores_padding_labels() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    logits = torch.tensor([[[5.0, -5.0], [-5.0, 5.0]]])
    labels = torch.tensor([[0, -100]])

    loss = stage.frame_classification_loss(logits, labels)

    assert loss < 0.001


def test_parrad_stage1_loss_rejects_batches_without_supervision() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    with pytest.raises(ValueError, match="supervised frame"):
        stage.frame_classification_loss(torch.randn(1, 2, 3), torch.full((1, 2), -100))


def test_token_targets_ignore_mixed_or_masked_updates() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    labels = torch.tensor([[0, 0, 1, 2], [1, -100, 2, 2]])

    targets = stage.token_classification_targets(labels, frames_per_update=2)

    torch.testing.assert_close(targets, torch.tensor([[0, -100], [-100, 2]]))


def test_stage1_token_supervision_reaches_radar_token_adapter() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    sample = batch()
    pure_label_batch = PARradBatch(
        rd=sample.rd,
        md=sample.md,
        labels=torch.tensor([[0, 0], [1, 1]]),
        timestamps=sample.timestamps,
        frame_mask=sample.frame_mask,
        sensor_index=sample.sensor_index,
        metadata=sample.metadata,
    )
    module = stage.PARradStage1Module(
        PARradCausalMDGRUEncoder(md_gru_config()),
        num_classes=2,
        frame_classification_weight=0.0,
        token_classification_weight=1.0,
    )

    losses = module.training_step(pure_label_batch)
    losses.loss.backward()

    assert torch.isfinite(losses.token_classification_loss)
    assert 0 <= losses.token_accuracy <= 1
    query_gradient = module.encoder.token_resampler.queries.grad
    projection_gradient = module.encoder.token_projection.weight.grad
    assert query_gradient is not None and query_gradient.count_nonzero() > 0
    assert projection_gradient is not None and projection_gradient.count_nonzero() > 0


def test_temporal_smoothness_penalizes_switches_inside_one_label_run() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    stable = torch.tensor([[[8.0, -8.0], [8.0, -8.0]]])
    flicker = torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]])
    labels = torch.tensor([[0, 0]])

    stable_loss = stage.boundary_aware_temporal_smoothness_loss(stable, labels)
    flicker_loss = stage.boundary_aware_temporal_smoothness_loss(flicker, labels)

    assert stable_loss < 1e-6
    assert flicker_loss > 0.6


def test_temporal_smoothness_ignores_true_and_masked_boundaries() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0], [8.0, -8.0]]])

    true_boundary = stage.boundary_aware_temporal_smoothness_loss(
        logits[:, :2], torch.tensor([[0, 1]])
    )
    masked_boundary = stage.boundary_aware_temporal_smoothness_loss(
        logits, torch.tensor([[0, -100, 0]])
    )

    assert true_boundary == 0
    assert masked_boundary == 0


def test_stage1_loss_adds_configured_temporal_smoothness() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    module = stage.PARradStage1Module(
        PARradRadarEncoder(tiny_config()),
        num_classes=2,
        temporal_smoothness_weight=0.5,
    )
    logits = torch.tensor([[[8.0, -8.0], [-8.0, 8.0]]])
    labels = torch.tensor([[0, 0]])

    metrics = module.loss_from_logits(logits, labels)

    torch.testing.assert_close(
        metrics.loss,
        metrics.classification_loss + 0.5 * metrics.temporal_smoothness_loss,
    )
    assert metrics.temporal_smoothness_loss > 0.6


def test_parrad_stage1_optimizer_step_updates_encoder() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    module = stage.PARradStage1Module(PARradRadarEncoder(tiny_config()), num_classes=2)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)
    before = module.encoder.rd_spatial.stem[0].weight.detach().clone()

    metrics = module.training_step(batch())
    optimizer.zero_grad(set_to_none=True)
    metrics.loss.backward()
    optimizer.step()

    assert torch.isfinite(metrics.loss)
    assert 0 <= metrics.accuracy <= 1
    assert not torch.equal(before, module.encoder.rd_spatial.stem[0].weight)


def test_parrad_stage1_forward_accepts_carried_stream_state() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    module = stage.PARradStage1Module(PARradRadarEncoder(tiny_config()), num_classes=2).eval()
    first_batch = batch()
    second_batch = batch()

    _, first = module(first_batch)
    logits, second = module(second_batch, state=first.state)

    assert logits.shape == (2, 2, 2)
    assert first.state.update_index == 1
    assert second.state.update_index == 2


def test_parrad_stage1_export_excludes_temporary_classifier() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage1")
    module = stage.PARradStage1Module(PARradRadarEncoder(tiny_config()), num_classes=2)

    exported = module.export_encoder_state()

    assert exported
    assert all("classifier" not in name for name in exported)


def parrad_stage2_module(*, adapter_trainable: bool = False):
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage2")
    model_config = ModelConfig(
        radar_dim=16,
        fusion_heads=4,
        llm_dim=32,
        llm_layers=4,
        llm_heads=4,
        cross_attn_dim=8,
        cross_attn_heads=2,
        semantic_dim=8,
        dropout=0.0,
    )
    adapter = TinyLLMAdapter(64, 32, 4, 4)
    if adapter_trainable:
        next(adapter.parameters()).requires_grad_(True)
    event_former = StreamingSemanticEventFormer(adapter, model_config, torch.tensor([1, 2, 3]))
    return stage.PARradStage2Module(
        PARradRadarEncoder(tiny_config()),
        event_former,
        SemanticProjection(32, 8),
        contrastive_temperature=0.1,
        checkpoint_streaming=False,
        num_classes=2,
    )


def test_parrad_stage2_has_stream_commit_and_three_losses() -> None:
    module = parrad_stage2_module()

    output = module.training_step(
        stage2_batch(),
        torch.tensor([[4, 5, 6], [7, 8, 9]]),
        torch.ones(2, 3, dtype=torch.bool),
    )

    assert output.stream_event_tokens.shape == output.commit_event_tokens.shape == (2, 8, 32)
    assert set(output.losses.__dict__) == {"text", "align", "consistency"}
    assert torch.isfinite(output.losses.total(1.0, 1.0))
    assert output.stream_state_norms.shape == (2, 1)


def test_supervised_alignment_treats_duplicate_labels_as_positives() -> None:
    from continuous_mmwave.stages.stage2 import (
        supervised_symmetric_alignment_loss,
        symmetric_alignment_loss,
    )

    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    labels = torch.tensor([2, 2, 7])

    supervised = supervised_symmetric_alignment_loss(keys, keys, labels, 0.07)
    diagonal_only = symmetric_alignment_loss(keys, keys, 0.07)

    assert torch.isfinite(supervised)
    assert supervised < diagonal_only


def test_parrad_event_targets_require_one_label_per_run() -> None:
    from continuous_mmwave.stages.parrad_stage2 import parrad_event_targets

    labels = torch.tensor([[0, 0, -100], [1, 1, 1]])
    mask = torch.tensor([[True, True, False], [True, True, True]])

    torch.testing.assert_close(parrad_event_targets(labels, mask), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="single action label"):
        parrad_event_targets(torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool))


def test_semantic_warmup_freezes_radar_and_adapter_only() -> None:
    from continuous_mmwave.stages.parrad_stage2 import set_parrad_stage2_phase

    module = parrad_stage2_module(adapter_trainable=True)

    set_parrad_stage2_phase(module, "semantic_warmup")

    assert not any(parameter.requires_grad for parameter in module.radar_encoder.parameters())
    assert not any(parameter.requires_grad for parameter in module.event_former.adapter.parameters())
    assert any(parameter.requires_grad for parameter in module.event_former.cross_attention.parameters())
    assert all(parameter.requires_grad for parameter in module.auxiliary_classifier.parameters())

    set_parrad_stage2_phase(module, "joint")
    assert not any(parameter.requires_grad for parameter in module.radar_encoder.parameters())
    assert any(parameter.requires_grad for parameter in module.event_former.adapter.parameters())


def test_parrad_stage2_exposes_disposable_action_logits() -> None:
    module = parrad_stage2_module()
    sample = stage2_batch()

    output = module.training_step(
        sample,
        torch.tensor([[1, 2, 3], [1, 4, 3]]),
        torch.ones(2, 3, dtype=torch.bool),
    )

    assert output.stream_class_logits is not None
    assert output.commit_class_logits is not None
    assert output.stream_class_logits.shape == output.commit_class_logits.shape == (2, 2)


def test_semantic_warmup_optimizes_action_head_without_changing_radar() -> None:
    from continuous_mmwave.stages.parrad_stage2 import set_parrad_stage2_phase
    from continuous_mmwave.train import train_stage2_batch

    module = parrad_stage2_module()
    set_parrad_stage2_phase(module, "semantic_warmup")
    optimizer = torch.optim.AdamW((p for p in module.parameters() if p.requires_grad), lr=1e-2)
    radar_before = {
        name: parameter.detach().clone() for name, parameter in module.radar_encoder.named_parameters()
    }
    alpha_before = module.event_former.cross_attention[0].alpha.detach().clone()

    metrics = train_stage2_batch(
        module,
        stage2_batch(),
        torch.tensor([[1, 2, 3], [1, 4, 3]]),
        torch.ones(2, 3, dtype=torch.bool),
        optimizer,
        1.0,
    )

    assert metrics["text"] == 0.0
    assert metrics["align"] == 0.0
    assert torch.isfinite(torch.tensor(metrics["auxiliary_classification_loss"]))
    assert 0.0 <= metrics["auxiliary_accuracy"] <= 1.0
    assert metrics["optimizer_steps"] == 1.0
    assert not torch.equal(alpha_before, module.event_former.cross_attention[0].alpha)
    for name, parameter in module.radar_encoder.named_parameters():
        torch.testing.assert_close(parameter, radar_before[name], rtol=0, atol=0)


def test_parrad_stage2_two_steps_update_encoder_and_projection() -> None:
    module = parrad_stage2_module()
    optimizer = torch.optim.AdamW((parameter for parameter in module.parameters() if parameter.requires_grad), lr=1e-3)
    encoder_before = module.radar_encoder.token_resampler.queries.detach().clone()
    projection_before = module.semantic_projection.semantic_query.detach().clone()

    for _ in range(2):
        output = module.training_step(
            stage2_batch(),
            torch.tensor([[4, 5, 6], [7, 8, 9]]),
            torch.ones(2, 3, dtype=torch.bool),
        )
        optimizer.zero_grad(set_to_none=True)
        output.losses.total(module.align_weight, module.consistency_weight).backward()
        optimizer.step()

    assert not torch.equal(encoder_before, module.radar_encoder.token_resampler.queries)
    assert not torch.equal(projection_before, module.semantic_projection.semantic_query)


def test_event_caption_templates_rotate_and_reject_mixed_events() -> None:
    stage = importlib.import_module("continuous_mmwave.stages.parrad_stage2")
    templates = {
        0: ("zero-a", "zero-b", "zero-c"),
        1: ("one-a", "one-b", "one-c"),
    }

    captions = stage.select_event_captions(
        torch.tensor([[0, 0], [1, 1]]),
        torch.ones(2, 2, dtype=torch.bool),
        templates,
        epoch=1,
    )

    assert captions == ("zero-b", "one-c")
    with pytest.raises(ValueError, match="exactly one activity label"):
        stage.select_event_captions(
            torch.tensor([[0, 1]]), torch.ones(1, 2, dtype=torch.bool), templates, epoch=0
        )


def test_default_parrad_templates_cover_every_activity_with_three_variants() -> None:
    from continuous_mmwave.stages.parrad_stage2 import default_parrad_event_templates

    templates = default_parrad_event_templates()

    assert set(templates) == set(range(14))
    assert all(len(variants) == 3 for variants in templates.values())
    assert all(len(set(variants)) == 3 for variants in templates.values())
    assert all(text.strip() for variants in templates.values() for text in variants)
