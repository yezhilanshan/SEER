from __future__ import annotations

import importlib
import copy

import torch

from continuous_mmwave.config import ModelConfig
from continuous_mmwave.models.event_former import StreamingSemanticEventFormer, TinyLLMAdapter
from continuous_mmwave.parrad_config import PARradConfig


def tiny_parrad_config() -> PARradConfig:
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


def rd(batch: int = 2, frames: int = 2) -> torch.Tensor:
    return torch.randn(batch, frames, 1, 16, 16)


def md(batch: int = 2, frames: int = 2) -> torch.Tensor:
    return torch.randn(batch, frames, 16)


def test_film_is_exact_identity_at_initialization() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    film = module.MotionFiLM(motion_dim=8, channels=16)
    feature = torch.randn(4, 16, 4, 8)
    motion = torch.randn(4, 8)

    torch.testing.assert_close(film(feature, motion), feature)


def test_parrad_encoder_emits_four_token_contract() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    encoder = module.PARradRadarEncoder(tiny_parrad_config()).eval()

    output = encoder.forward_update(rd(), md(), sensor_index=torch.tensor([0, 1]))

    assert output.radar_tokens.shape == (2, 4, 16)
    assert output.semantic_state.shape == (2, 16)
    assert output.motion_state.shape == (2, 8)
    assert output.frame_hidden.shape == (2, 2, 16)
    assert output.radar_token_mask.shape == (2, 4)
    assert output.frame_mask.shape == (2, 2)
    assert torch.isfinite(output.radar_tokens).all()


def test_rd_encoder_preserves_doppler_in_first_two_reductions() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    spatial = module.RDSpatialEncoder(tiny_parrad_config()).eval()
    shapes: list[torch.Size] = []
    hooks = [
        layer.register_forward_hook(lambda _module, _inputs, output: shapes.append(output.shape))
        for layer in (spatial.stem, spatial.block1, spatial.block2)
    ]

    try:
        spatial(rd(batch=1, frames=2).flatten(0, 1))
    finally:
        for hook in hooks:
            hook.remove()

    assert [shape[-1] for shape in shapes[:2]] == [16, 16]
    assert shapes[2][-1] == 8


def test_chunked_encoder_matches_forward_sequence() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    torch.manual_seed(4)
    encoder = module.PARradRadarEncoder(tiny_parrad_config()).eval()
    rd_sequence, md_sequence = rd(frames=4), md(frames=4)
    sensors = torch.tensor([0, 1])

    full = encoder.forward_sequence(rd_sequence, md_sequence, sensor_index=sensors)
    first = encoder.forward_update(rd_sequence[:, :2], md_sequence[:, :2], sensor_index=sensors)
    second = encoder.forward_update(
        rd_sequence[:, 2:], md_sequence[:, 2:], sensor_index=sensors, state=first.state
    )

    torch.testing.assert_close(torch.cat((first.radar_tokens, second.radar_tokens), 1), full.radar_tokens)
    torch.testing.assert_close(torch.cat((first.frame_hidden, second.frame_hidden), 1), full.frame_hidden)


def test_forward_sequence_continues_an_existing_stream_state() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    torch.manual_seed(8)
    encoder = module.PARradRadarEncoder(tiny_parrad_config()).eval()
    rd_sequence, md_sequence = rd(frames=8), md(frames=8)
    sensors = torch.tensor([0, 1])

    full = encoder.forward_sequence(rd_sequence, md_sequence, sensor_index=sensors)
    first = encoder.forward_sequence(rd_sequence[:, :4], md_sequence[:, :4], sensor_index=sensors)
    second = encoder.forward_sequence(
        rd_sequence[:, 4:],
        md_sequence[:, 4:],
        sensor_index=sensors,
        state=first.state,
    )

    assert first.state.update_index == 2
    assert second.state.update_index == full.state.update_index == 4
    torch.testing.assert_close(
        torch.cat((first.frame_hidden, second.frame_hidden), 1),
        full.frame_hidden,
    )


def test_detach_parrad_stream_state_removes_graph_history() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    encoder = module.PARradRadarEncoder(tiny_parrad_config()).train()

    state = encoder.forward_sequence(rd(frames=4), md(frames=4)).state
    detached = module.detach_parrad_stream_state(state)

    assert detached.update_index == state.update_index
    assert all(not cache.requires_grad and cache.grad_fn is None for cache in detached.rd_tcn.caches)
    assert all(not cache.requires_grad and cache.grad_fn is None for cache in detached.md_tcn.caches)
    assert all(original.data_ptr() == clean.data_ptr() for original, clean in zip(state.rd_tcn.caches, detached.rd_tcn.caches))


def test_all_padding_update_is_finite_and_zero() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    encoder = module.PARradRadarEncoder(tiny_parrad_config()).eval()

    output = encoder.forward_update(rd(), md(), frame_mask=torch.zeros(2, 2, dtype=torch.bool))

    assert torch.isfinite(output.radar_tokens).all()
    assert not output.radar_token_mask.any()
    assert output.radar_tokens.count_nonzero() == 0
    assert output.frame_hidden.count_nonzero() == 0


def test_masked_padding_does_not_change_valid_frame_batch_norm_features() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    torch.manual_seed(23)
    baseline = module.PARradRadarEncoder(tiny_parrad_config()).train()
    perturbed = copy.deepcopy(baseline).train()
    radar = rd(batch=1)
    micro_doppler = md(batch=1)
    radar_perturbed = radar.clone()
    md_perturbed = micro_doppler.clone()
    radar_perturbed[:, 1] = 1000
    md_perturbed[:, 1] = -1000
    mask = torch.tensor([[True, False]])

    baseline_output = baseline.forward_update(radar, micro_doppler, frame_mask=mask)
    perturbed_output = perturbed.forward_update(
        radar_perturbed,
        md_perturbed,
        frame_mask=mask,
    )

    torch.testing.assert_close(
        baseline_output.frame_hidden[:, 0],
        perturbed_output.frame_hidden[:, 0],
    )


def test_sensor_embedding_can_be_disabled_for_cross_sensor_diagnostics() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    encoder = module.PARradRadarEncoder(tiny_parrad_config()).eval()
    radar, micro_doppler = rd(batch=1), md(batch=1)

    without_sensor = encoder.forward_update(radar, micro_doppler, sensor_index=None)
    with_first = encoder.forward_update(radar, micro_doppler, sensor_index=torch.tensor([0]))
    with_second = encoder.forward_update(radar, micro_doppler, sensor_index=torch.tensor([1]))

    assert not torch.equal(with_first.radar_tokens, with_second.radar_tokens)
    assert without_sensor.radar_tokens.shape == with_first.radar_tokens.shape


def test_parrad_tokens_feed_existing_event_former() -> None:
    module = importlib.import_module("continuous_mmwave.models.parrad_encoder")
    encoder = module.PARradRadarEncoder(tiny_parrad_config()).eval()
    output = encoder.forward_update(rd(), md())
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
    former = StreamingSemanticEventFormer(
        TinyLLMAdapter(64, 32, 4, 4), model_config, torch.tensor([1, 2, 3])
    ).eval()

    event = former.stream_update(
        former.initial_state(2), output.radar_tokens, 0.5, output.radar_token_mask
    ).event_tokens

    assert event.shape == (2, 8, 32)
    assert torch.isfinite(event).all()
