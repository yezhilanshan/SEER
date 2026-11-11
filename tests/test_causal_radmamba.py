from __future__ import annotations

import json

import numpy as np
import torch

from continuous_mmwave.parrad_config import PARradConfig


def build_encoder():
    from continuous_mmwave.models.radmamba_encoder import CausalRadMambaEncoder

    return CausalRadMambaEncoder(
        md_bins=16,
        model_dim=8,
        output_dim=16,
        d_state=4,
        dt_rank=2,
        depth=1,
        channel_dim=2,
        dropout=0.0,
    ).eval()


def build_single_channel_encoder():
    from continuous_mmwave.models.radmamba_encoder import CausalRadMambaEncoder

    return CausalRadMambaEncoder(
        md_bins=16,
        model_dim=8,
        output_dim=16,
        d_state=4,
        dt_rank=2,
        depth=1,
        channel_dim=1,
        dropout=0.0,
    ).eval()


def test_single_channel_frontend_preserves_md_input_sensitivity() -> None:
    torch.manual_seed(3)
    encoder = build_single_channel_encoder()
    left_peak = torch.zeros(1, 4, 16)
    right_peak = torch.zeros(1, 4, 16)
    left_peak[..., 2] = 1
    right_peak[..., 13] = 1

    left_hidden = encoder(left_peak).frame_hidden
    right_hidden = encoder(right_peak).frame_hidden

    assert not torch.allclose(left_hidden, right_hidden, atol=1e-6, rtol=1e-5)


def test_single_channel_frontend_propagates_md_gradients() -> None:
    torch.manual_seed(5)
    encoder = build_single_channel_encoder()
    micro_doppler = torch.randn(1, 4, 16, requires_grad=True)

    encoder(micro_doppler).frame_hidden.square().sum().backward()

    assert micro_doppler.grad is not None
    assert micro_doppler.grad.abs().max() > 1e-8


def test_future_md_cannot_change_past_features() -> None:
    torch.manual_seed(11)
    encoder = build_encoder()
    original = torch.randn(2, 6, 16)
    changed = original.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100

    baseline = encoder(original)
    perturbed = encoder(changed)

    torch.testing.assert_close(
        baseline.frame_hidden[:, :3],
        perturbed.frame_hidden[:, :3],
    )


def test_chunked_md_matches_one_continuous_forward_pass() -> None:
    torch.manual_seed(13)
    encoder = build_encoder()
    sequence = torch.randn(2, 6, 16)

    full = encoder(sequence)
    first = encoder(sequence[:, :2])
    second = encoder(sequence[:, 2:], state=first.state)

    torch.testing.assert_close(
        torch.cat((first.frame_hidden, second.frame_hidden), dim=1),
        full.frame_hidden,
        atol=1e-6,
        rtol=1e-5,
    )
    assert torch.equal(second.state.position, full.state.position)


def test_padding_does_not_advance_stream_state() -> None:
    torch.manual_seed(17)
    encoder = build_encoder()
    sequence = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

    output = encoder(sequence, frame_mask=mask)
    first_only = encoder(sequence[:1, :2])

    assert output.frame_hidden[0, 2:].count_nonzero() == 0
    assert output.state.position.tolist() == [2, 4]
    torch.testing.assert_close(output.state.frontend_cache[0], first_only.state.frontend_cache[0])
    torch.testing.assert_close(output.state.layers[0].conv_cache[0], first_only.state.layers[0].conv_cache[0])
    torch.testing.assert_close(output.state.layers[0].ssm_state[0], first_only.state.layers[0].ssm_state[0])


def radmamba_parrad_config() -> PARradConfig:
    return PARradConfig(
        encoder_type="causal_radmamba",
        rd_height=16,
        rd_width=16,
        md_bins=16,
        radar_dim=16,
        motion_dim=8,
        fusion_heads=4,
        sensor_vocab_size=2,
        dropout=0.0,
        radmamba_dim=8,
        radmamba_d_state=4,
        radmamba_dt_rank=2,
        radmamba_depth=1,
        radmamba_channels=2,
    )


def test_parrad_factory_selects_md_only_radmamba_contract() -> None:
    from continuous_mmwave.models.parrad_encoder import (
        PARradCausalRadMambaEncoder,
        build_parrad_encoder,
    )

    torch.manual_seed(19)
    encoder = build_parrad_encoder(radmamba_parrad_config()).eval()
    micro_doppler = torch.randn(2, 2, 16)
    sensors = torch.tensor([0, 1])
    first_rd = torch.randn(2, 2, 1, 16, 16)
    changed_rd = torch.randn_like(first_rd) * 1000

    baseline = encoder.forward_update(first_rd, micro_doppler, sensor_index=sensors)
    perturbed = encoder.forward_update(changed_rd, micro_doppler, sensor_index=sensors)

    assert isinstance(encoder, PARradCausalRadMambaEncoder)
    assert baseline.radar_tokens.shape == (2, 4, 16)
    assert baseline.frame_hidden.shape == (2, 2, 16)
    assert baseline.motion_state.shape == (2, 8)
    torch.testing.assert_close(baseline.radar_tokens, perturbed.radar_tokens)
    torch.testing.assert_close(baseline.frame_hidden, perturbed.frame_hidden)


def test_parrad_radmamba_stream_state_detaches_without_resetting_time() -> None:
    from continuous_mmwave.models.parrad_encoder import (
        build_parrad_encoder,
        detach_parrad_stream_state,
    )

    encoder = build_parrad_encoder(radmamba_parrad_config()).train()
    state = encoder.forward_update(
        torch.randn(1, 2, 1, 16, 16),
        torch.randn(1, 2, 16),
    ).state

    detached = detach_parrad_stream_state(state)

    assert detached.update_index == state.update_index
    assert detached.radmamba.position.tolist() == [2]
    assert detached.radmamba.frontend_cache.grad_fn is None
    assert detached.radmamba.layers[0].conv_cache.grad_fn is None
    assert detached.radmamba.layers[0].ssm_state.grad_fn is None


def test_stage1_builder_uses_configured_radmamba_backend() -> None:
    from continuous_mmwave.models.parrad_encoder import PARradCausalRadMambaEncoder
    from continuous_mmwave.train import ExperimentConfig, _build_parrad_stage1

    experiment = ExperimentConfig(
        model=radmamba_parrad_config(),
        data={},
        training={},
        llm={},
        stage={"num_classes": 3, "label_run_temporal_smoothness_weight": 0.02},
    )

    stage = _build_parrad_stage1(experiment)

    assert isinstance(stage.encoder, PARradCausalRadMambaEncoder)
    assert stage.temporal_smoothness_weight == 0.02


def test_md_only_dataset_skips_rd_payload(tmp_path) -> None:
    from continuous_mmwave.data_parrad import PARradManifestDataset, parrad_collate

    recording = tmp_path / "recording.npz"
    np.savez_compressed(
        recording,
        rd=np.random.default_rng(3).normal(size=(3, 1, 16, 16)).astype(np.float32),
        md=np.random.default_rng(5).normal(size=(3, 16)).astype(np.float32),
        labels=np.array([0, 0, 1], dtype=np.int64),
        timestamps=np.arange(3, dtype=np.float64) * 0.09,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "path": recording.name,
                "subject_id": "s1",
                "session_id": "set000",
                "sensor_id": "60ghz",
                "environment_id": "lab",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    sample = PARradManifestDataset(manifest, radmamba_parrad_config())[0]
    batch = parrad_collate((sample,), frames_per_update=2)

    assert sample.rd.shape == (3, 0, 0, 0)
    assert sample.md.shape == (3, 16)
    assert batch.rd.shape == (1, 4, 0, 0, 0)
    assert batch.md.shape == (1, 4, 16)
