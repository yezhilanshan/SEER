from __future__ import annotations

import json

import numpy as np
import torch

from continuous_mmwave.parrad_config import PARradConfig


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
        dropout=0.0,
        md_gru_dim=16,
    )


def test_md_gru_is_causal_and_chunk_equivalent() -> None:
    from continuous_mmwave.models.md_gru_encoder import CausalMDGRUEncoder

    torch.manual_seed(41)
    encoder = CausalMDGRUEncoder(md_bins=16, hidden_dim=16, dropout=0.0).eval()
    sequence = torch.randn(2, 6, 16)
    changed = sequence.clone()
    changed[:, 3:] = torch.randn_like(changed[:, 3:]) * 100

    full = encoder(sequence)
    perturbed = encoder(changed)
    first = encoder(sequence[:, :2])
    second = encoder(sequence[:, 2:], state=first.state)

    torch.testing.assert_close(full.frame_hidden[:, :3], perturbed.frame_hidden[:, :3])
    torch.testing.assert_close(
        torch.cat((first.frame_hidden, second.frame_hidden), dim=1),
        full.frame_hidden,
        atol=1e-6,
        rtol=1e-5,
    )


def test_md_gru_padding_does_not_advance_hidden_state() -> None:
    from continuous_mmwave.models.md_gru_encoder import CausalMDGRUEncoder

    torch.manual_seed(43)
    encoder = CausalMDGRUEncoder(md_bins=16, hidden_dim=16, dropout=0.0).eval()
    sequence = torch.randn(2, 4, 16)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])

    output = encoder(sequence, frame_mask=mask)
    first_only = encoder(sequence[:1, :2])

    assert output.frame_hidden[0, 2:].count_nonzero() == 0
    torch.testing.assert_close(output.state.hidden[:, :1], first_only.state.hidden)


def test_parrad_factory_builds_stateful_md_gru_and_ignores_rd() -> None:
    from continuous_mmwave.models.parrad_encoder import (
        PARradCausalMDGRUEncoder,
        build_parrad_encoder,
        detach_parrad_stream_state,
    )

    torch.manual_seed(47)
    encoder = build_parrad_encoder(md_gru_config()).eval()
    md = torch.randn(2, 2, 16)
    sensors = torch.tensor([0, 1])
    placeholder_rd = torch.empty(2, 2, 0, 0, 0)

    first = encoder.forward_update(placeholder_rd, md, sensor_index=sensors)
    second = encoder.forward_update(placeholder_rd, md, sensor_index=sensors, state=first.state)
    detached = detach_parrad_stream_state(second.state)

    assert isinstance(encoder, PARradCausalMDGRUEncoder)
    assert first.frame_hidden.shape == (2, 2, 16)
    assert first.radar_tokens.shape == (2, 4, 16)
    assert second.state.update_index == 2
    assert detached.gru.hidden.grad_fn is None


def test_md_gru_dataset_does_not_load_rd_payload(tmp_path) -> None:
    from continuous_mmwave.data_parrad import PARradManifestDataset

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

    sample = PARradManifestDataset(manifest, md_gru_config())[0]

    assert sample.rd.shape == (3, 0, 0, 0)
    assert sample.md.shape == (3, 16)
