from __future__ import annotations

import pytest
import torch
from torch import nn


def test_masked_token_mean_excludes_padding_and_zeroes_empty_rows() -> None:
    from continuous_mmwave.parrad_stage2_diagnostics import masked_token_mean

    tokens = torch.tensor(
        [
            [[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]],
            [[7.0, 9.0], [11.0, 13.0], [17.0, 19.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [False, False, False]])

    pooled = masked_token_mean(tokens, mask)

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 4.0], [0.0, 0.0]]))


def test_gate_summary_reports_bounded_update_distribution() -> None:
    from continuous_mmwave.parrad_stage2_diagnostics import summarize_gate_values

    summary = summarize_gate_values(torch.tensor([[[0.1], [0.5]], [[0.9], [0.2]]]))

    assert summary == pytest.approx({
        "min": 0.1,
        "mean": 0.425,
        "max": 0.9,
        "near_zero_fraction": 0.0,
        "near_one_fraction": 0.0,
    })


def test_action_classification_diagnostics_compare_stream_and_commit() -> None:
    from continuous_mmwave.parrad_stage2_diagnostics import action_classification_diagnostics

    summary = action_classification_diagnostics(
        torch.tensor([[3.0, 0.0], [0.0, 3.0]]),
        torch.tensor([[0.0, 3.0], [0.0, 3.0]]),
        torch.tensor([0, 1]),
    )

    assert summary["stream_accuracy"] == 1.0
    assert summary["commit_accuracy"] == 0.5
    assert summary["mean_accuracy"] == 0.75
    assert summary["classification_loss"] > 0


def test_checkpoint_probe_keeps_joint_adapter_in_optimizer_union() -> None:
    from continuous_mmwave.parrad_stage2_diagnostics import _checkpoint_probe_trainable

    class ProbeModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.radar_encoder = nn.Linear(2, 2)
            self.event_former = nn.Module()
            self.event_former.adapter = nn.Linear(2, 2)
            self.projection = nn.Linear(2, 2)
            self._joint_adapter_parameter_ids = frozenset(
                id(parameter) for parameter in self.event_former.adapter.parameters()
            )

    module = ProbeModule()
    module.radar_encoder.requires_grad_(False)
    module.event_former.adapter.requires_grad_(False)

    parameters = _checkpoint_probe_trainable(module)
    parameter_ids = {id(parameter) for parameter in parameters}

    assert all(id(parameter) not in parameter_ids for parameter in module.radar_encoder.parameters())
    assert all(id(parameter) in parameter_ids for parameter in module.event_former.adapter.parameters())
    assert all(id(parameter) in parameter_ids for parameter in module.projection.parameters())


def test_checkpoint_probe_scaler_uses_configured_initial_scale(monkeypatch) -> None:
    from continuous_mmwave.parrad_stage2_diagnostics import _checkpoint_probe_scaler

    captured = {}

    def fake_scaler(device, *, init_scale):
        captured.update(device=device, init_scale=init_scale)
        return object()

    monkeypatch.setattr(torch.amp, "GradScaler", fake_scaler)

    scaler = _checkpoint_probe_scaler({"amp_init_scale": 512}, amp=True)

    assert scaler is not None
    assert captured == {"device": "cuda", "init_scale": 512.0}


def test_cross_attention_gradient_rows_keep_every_projection() -> None:
    from continuous_mmwave.parrad_stage2_diagnostics import cross_attention_gradient_rows

    rows = [
        {"name": "event_former.cross_attention.0.query.weight", "l2_norm": 1.0},
        {"name": "event_former.cross_attention.0.key.weight", "l2_norm": 2.0},
        {"name": "event_former.cross_attention.0.value.weight", "l2_norm": 3.0},
        {"name": "event_former.cross_attention.0.output.weight", "l2_norm": 4.0},
        {"name": "event_former.state_gate.network.2.weight", "l2_norm": 5.0},
    ]

    selected = cross_attention_gradient_rows(rows)

    assert [row["l2_norm"] for row in selected] == [1.0, 2.0, 3.0, 4.0]
