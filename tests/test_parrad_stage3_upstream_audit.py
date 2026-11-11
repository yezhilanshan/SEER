from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from continuous_mmwave.parrad_stage3_upstream_audit import (
    ControllerStepTap, audit_validation_records, proposal_attribution,
    adjacent_oracle_pair_diagnostic, semantic_counterfactuals, summarize_upstream_rows,
    normalized_cosine_difference, semantic_cosine_age_diagnostic,
)
from test_parrad_controller import _controller, _mask, _tokens


def test_upstream_tap_preserves_controller_output_and_separates_shadow_radar_call() -> None:
    controller = _controller(reject_timeout_s=10)
    with ControllerStepTap(controller) as tap:
        first = tap.step(_tokens(1), _mask(), 0.0)
        assert first.local_probability is None
        assert first.radar_probability == pytest.approx(torch.sigmoid(torch.tensor(-10.0)).item())
        proposal = tap.step(_tokens(10), _mask(), 0.18)
        assert proposal.output.decision == "proposal"
        assert proposal.local_probability == pytest.approx(torch.sigmoid(torch.tensor(10.0)).item())
        scored = tap.step(_tokens(10), _mask(), 0.36)
        assert scored.output.decision == "none"
        assert scored.radar_calls == 2
        assert scored.semantic_active is not None
        assert scored.semantic_candidate is not None
        assert scored.semantic_logit == pytest.approx(-10.0)
        assert scored.semantic_maturity == pytest.approx(1.0)


def test_semantic_counterfactuals_isolate_reference_candidate_and_maturity_shift() -> None:
    class Difference(nn.Module):
        def forward(self, active, candidate, maturity):
            return candidate[:, 0] - active[:, 0] + maturity

    result = semantic_counterfactuals(
        Difference(),
        online_reference=torch.tensor([[1.0, 0.0]]),
        online_candidate=torch.tensor([[3.0, 0.0]]),
        oracle_reference=torch.tensor([[2.0, 0.0]]),
        oracle_candidate=torch.tensor([[5.0, 0.0]]),
        online_maturity=0.5,
        oracle_maturity=1.0,
    )
    assert result["online_logit"] == pytest.approx(2.5)
    assert result["oracle_pair_online_maturity_logit"] == pytest.approx(3.5)
    assert result["online_reference_oracle_candidate_logit"] == pytest.approx(4.5)
    assert result["oracle_reference_online_candidate_logit"] == pytest.approx(1.5)
    assert result["oracle_pair_oracle_maturity_logit"] == pytest.approx(4.0)


@pytest.mark.parametrize(
    ("local", "radar", "threshold", "expected"),
    [(0.8, 0.1, 0.6, "local_only"), (0.1, 0.8, 0.6, "radar_only"),
     (0.8, 0.8, 0.6, "both"), (0.4, 0.4, 0.6, "fused_only"),
     (0.1, 0.1, 0.6, "below_threshold")],
)
def test_proposal_attribution_uses_actual_or_fusion(local, radar, threshold, expected) -> None:
    assert proposal_attribution(local, radar, threshold) == expected


def test_upstream_summary_separates_proposal_sources_and_semantic_pair_spaces() -> None:
    physical = [
        dict(sensor_id="14xx", eligible=True, boundary_distance_s=0.1,
             local_probability=0.9, radar_probability=0.1, proposal_probability=0.91,
             decision="proposal", attribution="local_only"),
        dict(sensor_id="14xx", eligible=True, boundary_distance_s=2.0,
             local_probability=0.1, radar_probability=0.9, proposal_probability=0.91,
             decision="proposal", attribution="radar_only"),
    ]
    semantic = [
        dict(candidate_id="a", sensor_id="14xx", age_updates=2, eventual_gt_run_switch=True,
             online_probability=0.2, oracle_pair_online_maturity_probability=0.8,
             online_reference_oracle_candidate_probability=0.7,
             oracle_reference_online_candidate_probability=0.3,
             oracle_pair_oracle_maturity_probability=0.9,
             reference_cosine_distance=0.2, candidate_cosine_distance=0.4),
        dict(candidate_id="b", sensor_id="14xx", age_updates=2, eventual_gt_run_switch=False,
             online_probability=0.8, oracle_pair_online_maturity_probability=0.2,
             online_reference_oracle_candidate_probability=0.3,
             oracle_reference_online_candidate_probability=0.7,
             oracle_pair_oracle_maturity_probability=0.1,
             reference_cosine_distance=0.1, candidate_cosine_distance=0.3),
    ]
    result = summarize_upstream_rows(physical, semantic, proposal_threshold=0.55)
    assert result["physical"]["far_proposals_gt_1p44s"]["count"] == 1
    assert result["physical"]["far_proposals_gt_1p44s"]["attribution"] == {"radar_only": 1}
    assert result["semantic"]["first"]["online_probability"]["auroc"] == 0.0
    assert result["semantic"]["first"]["oracle_pair_online_maturity_probability"]["auroc"] == 1.0


def test_upstream_replay_rejects_test_split_before_touching_runtime() -> None:
    with pytest.raises(ValueError, match="validation"):
        audit_validation_records(None, (SimpleNamespace(split="test"),), None)


def test_adjacent_oracle_pairs_compare_same_recording_at_matched_maturity() -> None:
    class Difference(nn.Module):
        def forward(self, active, candidate, maturity):
            return candidate[:, 0] - active[:, 0] + maturity

    record = SimpleNamespace(
        frame_labels=np.array([[0], [0], [1], [1]], dtype=np.int64),
        timestamps=np.array([[0.0], [0.18], [0.36], [0.54]], dtype=np.float64),
        oracle_run_id=np.array([0, 0, 1, 1], dtype=np.int64),
        oracle_keys=np.array([[1.0, 0.0], [2.0, 0.0], [5.0, 0.0], [6.0, 0.0]], dtype=np.float16),
        oracle_maturity=np.array([0.5, 1.0, 0.5, 1.0], dtype=np.float32),
    )
    item = SimpleNamespace(split="validation", key="one", metadata={"sensor_id": "14xx"}, record=record)
    result = adjacent_oracle_pair_diagnostic((item,), Difference())
    assert result["nascent"]["pairs"] == 1
    assert result["nascent"]["paired_win_rate"] == 1.0
    assert result["mature"]["paired_win_rate"] == 1.0


def test_adjacent_oracle_pairs_exclude_ignored_gap() -> None:
    record = SimpleNamespace(
        frame_labels=np.array([[0], [0], [-100], [1], [1]], dtype=np.int64),
        timestamps=np.array([[0.0], [0.18], [0.36], [0.54], [0.72]], dtype=np.float64),
        oracle_run_id=np.array([0, 0, -1, 1, 1], dtype=np.int64),
        oracle_keys=np.ones((5, 2), dtype=np.float16),
        oracle_maturity=np.array([0.5, 1.0, 0.0, 0.5, 1.0], dtype=np.float32),
    )
    item = SimpleNamespace(split="validation", key="gap", metadata={"sensor_id": "14xx"}, record=record)
    assert adjacent_oracle_pair_diagnostic((item,), nn.Identity())["nascent"]["pairs"] == 0


@pytest.mark.parametrize(
    ("candidate", "expected"),
    [([1.0, 0.0], 0.0), ([0.0, 1.0], 0.5), ([-1.0, 0.0], 1.0)],
)
def test_normalized_cosine_difference_has_fixed_zero_to_one_scale(candidate, expected) -> None:
    assert normalized_cosine_difference(
        torch.tensor([[1.0, 0.0]]), torch.tensor([candidate])
    ) == pytest.approx(expected)


def test_normalized_cosine_difference_excludes_zero_norm_keys() -> None:
    assert normalized_cosine_difference(torch.zeros(1, 2), torch.ones(1, 2)) is None


def test_semantic_cosine_age_diagnostic_samples_one_update_per_candidate_per_bin() -> None:
    def row(candidate, age, target, score, sensor="14xx"):
        return dict(
            candidate_id=candidate, sensor_id=sensor, recording_key="rec",
            age_s=age, eventual_gt_run_switch=target,
            reference_run_id=0, current_run_id=int(target),
            online_cosine_probability=score, online_probability=1 - score,
        )

    rows = [
        row("a", 0.18, True, 0.9), row("a", 0.36, True, 0.1),
        row("b", 0.18, False, 0.1), row("b", 0.36, False, 0.9),
        row("a", 0.54, True, 0.8), row("b", 0.54, False, 0.2),
        row("a", 0.90, True, 0.7), row("b", 0.90, False, 0.3),
        row("a", 1.26, True, 0.6), row("b", 1.26, False, 0.4),
    ]
    result = semantic_cosine_age_diagnostic(rows)
    bins = result["age_bins"]
    assert [bins[name]["candidate_count"] for name in ("lt_0p4", "0p4_to_0p8", "0p8_to_1p2", "gt_1p2")] == [2, 2, 2, 2]
    assert bins["lt_0p4"]["eventual_switch"]["cosine"]["auroc"] == 1.0
    assert bins["lt_0p4"]["eventual_switch"]["trained_head"]["auroc"] == 0.0
    assert bins["gt_1p2"]["contemporaneous_switch"]["cosine"]["auroc"] == 1.0
    assert result["exact_ages"]["0.36"]["candidate_count"] == 2
