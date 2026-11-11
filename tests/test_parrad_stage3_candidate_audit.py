from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from continuous_mmwave.parrad_stage3_candidate_audit import (
    CandidateFeatureRow,
    _bootstrap_recordings,
    candidate_diagnostic,
    candidate_feature_rows,
    collect_candidate_episodes,
    feature_distribution,
    fit_candidate_probe,
    score_frozen_verify,
    score_binary,
)
from continuous_mmwave.parrad_stage3_data import Stage3CacheItem, Stage3CacheRecord
from continuous_mmwave.parrad_stage3_diagnostics import ControllerThresholds


class _ScriptedController:
    update_period_s = 0.18

    def __init__(self, decisions: tuple[str, ...]) -> None:
        self.decisions = decisions
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def step_tokens(self, radar_tokens, radar_mask, timestamp, metadata=None):
        del radar_tokens, radar_mask, timestamp, metadata
        index = self.index
        decision = self.decisions[index]
        self.index += 1
        candidate_update = index >= 2 and decision != "proposal"
        return SimpleNamespace(
            decision=decision,
            difference_probability=(0.2 * index if candidate_update else None),
            stability=(0.1 * index if candidate_update else None),
            candidate_maturity=(min(index / 2, 1.0) if candidate_update else None),
            candidate_proposal_time=(0.18 if candidate_update else None),
            verify_probability=(0.3 if candidate_update else None),
        )


def _record(labels: list[int], *, key: str = "recording", sensor: str = "xWR14xx") -> Stage3CacheItem:
    count = len(labels)
    return Stage3CacheItem(
        split="validation",
        key=key,
        metadata={"subject_id": "subject_05", "session_id": "set000", "sensor_id": sensor},
        record=Stage3CacheRecord(
            radar_tokens=np.ones((count, 2, 4), dtype=np.float16),
            radar_mask=np.ones((count, 2), dtype=np.bool_),
            frame_labels=np.asarray(labels, dtype=np.int64)[:, None],
            timestamps=(np.arange(count, dtype=np.float64) * 0.18)[:, None],
            oracle_keys=np.ones((count, 4), dtype=np.float16),
            oracle_maturity=np.ones(count, dtype=np.float32),
            oracle_run_id=np.arange(count, dtype=np.int64),
        ),
    )


_THRESHOLDS = ControllerThresholds(0.55, 0.45, 0.2, 6, "grid", 2)


def test_collects_one_episode_with_ordered_updates_and_ground_truth_identity() -> None:
    runtime = SimpleNamespace(controller=_ScriptedController(("none", "proposal", "none", "none", "confirm")))

    episodes = collect_candidate_episodes(
        runtime, (_record([0, 0, 1, 1, 1]),), _THRESHOLDS, boundary_tolerance_s=0.2
    )

    assert len(episodes) == 1
    episode = episodes[0]
    assert (episode.recording_key, episode.sensor_id, episode.proposal_time) == (
        "recording", "xWR14xx", 0.18
    )
    assert episode.terminal_reason == "confirm"
    assert [update.timestamp for update in episode.updates] == pytest.approx([0.36, 0.54, 0.72])
    assert [update.maturity for update in episode.updates] == pytest.approx([1.0, 1.0, 1.0])
    assert episode.nearest_boundary_time == pytest.approx(0.36)
    assert episode.nearest_boundary_distance_s == pytest.approx(0.18)
    assert episode.proximity_target is True
    assert episode.gt_run_at_proposal == 0
    assert episode.gt_run_at_terminal == 1


def test_same_proposal_time_in_two_recordings_remains_two_episodes() -> None:
    runtime = SimpleNamespace(controller=_ScriptedController(("none", "proposal", "confirm")))

    episodes = collect_candidate_episodes(
        runtime,
        (_record([0, 0, 1], key="first"), _record([0, 0, 1], key="second", sensor="xWR68xx")),
        _THRESHOLDS,
        boundary_tolerance_s=0.2,
    )

    assert len(episodes) == 2
    assert {(episode.recording_key, episode.proposal_time) for episode in episodes} == {
        ("first", 0.18), ("second", 0.18)
    }


def test_unfinished_candidate_is_explicit_and_ignored_gap_is_not_a_boundary() -> None:
    runtime = SimpleNamespace(controller=_ScriptedController(("none", "proposal", "none", "none")))

    (episode,) = collect_candidate_episodes(
        runtime, (_record([0, 0, -100, 1]),), _THRESHOLDS, boundary_tolerance_s=0.72
    )

    assert episode.terminal_reason == "recording_end"
    assert len(episode.updates) == 2
    assert episode.nearest_boundary_time is None
    assert episode.nearest_boundary_distance_s is None
    assert episode.proximity_target is False
    assert episode.gt_run_at_terminal == 1


def test_first_and_last_snapshot_each_count_one_candidate() -> None:
    runtime = SimpleNamespace(controller=_ScriptedController(("none", "proposal", "none", "none", "confirm")))
    episodes = collect_candidate_episodes(
        runtime, (_record([0, 0, 1, 1, 1]),), _THRESHOLDS, boundary_tolerance_s=0.2
    )

    first = candidate_feature_rows(episodes, position="first", boundary_tolerance_s=0.2)
    last = candidate_feature_rows(episodes, position="last", boundary_tolerance_s=0.2)

    assert len(first) == len(last) == 1
    assert first[0].inputs == pytest.approx((0.4, 0.2, 1.0))
    assert last[0].inputs == pytest.approx((0.8, 0.4, 1.0))
    assert first[0].target is True and last[0].target is True


def test_single_class_sensor_auc_is_undefined() -> None:
    result = score_binary([True, True], [0.2, 0.8])

    assert result["count"] == 2
    assert result["positive_prevalence"] == 1.0
    assert result["auroc"] is None
    assert result["average_precision"] == 1.0


def test_probe_standardizes_using_train_only_and_keeps_one_target_per_row() -> None:
    train = (
        CandidateFeatureRow("train", "a", "xWR14xx", 0.0, (0.0, 0.0, 0.0), False),
        CandidateFeatureRow("train", "b", "xWR14xx", 0.0, (2.0, 2.0, 2.0), True),
    )
    validation = (
        CandidateFeatureRow("validation", "c", "xWR68xx", 0.0, (100.0, 100.0, 100.0), False),
        CandidateFeatureRow("validation", "d", "xWR68xx", 0.0, (101.0, 101.0, 101.0), True),
    )

    result = fit_candidate_probe(train, validation, hidden_dim=0, epochs=2, seed=42)

    assert result["train_count"] == 2
    assert result["validation_count"] == 2
    assert result["train_mean"] == pytest.approx([1.0, 1.0, 1.0])
    assert result["train_std"] == pytest.approx([1.0, 1.0, 1.0])
    assert result["validation_metrics"]["count"] == 2


def test_collector_refuses_test_split_before_replay() -> None:
    runtime = SimpleNamespace(controller=_ScriptedController(("none", "proposal", "confirm")))
    test_item = _record([0, 0, 1])
    test_item = Stage3CacheItem("test", test_item.key, test_item.metadata, test_item.record)

    with pytest.raises(ValueError, match="train or validation"):
        collect_candidate_episodes(runtime, (test_item,), _THRESHOLDS)
    assert runtime.controller.index == 0


def test_diagnostic_preserves_train_validation_and_undefined_sensor_auc() -> None:
    runtime = SimpleNamespace(controller=_ScriptedController(("none", "proposal", "confirm")))
    first = _record([0, 0, 1], key="positive", sensor="xWR14xx")
    second = _record([0, 0, 0], key="negative", sensor="xWR68xx")
    train_records = (
        Stage3CacheItem("train", first.key, first.metadata, first.record),
        Stage3CacheItem("train", second.key, second.metadata, second.record),
    )
    train = collect_candidate_episodes(runtime, train_records, _THRESHOLDS)
    validation = collect_candidate_episodes(runtime, (first, second), _THRESHOLDS)

    report = candidate_diagnostic(train, validation)

    assert report["source_split"] == ["train", "validation"]
    assert report["test_evaluation_allowed"] is False
    assert report["episodes"]["validation"]["count"] == 2
    assert report["tolerances"]["0.36"]["first"]["validation_candidates"] == 2
    assert (
        report["tolerances"]["0.36"]["first"]["probes"]["linear"]
        ["validation_sensor_metrics"]["xWR68xx"]["auroc"] is None
    )


def test_feature_distribution_separates_classes_and_sensors() -> None:
    rows = (
        CandidateFeatureRow("validation", "a", "xWR14xx", 0.0, (0.2, 0.1, 0.5), True),
        CandidateFeatureRow("validation", "b", "xWR14xx", 0.0, (0.4, 0.3, 0.7), True),
        CandidateFeatureRow("validation", "c", "xWR14xx", 0.0, (0.8, 0.6, 1.0), False),
        CandidateFeatureRow("validation", "d", "xWR68xx", 0.0, (0.6, 0.4, 0.9), False),
    )

    result = feature_distribution(rows)

    assert result["overall"]["positive"]["difference_probability"]["median"] == pytest.approx(0.3)
    assert result["overall"]["negative"]["difference_probability"]["median"] == pytest.approx(0.7)
    assert result["by_sensor"]["xWR14xx"]["positive"]["count"] == 2
    assert result["by_sensor"]["xWR68xx"]["positive"]["count"] == 0


def test_frozen_verify_scores_candidates_without_fitting() -> None:
    rows = (
        CandidateFeatureRow("validation", "a", "xWR14xx", 0.0, (0.0, 0.0, 0.0), False),
        CandidateFeatureRow("validation", "b", "xWR14xx", 0.0, (1.0, 0.0, 0.0), True),
    )
    head = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))

    result = score_frozen_verify(rows, head)

    assert result["validation_metrics"]["auroc"] == 1.0
    assert result["validation_metrics"]["average_precision"] == 1.0
    assert result["validation_metrics"]["count"] == 2


def test_bootstrap_keeps_paired_sensor_views_in_one_session() -> None:
    rows = (
        CandidateFeatureRow("validation", "subject_05__set000__TI_xWR14xx", "TI_xWR14xx", 0.0, (0.0, 0.0, 0.0), False),
        CandidateFeatureRow("validation", "subject_05__set000__TI_xWR68xx", "TI_xWR68xx", 0.0, (1.0, 0.0, 0.0), True),
        CandidateFeatureRow("validation", "subject_05__set001__TI_xWR14xx", "TI_xWR14xx", 0.0, (0.2, 0.0, 0.0), False),
        CandidateFeatureRow("validation", "subject_05__set001__TI_xWR68xx", "TI_xWR68xx", 0.0, (0.8, 0.0, 0.0), True),
    )

    result = _bootstrap_recordings(rows, [0.1, 0.9, 0.2, 0.8], repetitions=10)

    assert result["recordings"] == 2
    assert result["valid_auc_repetitions"] == 10
