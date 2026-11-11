from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from continuous_mmwave.parrad_stage3_data import (
    Stage3CacheItem,
    Stage3CacheRecord,
    VerifyFeatureSample,
)
from continuous_mmwave.parrad_stage3_diagnostics import ControllerThresholds
from continuous_mmwave.parrad_stage3_hard_negative import (
    _balanced_rehearsal_samples,
    mine_closed_loop_verify_samples,
)


class _ScriptedController:
    update_period_s = 0.18

    def __init__(self) -> None:
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def step_tokens(self, radar_tokens, radar_mask, timestamp, metadata=None):
        del radar_tokens, radar_mask, timestamp, metadata
        outputs = (
            SimpleNamespace(
                difference_probability=None,
                stability=None,
                candidate_maturity=None,
                candidate_proposal_time=None,
                verify_probability=None,
                decision="proposal",
            ),
            SimpleNamespace(
                difference_probability=0.8,
                stability=0.7,
                candidate_maturity=0.5,
                candidate_proposal_time=0.18,
                verify_probability=0.9,
                decision="confirm",
            ),
            SimpleNamespace(
                difference_probability=None,
                stability=None,
                candidate_maturity=None,
                candidate_proposal_time=None,
                verify_probability=None,
                decision="none",
            ),
        )
        output = outputs[self.index]
        self.index += 1
        return output


def _record(labels: list[int]) -> Stage3CacheItem:
    updates = len(labels)
    return Stage3CacheItem(
        split="train",
        key="recording",
        metadata={"subject_id": "s1", "session_id": "r1", "sensor_id": "xWR14xx"},
        record=Stage3CacheRecord(
            radar_tokens=np.ones((updates, 2, 4), dtype=np.float16),
            radar_mask=np.ones((updates, 2), dtype=np.bool_),
            frame_labels=np.asarray(labels, dtype=np.int64)[:, None],
            timestamps=(np.arange(updates, dtype=np.float64) * 0.18)[:, None],
            oracle_keys=np.ones((updates, 4), dtype=np.float16),
            oracle_maturity=np.ones(updates, dtype=np.float32),
            oracle_run_id=np.arange(updates, dtype=np.int64),
        ),
    )


def test_closed_loop_miner_labels_proposal_against_posthoc_boundaries() -> None:
    runtime = SimpleNamespace(controller=_ScriptedController())
    thresholds = ControllerThresholds(0.55, 0.45, 0.2, 6, "grid", 2)

    positive = mine_closed_loop_verify_samples(
        runtime, (_record([0, 1, 1]),), thresholds, boundary_tolerance_s=0.18
    )
    negative = mine_closed_loop_verify_samples(
        runtime, (_record([0, 0, 0]),), thresholds, boundary_tolerance_s=0.18
    )

    assert len(positive) == len(negative) == 1
    assert positive[0].target == 1.0
    assert negative[0].target == 0.0
    assert negative[0].kind == "closed_loop_hard_negative"
    assert negative[0].inputs == pytest.approx((0.8, 0.7, 0.5))


def test_rehearsal_keeps_oracle_examples_and_balances_class_weight() -> None:
    oracle = (
        VerifyFeatureSample((0.9, 0.9, 1.0), 1.0, 1.0, "oracle_true"),
        VerifyFeatureSample((0.1, 0.1, 1.0), 0.0, 1.0, "oracle_false"),
    )
    closed = (
        VerifyFeatureSample((0.8, 0.8, 0.5), 0.0, 1.0, "closed_loop_hard_negative"),
    )

    samples = _balanced_rehearsal_samples(oracle, closed, hard_negative_weight=2.0)

    positive_weight = sum(sample.weight for sample in samples if sample.target == 1.0)
    negative_weight = sum(sample.weight for sample in samples if sample.target == 0.0)
    assert len(samples) == 3
    assert positive_weight == pytest.approx(negative_weight)
    hard = next(sample for sample in samples if sample.kind == "closed_loop_hard_negative")
    regular = next(sample for sample in samples if sample.kind == "oracle_false")
    assert hard.weight == pytest.approx(2 * regular.weight)
