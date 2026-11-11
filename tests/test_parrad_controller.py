from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from continuous_mmwave.data_parrad import PARradSample
from continuous_mmwave.models.transition import PrototypeState
from continuous_mmwave.parrad_controller import PARradOnlineController, PARradStage3Runtime
from continuous_mmwave.parrad_stage3_diagnostics import ControllerThresholds, evaluate_controller


class _EventFormer:
    def initial_state(self, batch: int) -> torch.Tensor:
        return torch.zeros((batch, 2, 4))

    def build_prompt_cache(self, batch: int):
        return None

    def stream_update(self, previous, radar, maturity, radar_mask, prompt_cache):
        increment = radar.mean(dim=(1, 2))[:, None, None]
        gates = torch.full((*previous.shape[:2], 1), maturity, dtype=previous.dtype)
        return SimpleNamespace(event_tokens=previous + increment, gates=gates)


class _Projection(nn.Module):
    def forward(self, events):
        return torch.nn.functional.normalize(events.mean(dim=1), dim=-1)


class _Local(nn.Module):
    def __init__(self, threshold: float) -> None:
        super().__init__()
        self.threshold = threshold

    def forward(self, previous, current, mask):
        high = current.mean(dim=(1, 2)) >= self.threshold
        return torch.where(high, torch.full_like(high, 10.0, dtype=torch.float32), torch.full_like(high, -10.0, dtype=torch.float32))


class _Radar(nn.Module):
    def forward(self, tokens, mask, state, update_period_s):
        value = tokens.mean(dim=1)
        if state is None:
            updated = PrototypeState.initialize(value)
        else:
            updated = PrototypeState(
                fast=value.clone(), mean=value.clone(), m2=state.m2.clone(),
                count=state.count + 1, age_seconds=state.age_seconds + update_period_s, alpha=0.5,
            )
        return torch.full((tokens.shape[0],), -10.0), updated


class _ConstantDifference(nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, active, candidate, maturity):
        return torch.full((active.shape[0],), self.logit, device=active.device)


class _ConstantVerify(nn.Module):
    def __init__(self, logit: float) -> None:
        super().__init__()
        self.logit = logit

    def forward(self, inputs):
        return torch.full(inputs.shape[:-1], self.logit, device=inputs.device)


def _controller(
    *, local_threshold: float = 5.0, difference_logit: float = -10.0,
    verify_logit: float = -10.0, reject_timeout_s: float = 0.18,
) -> PARradOnlineController:
    return PARradOnlineController(
        event_former=_EventFormer(),
        semantic_projection=_Projection(),
        local_change=_Local(local_threshold),
        radar_transition=_Radar(),
        semantic_difference=_ConstantDifference(difference_logit),
        verify=_ConstantVerify(verify_logit),
        update_period_s=0.18,
        proposal_threshold=0.8,
        proposal_persistence=1,
        confirm_threshold=0.8,
        confirm_persistence=1,
        reject_threshold=0.2,
        reject_persistence=1,
        reject_timeout_s=reject_timeout_s,
        maturity_seconds=0.36,
    )


def _tokens(value: float) -> torch.Tensor:
    return torch.full((1, 4, 4), value, dtype=torch.float32)


def _mask() -> torch.Tensor:
    return torch.ones((1, 4), dtype=torch.bool)


def test_proposal_creates_frozen_reference_candidate_and_shadow() -> None:
    controller = _controller()
    controller.step_tokens(_tokens(1), _mask(), timestamp=0.0)
    before = controller.active.event_tokens.clone()

    output = controller.step_tokens(_tokens(10), _mask(), timestamp=0.18)

    assert output.state == "candidate"
    assert torch.equal(controller.reference.event_tokens, before)
    assert torch.equal(controller.shadow.event_tokens_before_update, before)
    assert torch.equal(controller.candidate.event_tokens_before_update, controller.event_former.initial_state(1))


def test_second_proposal_is_ignored_while_candidate_exists() -> None:
    controller = _controller(reject_timeout_s=10)
    controller.step_tokens(_tokens(1), _mask(), 0.0)
    controller.step_tokens(_tokens(10), _mask(), 0.18)

    output = controller.step_tokens(_tokens(10), _mask(), 0.36)

    assert output.decision == "none"
    assert controller.fork_count == 1


def test_reject_matches_uninterrupted_shadow_without_replay() -> None:
    uninterrupted = _controller(local_threshold=100)
    forked = _controller()
    for timestamp, value in ((0.0, 1.0), (0.18, 10.0), (0.36, 2.0)):
        uninterrupted.step_tokens(_tokens(value), _mask(), timestamp)
        forked.step_tokens(_tokens(value), _mask(), timestamp)

    assert forked.replay_count == 0
    torch.testing.assert_close(forked.active.event_tokens, uninterrupted.active.event_tokens)
    assert forked.active.prototype == uninterrupted.active.prototype


def test_confirm_owns_proposal_boundary_and_never_emits_empty_job() -> None:
    controller = _controller(difference_logit=10, verify_logit=10, reject_timeout_s=10)
    first = controller.step_tokens(_tokens(10), _mask(), 0.0)
    assert first.decision == "none"
    controller.step_tokens(_tokens(1), _mask(), 0.18)
    proposal = controller.step_tokens(_tokens(10), _mask(), 0.36)

    confirmed = controller.step_tokens(_tokens(2), _mask(), 0.54, {"recording": "r1"})

    assert proposal.decision == "proposal"
    assert confirmed.decision == "confirm"
    assert confirmed.finalization_job.end_time == pytest.approx(0.36)
    assert confirmed.finalization_job.radar_tokens.numel() > 0
    assert controller.active.start_time == pytest.approx(0.36)
    assert confirmed.candidate_proposal_time == pytest.approx(0.36)
    assert confirmed.candidate_maturity == pytest.approx(1.0)


def test_short_candidate_has_zero_stability_and_can_reject() -> None:
    controller = _controller(reject_timeout_s=0.18)
    controller.step_tokens(_tokens(1), _mask(), 0.0)
    controller.step_tokens(_tokens(10), _mask(), 0.18)

    output = controller.step_tokens(_tokens(2), _mask(), 0.36)

    assert output.stability == 0.0
    assert output.decision == "reject"


def test_true_boundary_during_false_candidate_does_not_refork_or_replay() -> None:
    controller = _controller(reject_timeout_s=0.36)
    for timestamp, value in ((0.0, 1.0), (0.18, 10.0), (0.36, 10.0), (0.54, 2.0)):
        controller.step_tokens(_tokens(value), _mask(), timestamp)

    assert controller.fork_count == 1
    assert controller.replay_count == 0
    assert controller.decision_trace == ["none", "proposal", "none", "reject"]


class _CountingEncoder:
    def __init__(self) -> None:
        self.calls = 0

    def forward_update(self, rd, md, sensor_index, state, frame_mask):
        self.calls += 1
        return SimpleNamespace(
            radar_tokens=_tokens(float(self.calls)),
            radar_token_mask=_mask(),
            state=self.calls,
        )


class _DeviceCheckingEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("device_anchor", torch.empty(0, device="meta"))
        self.seen_device: torch.device | None = None

    def forward_update(self, rd, md, sensor_index, state, frame_mask):
        devices = {rd.device, md.device, sensor_index.device, frame_mask.device}
        assert devices == {self.device_anchor.device}
        self.seen_device = md.device
        return SimpleNamespace(
            radar_tokens=_tokens(1.0),
            radar_token_mask=_mask(),
            state=1,
        )


def test_runtime_calls_radar_encoder_once_per_external_update() -> None:
    encoder = _CountingEncoder()
    runtime = PARradStage3Runtime(encoder, _controller(local_threshold=1.5, reject_timeout_s=10))
    for index in range(3):
        runtime.step(
            torch.zeros((1, 2, 1, 1, 1)),
            torch.zeros((1, 2, 4)),
            torch.tensor([0]),
            torch.ones((1, 2), dtype=torch.bool),
            index * 0.18,
        )

    assert encoder.calls == 3


def test_runtime_reset_clears_encoder_and_controller_recording_state() -> None:
    encoder = _CountingEncoder()
    runtime = PARradStage3Runtime(encoder, _controller(local_threshold=100))
    runtime.step(
        torch.zeros((1, 2, 1, 1, 1)), torch.zeros((1, 2, 4)), torch.tensor([0]),
        torch.ones((1, 2), dtype=torch.bool), 0.0,
    )

    runtime.reset()

    assert runtime.encoder_state is None
    assert runtime.controller.active is None
    assert runtime.controller.decision_trace == []


def test_continuous_evaluator_moves_recording_updates_to_runtime_device() -> None:
    encoder = _DeviceCheckingEncoder()
    runtime = PARradStage3Runtime(encoder, _controller(local_threshold=100))
    sample = PARradSample(
        rd=torch.zeros((2, 1, 1, 1)),
        md=torch.zeros((2, 4)),
        labels=torch.tensor([0, 0]),
        timestamps=torch.arange(2, dtype=torch.float64) * 0.09,
        sensor_index=0,
        subject_id="s1",
        session_id="session1",
        sensor_id="60ghz",
        environment_id="lab",
    )

    evaluate_controller(
        runtime,
        [sample],
        ControllerThresholds(0.8, 0.8, 0.2, 10, "validation-fingerprint"),
    )

    assert encoder.seen_device == torch.device("meta")


def test_continuous_evaluator_matches_confirmed_boundary_after_inference() -> None:
    runtime = PARradStage3Runtime(
        _CountingEncoder(),
        _controller(local_threshold=1.5, difference_logit=10, verify_logit=10, reject_timeout_s=10),
    )
    sample = PARradSample(
        rd=torch.zeros((6, 1, 1, 1)),
        md=torch.zeros((6, 4)),
        labels=torch.tensor([0, 0, 1, 1, 1, 1]),
        timestamps=torch.arange(6, dtype=torch.float64) * 0.09,
        sensor_index=0,
        subject_id="s1",
        session_id="session1",
        sensor_id="60ghz",
        environment_id="lab",
    )

    metrics = evaluate_controller(
        runtime,
        [sample],
        ControllerThresholds(0.8, 0.8, 0.2, 10, "validation-fingerprint"),
    )

    assert metrics["boundary_036"]["f1"] == pytest.approx(1.0)
    assert metrics["recordings"][0]["predicted_boundaries"] == pytest.approx([0.18])
    assert metrics["median_confirmation_latency_s"] == pytest.approx(0.18)


def test_continuous_evaluator_can_capture_state_norm_transition_trace() -> None:
    runtime = PARradStage3Runtime(
        _CountingEncoder(),
        _controller(
            local_threshold=1.5,
            difference_logit=10,
            verify_logit=10,
            reject_timeout_s=10,
        ),
    )
    sample = PARradSample(
        rd=torch.zeros((6, 1, 1, 1)),
        md=torch.zeros((6, 4)),
        labels=torch.tensor([0, 0, 1, 1, 1, 1]),
        timestamps=torch.arange(6, dtype=torch.float64) * 0.09,
        sensor_index=0,
        subject_id="s1",
        session_id="session1",
        sensor_id="60ghz",
        environment_id="lab",
    )

    metrics = evaluate_controller(
        runtime,
        [sample],
        ControllerThresholds(0.8, 0.8, 0.2, 10, "validation-fingerprint"),
        capture_state_trace=True,
    )

    trace = metrics["state_trace"]
    assert len(trace) == 3
    assert [row["decision"] for row in trace] == ["none", "proposal", "confirm"]
    assert trace[1]["state_before"] == "active"
    assert trace[1]["state_after"] == "candidate"
    assert trace[2]["state_before"] == "candidate"
    assert trace[2]["state_after"] == "active"
    assert trace[2]["label"] == 1
    assert trace[2]["nearest_truth_distance_s"] == pytest.approx(0.18)
    assert all(math.isfinite(row["state_norm_after"]) for row in trace)


def test_confirmation_latency_is_measured_from_matched_truth_boundary() -> None:
    runtime = PARradStage3Runtime(
        _CountingEncoder(),
        _controller(local_threshold=1.5, difference_logit=10, verify_logit=10, reject_timeout_s=10),
    )
    sample = PARradSample(
        rd=torch.zeros((6, 1, 1, 1)),
        md=torch.zeros((6, 4)),
        labels=torch.tensor([0, 1, 1, 1, 1, 1]),
        timestamps=torch.arange(6, dtype=torch.float64) * 0.09,
        sensor_index=0,
        subject_id="s1",
        session_id="session1",
        sensor_id="60ghz",
        environment_id="lab",
    )

    metrics = evaluate_controller(
        runtime,
        [sample],
        ControllerThresholds(0.8, 0.8, 0.2, 10, "validation-fingerprint"),
    )

    assert metrics["median_confirmation_latency_s"] == pytest.approx(0.27)


def test_proposal_only_baseline_is_not_suppressed_by_full_controller_candidate() -> None:
    runtime = PARradStage3Runtime(
        _CountingEncoder(),
        _controller(local_threshold=1.5, difference_logit=-10, verify_logit=-10, reject_timeout_s=10),
    )
    sample = PARradSample(
        rd=torch.zeros((8, 1, 1, 1)),
        md=torch.zeros((8, 4)),
        labels=torch.zeros(8, dtype=torch.long),
        timestamps=torch.arange(8, dtype=torch.float64) * 0.09,
        sensor_index=0,
        subject_id="s1",
        session_id="session1",
        sensor_id="60ghz",
        environment_id="lab",
    )

    metrics = evaluate_controller(
        runtime,
        [sample],
        ControllerThresholds(0.8, 0.8, 0.2, 10, "validation-fingerprint"),
    )

    assert metrics["recordings"][0]["proposal_boundaries"] == pytest.approx(
        [0.18, 0.36, 0.54]
    )


def test_controller_metrics_aggregate_recordings_with_independent_timestamp_origins() -> None:
    runtime = PARradStage3Runtime(
        _CountingEncoder(),
        _controller(local_threshold=1.5, difference_logit=10, verify_logit=10, reject_timeout_s=10),
    )

    def sample(start: float, session: str) -> PARradSample:
        return PARradSample(
            rd=torch.zeros((6, 1, 1, 1)),
            md=torch.zeros((6, 4)),
            labels=torch.tensor([0, 0, 1, 1, 1, 1]),
            timestamps=start + torch.arange(6, dtype=torch.float64) * 0.09,
            sensor_index=0,
            subject_id="s1",
            session_id=session,
            sensor_id="60ghz",
            environment_id="lab",
        )

    metrics = evaluate_controller(
        runtime,
        [sample(100.0, "late-clock"), sample(0.0, "zero-clock")],
        ControllerThresholds(0.8, 0.8, 0.2, 10, "validation-fingerprint"),
    )

    assert metrics["boundary_036"]["true_positives"] == 2


def test_over_segmentation_does_not_reward_missing_all_boundaries() -> None:
    runtime = PARradStage3Runtime(
        _CountingEncoder(),
        _controller(local_threshold=100, reject_timeout_s=10),
    )
    sample = PARradSample(
        rd=torch.zeros((6, 1, 1, 1)),
        md=torch.zeros((6, 4)),
        labels=torch.tensor([0, 0, 1, 1, 1, 1]),
        timestamps=torch.arange(6, dtype=torch.float64) * 0.09,
        sensor_index=0,
        subject_id="s1",
        session_id="session1",
        sensor_id="60ghz",
        environment_id="lab",
    )

    metrics = evaluate_controller(
        runtime,
        [sample],
        ControllerThresholds(0.8, 0.8, 0.2, 10, "validation-fingerprint"),
    )

    assert metrics["over_segmentation"] == 0.0
    assert metrics["under_segmentation"] == 1.0
    assert metrics["predicted_to_truth_ratio"] == 0.0
