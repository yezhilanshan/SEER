"""Online Proposal/Candidate/Semantic-Verify Fork-and-Shadow controller."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch
from torch import Tensor

from continuous_mmwave.models.transition import PrototypeState, candidate_stability


@dataclass(frozen=True)
class ActiveEventState:
    event_tokens: Tensor
    prototype: PrototypeState
    start_time: float
    radar_chunks: tuple[Tensor, ...]
    radar_masks: tuple[Tensor, ...]
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class FrozenActiveReference:
    event_tokens: Tensor
    semantic_key: Tensor
    prototype: PrototypeState
    start_time: float
    proposal_time: float
    radar_chunks: tuple[Tensor, ...]
    radar_masks: tuple[Tensor, ...]
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class CandidateBranch:
    event_tokens_before_update: Tensor
    event_tokens: Tensor
    prototype: PrototypeState
    start_time: float
    radar_chunks: tuple[Tensor, ...]
    radar_masks: tuple[Tensor, ...]
    key_history: tuple[Tensor, ...]
    age_updates: int
    metadata: Mapping[str, str]

    def as_active(self) -> ActiveEventState:
        return ActiveEventState(
            self.event_tokens,
            self.prototype,
            self.start_time,
            self.radar_chunks,
            self.radar_masks,
            self.metadata,
        )


@dataclass(frozen=True)
class ShadowBranch:
    event_tokens_before_update: Tensor
    event_tokens: Tensor
    prototype: PrototypeState
    start_time: float
    radar_chunks: tuple[Tensor, ...]
    radar_masks: tuple[Tensor, ...]
    metadata: Mapping[str, str]

    def as_active(self) -> ActiveEventState:
        return ActiveEventState(
            self.event_tokens,
            self.prototype,
            self.start_time,
            self.radar_chunks,
            self.radar_masks,
            self.metadata,
        )


@dataclass(frozen=True)
class FinalizationJob:
    start_time: float
    end_time: float
    radar_tokens: Tensor
    radar_mask: Tensor
    provisional_event_tokens: Tensor
    metadata: Mapping[str, str]


@dataclass(frozen=True)
class ControllerOutput:
    state: Literal["active", "candidate"]
    proposal_probability: float
    difference_probability: float | None
    stability: float | None
    verify_probability: float | None
    candidate_maturity: float | None
    candidate_proposal_time: float | None
    decision: Literal["none", "proposal", "confirm", "reject"]
    finalization_job: FinalizationJob | None


class PARradOnlineController:
    """Batch-one online controller with exact no-replay Reject semantics."""

    def __init__(
        self,
        *,
        event_former: Any,
        semantic_projection: Any,
        local_change: Any,
        radar_transition: Any,
        semantic_difference: Any,
        verify: Any,
        update_period_s: float = 0.18,
        proposal_threshold: float = 0.5,
        proposal_persistence: int = 1,
        confirm_threshold: float = 0.5,
        confirm_persistence: int = 2,
        reject_threshold: float = 0.5,
        reject_persistence: int = 2,
        reject_timeout_s: float = 1.8,
        maturity_seconds: float = 1.2,
        tau_stability: float = 0.1,
    ) -> None:
        positive = (update_period_s, proposal_persistence, confirm_persistence, reject_persistence, reject_timeout_s, maturity_seconds, tau_stability)
        if any(value <= 0 for value in positive):
            raise ValueError("controller timing and persistence values must be positive")
        if any(not 0 <= value <= 1 for value in (proposal_threshold, confirm_threshold, reject_threshold)):
            raise ValueError("controller thresholds must be in [0,1]")
        self.event_former = event_former
        self.semantic_projection = semantic_projection
        self.local_change = local_change
        self.radar_transition = radar_transition
        self.semantic_difference = semantic_difference
        self.verify = verify
        self.update_period_s = update_period_s
        self.proposal_threshold = proposal_threshold
        self.proposal_persistence = proposal_persistence
        self.confirm_threshold = confirm_threshold
        self.confirm_persistence = confirm_persistence
        self.reject_threshold = reject_threshold
        self.reject_persistence = reject_persistence
        self.reject_timeout_s = reject_timeout_s
        self.maturity_seconds = maturity_seconds
        self.tau_stability = tau_stability
        self.prompt_cache = event_former.build_prompt_cache(1)
        self.active: ActiveEventState | None = None
        self.reference: FrozenActiveReference | None = None
        self.candidate: CandidateBranch | None = None
        self.shadow: ShadowBranch | None = None
        self._previous_radar: Tensor | None = None
        self._previous_mask: Tensor | None = None
        self._last_timestamp: float | None = None
        self._proposal_streak = 0
        self._confirm_streak = 0
        self._reject_streak = 0
        self.fork_count = 0
        self.replay_count = 0
        self.decision_trace: list[str] = []

    def reset(self) -> None:
        """Reset recording-local state while retaining frozen modules and thresholds."""
        self.active = None
        self.reference = None
        self.candidate = None
        self.shadow = None
        self._previous_radar = None
        self._previous_mask = None
        self._last_timestamp = None
        self._proposal_streak = 0
        self._confirm_streak = 0
        self._reject_streak = 0
        self.fork_count = 0
        self.replay_count = 0
        self.decision_trace = []

    @property
    def state(self) -> Literal["active", "candidate"]:
        return "candidate" if self.candidate is not None else "active"

    def _validate(self, radar_tokens: Tensor, radar_mask: Tensor, timestamp: float) -> None:
        if radar_tokens.ndim != 3 or radar_tokens.shape[0] != 1:
            raise ValueError("online controller requires radar_tokens [1,R,D]")
        if radar_mask.shape != radar_tokens.shape[:2] or not radar_mask.bool().any():
            raise ValueError("online controller requires a valid radar mask [1,R]")
        if not torch.isfinite(radar_tokens).all() or not math.isfinite(timestamp):
            raise ValueError("controller inputs must be finite")
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("controller timestamps must be strictly increasing")

    def _event_update(self, previous: Tensor, tokens: Tensor, mask: Tensor, maturity: float) -> Tensor:
        return self.event_former.stream_update(
            previous, tokens, maturity, mask, self.prompt_cache
        ).event_tokens

    def _physical_scores(self, tokens: Tensor, mask: Tensor, prototype: PrototypeState | None):
        if self._previous_radar is None or self._previous_mask is None:
            local_probability = 0.0
        else:
            common_mask = mask.bool() & self._previous_mask.bool()
            local_probability = float(torch.sigmoid(self.local_change(self._previous_radar, tokens, common_mask))[0])
        radar_logits, updated = self.radar_transition(tokens, mask, prototype, self.update_period_s)
        radar_probability = float(torch.sigmoid(radar_logits)[0])
        proposal_probability = 1 - (1 - local_probability) * (1 - radar_probability)
        return proposal_probability, updated

    def _clear_candidate(self) -> None:
        self.reference = None
        self.candidate = None
        self.shadow = None
        self._confirm_streak = 0
        self._reject_streak = 0

    def _record(self, decision: str) -> None:
        self.decision_trace.append(decision)

    @torch.inference_mode()
    def step_tokens(
        self,
        radar_tokens: Tensor,
        radar_mask: Tensor,
        timestamp: float,
        metadata: Mapping[str, str] | None = None,
    ) -> ControllerOutput:
        self._validate(radar_tokens, radar_mask, timestamp)
        metadata = {} if metadata is None else dict(metadata)
        tokens = radar_tokens.clone()
        mask = radar_mask.bool().clone()
        finalization: FinalizationJob | None = None
        difference_probability: float | None = None
        stability_value: float | None = None
        verify_probability: float | None = None
        candidate_maturity: float | None = None
        candidate_proposal_time: float | None = None

        if self.candidate is None:
            current_state = self.event_former.initial_state(1) if self.active is None else self.active.event_tokens
            current_prototype = None if self.active is None else self.active.prototype
            proposal_probability, updated_prototype = self._physical_scores(tokens, mask, current_prototype)
            self._proposal_streak = self._proposal_streak + 1 if proposal_probability >= self.proposal_threshold else 0
            can_fork = self.active is not None and bool(self.active.radar_chunks)
            if can_fork and self._proposal_streak >= self.proposal_persistence:
                assert self.active is not None
                reference_key = self.semantic_projection(self.active.event_tokens).clone()
                self.reference = FrozenActiveReference(
                    event_tokens=self.active.event_tokens.clone(),
                    semantic_key=reference_key,
                    prototype=self.active.prototype,
                    start_time=self.active.start_time,
                    proposal_time=timestamp,
                    radar_chunks=tuple(chunk.clone() for chunk in self.active.radar_chunks),
                    radar_masks=tuple(value.clone() for value in self.active.radar_masks),
                    metadata=dict(self.active.metadata),
                )
                candidate_before = self.event_former.initial_state(1).clone()
                _, candidate_prototype = self.radar_transition(tokens, mask, None, self.update_period_s)
                candidate_tokens = self._event_update(candidate_before, tokens, mask, min(self.update_period_s / self.maturity_seconds, 1.0))
                candidate_key = self.semantic_projection(candidate_tokens).clone()
                shadow_before = self.active.event_tokens.clone()
                shadow_tokens = self._event_update(shadow_before, tokens, mask, 1.0)
                self.candidate = CandidateBranch(
                    candidate_before, candidate_tokens, candidate_prototype, timestamp,
                    (tokens.clone(),), (mask.clone(),), (candidate_key,), 1, metadata,
                )
                self.shadow = ShadowBranch(
                    shadow_before, shadow_tokens, updated_prototype, self.active.start_time,
                    self.active.radar_chunks + (tokens.clone(),),
                    self.active.radar_masks + (mask.clone(),),
                    dict(self.active.metadata),
                )
                self.fork_count += 1
                self._proposal_streak = 0
                decision = "proposal"
            else:
                updated_tokens = self._event_update(current_state, tokens, mask, 1.0)
                start_time = timestamp if self.active is None else self.active.start_time
                chunks = () if self.active is None else self.active.radar_chunks
                masks = () if self.active is None else self.active.radar_masks
                active_metadata = metadata if self.active is None else {**self.active.metadata, **metadata}
                self.active = ActiveEventState(
                    updated_tokens,
                    updated_prototype,
                    start_time,
                    chunks + (tokens.clone(),),
                    masks + (mask.clone(),),
                    active_metadata,
                )
                decision = "none"
        else:
            assert self.reference is not None and self.shadow is not None
            proposal_probability, candidate_prototype = self._physical_scores(tokens, mask, self.candidate.prototype)
            _, shadow_prototype = self.radar_transition(tokens, mask, self.shadow.prototype, self.update_period_s)
            age_updates = self.candidate.age_updates + 1
            maturity = min(age_updates * self.update_period_s / self.maturity_seconds, 1.0)
            candidate_maturity = maturity
            candidate_proposal_time = self.reference.proposal_time
            candidate_before = self.candidate.event_tokens.clone()
            candidate_tokens = self._event_update(candidate_before, tokens, mask, maturity)
            candidate_key = self.semantic_projection(candidate_tokens).clone()
            key_history = self.candidate.key_history + (candidate_key,)
            self.candidate = replace(
                self.candidate,
                event_tokens_before_update=candidate_before,
                event_tokens=candidate_tokens,
                prototype=candidate_prototype,
                radar_chunks=self.candidate.radar_chunks + (tokens.clone(),),
                radar_masks=self.candidate.radar_masks + (mask.clone(),),
                key_history=key_history,
                age_updates=age_updates,
                metadata={**self.candidate.metadata, **metadata},
            )
            shadow_before = self.shadow.event_tokens.clone()
            self.shadow = replace(
                self.shadow,
                event_tokens_before_update=shadow_before,
                event_tokens=self._event_update(shadow_before, tokens, mask, 1.0),
                prototype=shadow_prototype,
                radar_chunks=self.shadow.radar_chunks + (tokens.clone(),),
                radar_masks=self.shadow.radar_masks + (mask.clone(),),
            )
            difference_probability = float(torch.sigmoid(self.semantic_difference(
                self.reference.semantic_key,
                candidate_key,
                torch.tensor([maturity], dtype=candidate_key.dtype, device=candidate_key.device),
            ))[0])
            stability_tensor = candidate_stability(torch.stack(key_history, dim=1), self.tau_stability)
            stability_value = float(stability_tensor[0])
            verify_inputs = torch.tensor(
                [[difference_probability, stability_value, maturity]],
                dtype=candidate_key.dtype,
                device=candidate_key.device,
            )
            verify_probability = float(torch.sigmoid(self.verify(verify_inputs))[0])
            self._confirm_streak = self._confirm_streak + 1 if verify_probability >= self.confirm_threshold else 0
            elapsed_candidate_s = (age_updates - 1) * self.update_period_s
            timed_out = elapsed_candidate_s >= self.reject_timeout_s
            self._reject_streak = self._reject_streak + 1 if timed_out and difference_probability < self.reject_threshold else 0
            if self._confirm_streak >= self.confirm_persistence:
                old_tokens = torch.cat(self.reference.radar_chunks, dim=1)
                old_masks = torch.cat(self.reference.radar_masks, dim=1)
                finalization = FinalizationJob(
                    self.reference.start_time,
                    self.reference.proposal_time,
                    old_tokens,
                    old_masks,
                    self.reference.event_tokens.clone(),
                    dict(self.reference.metadata),
                )
                self.active = self.candidate.as_active()
                self._clear_candidate()
                decision = "confirm"
            elif self._reject_streak >= self.reject_persistence:
                self.active = self.shadow.as_active()
                self._clear_candidate()
                decision = "reject"
            else:
                decision = "none"

        self._previous_radar = tokens
        self._previous_mask = mask
        self._last_timestamp = timestamp
        self._record(decision)
        return ControllerOutput(
            state=self.state,
            proposal_probability=proposal_probability,
            difference_probability=difference_probability,
            stability=stability_value,
            verify_probability=verify_probability,
            candidate_maturity=candidate_maturity,
            candidate_proposal_time=candidate_proposal_time,
            decision=decision,
            finalization_job=finalization,
        )


class PARradStage3Runtime:
    """Own continuous Radar Encoder state and call it exactly once per update."""

    def __init__(self, radar_encoder: Any, controller: PARradOnlineController) -> None:
        self.radar_encoder = radar_encoder
        self.controller = controller
        self.encoder_state = None

    def reset(self) -> None:
        self.encoder_state = None
        self.controller.reset()

    @torch.inference_mode()
    def step(
        self,
        rd: Tensor,
        md: Tensor,
        sensor_index: Tensor,
        frame_mask: Tensor,
        timestamp: float,
        metadata: Mapping[str, str] | None = None,
    ) -> ControllerOutput:
        encoded = self.radar_encoder.forward_update(
            rd,
            md,
            sensor_index=sensor_index,
            state=self.encoder_state,
            frame_mask=frame_mask,
        )
        self.encoder_state = encoded.state
        return self.controller.step_tokens(
            encoded.radar_tokens,
            encoded.radar_token_mask,
            timestamp,
            metadata,
        )
