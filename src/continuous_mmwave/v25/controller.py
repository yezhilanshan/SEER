"""v2.5 online forks, sequential evidence and nonblocking state reconciliation.

Only the owning streaming thread mutates controller state. Workers read frozen
modules and immutable snapshots; results are polled, never awaited by step_tokens.
The thread executor is an execution mechanism, not a hard realtime GPU guarantee.
"""
from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace, asdict
from typing import Any

import torch
from torch import Tensor

from ..models.transition import candidate_stability
from .models import physical_difference
from .evidence import EvidenceConfig, SequentialEvidence, Episode
from .buffer import FeatureBuffer, FeatureFrame, CommitArchive
from .boundary import localize_boundary


@dataclass(frozen=True)
class ControllerConfig:
    update_period_s: float = .18
    proposal_threshold: float = .6
    proposal_persistence: int = 2
    delta_pre_s: float = .5
    stability_variance_tau: float = .1
    update_warmup_s: float = 0.
    boundary_local_weight: float = 1.
    boundary_radar_weight: float = 1.
    boundary_semantic_weight: float = 1.
    evidence_weights: tuple[float, ...] | None = None
    replay_batch_updates: int = 16
    commit_max_tokens: int = 256

    def __post_init__(self):
        scalars = [v for v in asdict(self).values() if isinstance(v,(int,float))]
        if not all(math.isfinite(v) for v in scalars) or not (self.update_period_s > 0 and
            0 < self.proposal_threshold <= 1 and isinstance(self.proposal_persistence,int) and self.proposal_persistence > 0
            and self.delta_pre_s >= 0 and self.stability_variance_tau > 0 and self.update_warmup_s >= 0
            and isinstance(self.replay_batch_updates,int) and self.replay_batch_updates > 0 and self.commit_max_tokens >= 4):
            raise ValueError('invalid v2.5 controller configuration')
        weights = (self.boundary_local_weight,self.boundary_radar_weight,self.boundary_semantic_weight)
        if min(weights) < 0 or sum(weights) <= 0:
            raise ValueError('invalid boundary weights')


@dataclass(frozen=True)
class Branch:
    event_id: str
    start_time: float
    event_tokens: Tensor
    prototype: Any
    watermark: int
    timestamp: float
    corrected: bool = True
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FinalizationJob:
    event_id: str
    start_time: float
    end_time: float
    confirmed_at: float
    confidence: float
    radar_tokens: Tensor
    radar_mask: Tensor
    metadata: dict
    confirmed: bool = True


@dataclass(frozen=True)
class ControllerOutput:
    state: str
    decision: str
    proposal_probability: float
    difference_probability: float | None = None
    stability: float | None = None
    evidence_score: float | None = None
    refined_boundary: float | None = None
    finalization_job: FinalizationJob | None = None
    reconciliation_pending: bool = False
    failures: tuple[str, ...] = ()


@dataclass
class Reconciliation:
    event_id: str
    epoch: int
    pin: str
    boundary: float
    future: Any
    history: dict = field(default_factory=dict)


@dataclass
class PendingBoundary:
    old_event_id: str
    new_event_id: str
    epoch: int
    episode: Episode
    confirmed_at: float
    confidence: float
    old_metadata: dict
    future: Any


class V25Controller:
    architecture = 'v25-sequential-evidence'

    def __init__(self, event_former, semantic_projection, heads, config=None, evidence_config=None):
        if getattr(event_former.state_gate,'architecture',None) != 'v25-age-free-tanh':
            raise ValueError('v2.5 requires an explicitly migrated age-free EventFormer')
        if hasattr(heads,'verify'):
            raise ValueError('Verify head is not part of v2.5')
        self.event_former, self.semantic_projection, self.heads = event_former, semantic_projection, heads
        self.config = config or ControllerConfig()
        self.evidence_config = evidence_config or EvidenceConfig()
        for module in (event_former,semantic_projection,heads):
            module.eval()
        self.prompt_cache = event_former.build_prompt_cache(1)
        self._executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='v25-reconcile')
        self._boundary_executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='v25-boundary')
        self._boundaries,self._ready_jobs,self._latest_boundary_future = [],[],None
        self._epoch, self._pending, self._orphans, self._closed = 0, None, [], False
        self.reset()

    def reset(self):
        self._epoch += 1
        if self._pending is not None:
            self._pending.future.cancel()
            self._orphans.append(self._pending.future)
        for task in self._boundaries:
            task.future.cancel()
            self._orphans.append(task.future)
        self._boundaries,self._ready_jobs,self._latest_boundary_future = [],[],None
        self._pending = None
        self.buffer = FeatureBuffer(self.config.delta_pre_s+self.evidence_config.timeout_s+self.config.update_period_s)
        self.archive = CommitArchive(self.config.commit_max_tokens,
                                     self.config.delta_pre_s+self.evidence_config.timeout_s+self.config.update_period_s)
        self.active = self.candidate = self.shadow = self.reference = None
        self._reference_key = self._reference_evidence = None
        self._candidate_keys, self._candidate_records = [], []
        self._episode = self._accumulator = None
        self.episodes, self.decision_trace, self.failures = [], [], []
        self._previous_frame, self._sequence, self._proposal_streak, self._event_serial = None,-1,0,0
        self.metrics = dict(atomic_swaps=0,rebuild_updates=0,reconciliation_failures=0,stale_jobs=0,
                            disjoint_masks=0,max_pinned_updates=0)

    @property
    def state(self):
        return 'candidate' if self.candidate is not None else 'active'

    def _id(self):
        self._event_serial += 1
        return f'r{self._epoch}:e{self._event_serial}'

    def _advance(self, branch, tokens, mask, timestamp, sequence, *, prototype=None):
        if branch is None:
            previous = self.event_former.initial_state(1)
            start = timestamp
            event_id = self._id()
            period = self.config.update_period_s
        else:
            previous,start,event_id = branch.event_tokens,branch.start_time,branch.event_id
            period = timestamp-branch.timestamp
        if prototype is None:
            _,prototype = self.heads.radar(tokens,mask,None if branch is None else branch.prototype,period)
        proposed = self.event_former.stream_update(previous,tokens,1.,mask,self.prompt_cache).event_tokens
        if self.config.update_warmup_s:
            eta = min(1.,max(0.,(timestamp-start+self.config.update_period_s)/self.config.update_warmup_s))
            proposed = previous+eta*(proposed-previous)
        return Branch(event_id,start,proposed.detach().clone(),prototype,sequence,timestamp,
                      True if branch is None else branch.corrected,{} if branch is None else branch.metadata)

    def _new_candidate(self, tokens, mask, timestamp, sequence, metadata):
        # Worker and streaming IDs are assigned on the owning thread only.
        initial = Branch(self._id(),timestamp,self.event_former.initial_state(1).clone(),None,
                         sequence-1,timestamp-self.config.update_period_s,False,dict(metadata))
        return self._advance(initial,tokens,mask,timestamp,sequence)

    def _physical(self,tokens,mask,branch,timestamp):
        local = 0.
        if self._previous_frame is not None:
            common = mask & self._previous_frame.mask
            if common.any():
                local = float(self.heads.local(self._previous_frame.tokens,tokens,common).sigmoid()[0])
            else:
                self.metrics['disjoint_masks'] += 1
        period = self.config.update_period_s if branch is None else timestamp-branch.timestamp
        logits, prototype = self.heads.radar(tokens,mask,None if branch is None else branch.prototype,period)
        radar = float(logits.sigmoid()[0])
        return local,radar,prototype

    def _observe(self,timestamp):
        key = self.semantic_projection(self.candidate.event_tokens).detach().clone()
        evidence = self.heads.evidence(self.candidate.event_tokens).detach().clone()
        self._candidate_keys.append(key)
        self._candidate_keys = self._candidate_keys[-3:]
        stability = float(candidate_stability(torch.stack(self._candidate_keys,1),self.config.stability_variance_tau)[0])
        difference = float(self.heads.semantic(self._reference_key,key).sigmoid()[0])
        d_evidence = float(physical_difference(self._reference_evidence,evidence,self.config.evidence_weights)[0])
        age = timestamp-self._episode.proposal_time
        result = self._accumulator.update(difference,stability,d_evidence,age)
        self._episode.observations.append(dict(timestamp=timestamp,age_s=age,difference=difference,
            stability=stability,physical_difference=d_evidence,score=result.score,reliability=result.reliability))
        self._candidate_records.append((key,evidence))
        return difference,stability,result

    def _finish_episode(self,decision,timestamp,boundary=None):
        self._episode.decision,self._episode.end_time,self._episode.refined_boundary = decision,timestamp,boundary
        self.episodes.append(self._episode)
        self._episode = None
        self.candidate = self.shadow = self.reference = None
        self._candidate_keys,self._candidate_records = [],[]
        self._reference_key = self._reference_evidence = self._accumulator = None
        self._proposal_streak = 0

    def _release_pin(self,owner):
        self.buffer.release(owner)
        self.archive.pin_start = min(self.buffer.pins.values()) if self.buffer.pins else None
        if self.archive.chunks:
            self.archive._compact(self.archive.chunks[-1].end)

    @torch.inference_mode()
    def _rebuild(self, event_id, start_time, frames, initial):
        # Do not call _advance(None): that would mutate the owner's ID counter.
        branch = initial or Branch(event_id,start_time,self.event_former.initial_state(1).clone(),None,
            frames[0].sequence-1,frames[0].timestamp-self.config.update_period_s,True)
        history = {}
        for frame in frames:
            branch = self._advance(branch,frame.tokens,frame.mask,frame.timestamp,frame.sequence)
            history[frame.sequence] = branch
        return branch,history

    def _submit(self,task,initial=None):
        after = -1 if initial is None else initial.watermark
        frames = self.buffer.since(task.boundary,after_sequence=after)[:self.config.replay_batch_updates]
        if not frames:
            raise RuntimeError('pinned reconciliation features missing')
        task.future = self._executor.submit(self._rebuild,task.event_id,task.boundary,frames,initial)

    def _start_reconciliation(self,pin,boundary):
        if self._pending is not None:
            self._pending.future.cancel()
            self._orphans.append(self._pending.future)
            self._release_pin(self._pending.pin)
            self.metrics['stale_jobs'] += 1
        task = Reconciliation(self.active.event_id,self._epoch,pin,boundary,None)
        self._pending = task
        self._submit(task)

    @torch.inference_mode()
    def _localize(self,frames,proposal_time,confirm_time,active_start,parent):
        # Boundary tasks are FIFO. A predecessor's result is therefore ready
        # before its child runs, even when the owner has not polled it yet.
        if parent is not None:
            active_start = parent.result()[0].timestamp
        localized = localize_boundary(frames,proposal_time=proposal_time,confirm_time=confirm_time,
            active_start=active_start,delta_pre_s=self.config.delta_pre_s,
            weights=(self.config.boundary_local_weight,self.config.boundary_radar_weight,self.config.boundary_semantic_weight))
        return localized,active_start

    def _queue_boundary(self,timestamp,score):
        episode = self._episode
        parent = self._latest_boundary_future
        frames = tuple(self.buffer.frames)
        future = self._boundary_executor.submit(self._localize,frames,episode.proposal_time,timestamp,
                                                 self.reference.start_time,parent)
        confidence = 1/(1+math.exp(-min(50.,max(-50.,score))))
        self._boundaries.append(PendingBoundary(self.reference.event_id,self.candidate.event_id,self._epoch,
            episode,timestamp,confidence,dict(self.reference.metadata),future))
        self._latest_boundary_future = future
        # An older corrected Active can no longer replace the newly accepted
        # lineage. Its Commit still remains queued, with the final interval.
        if self._pending is not None:
            self._pending.future.cancel()
            self._orphans.append(self._pending.future)
            self._release_pin(self._pending.pin)
            self._pending = None
            self.metrics['stale_jobs'] += 1

    def _poll_boundaries(self):
        while self._boundaries and self._boundaries[0].future.done():
            task = self._boundaries.pop(0)
            if task.epoch != self._epoch:
                self._release_pin(task.episode.candidate_id)
                continue
            try:
                result,old_start = task.future.result()
                old,new = self.archive.partition(result.timestamp)
                tokens,mask = CommitArchive.tensors(old)
                self._ready_jobs.append(FinalizationJob(task.old_event_id,old_start,result.timestamp,
                    task.confirmed_at,task.confidence,tokens,mask,task.old_metadata))
                self.archive.chunks = new
                task.episode.refined_boundary = result.timestamp
                if self.active is not None and self.active.event_id == task.new_event_id:
                    self.active = replace(self.active,start_time=result.timestamp,corrected=False)
                    if self.candidate is not None:
                        self.shadow = replace(self.shadow,start_time=result.timestamp,corrected=False)
                        self.reference = replace(self.reference,start_time=result.timestamp,corrected=False)
                    self._start_reconciliation(task.episode.candidate_id,result.timestamp)
                else:
                    self.metrics['stale_jobs'] += 1
                    self._release_pin(task.episode.candidate_id)
            except Exception as error:
                self.failures.append(f'Boundary {task.old_event_id}: {type(error).__name__}: {error}')
                self.metrics['reconciliation_failures'] += 1
                self._release_pin(task.episode.candidate_id)

    def drain_finalizations(self):
        self.poll()
        jobs,self._ready_jobs = self._ready_jobs,[]
        return jobs

    @torch.inference_mode()
    def poll(self):
        self._orphans = [f for f in self._orphans if not f.done()]
        self._poll_boundaries()
        task = self._pending
        if task is None or not task.future.done():
            return
        if task.epoch != self._epoch or self.active is None or task.event_id != self.active.event_id:
            self._release_pin(task.pin)
            self._pending = None
            self.metrics['stale_jobs'] += 1
            return
        try:
            branch,history = task.future.result()  # done() above, never a streaming wait
            task.history.update(history)
            self.metrics['rebuild_updates'] += len(history)
            if branch.watermark < self._sequence:
                self._submit(task,branch)
                return
            if branch.watermark != self._sequence:
                raise RuntimeError('reconciliation watermark ahead of stream')
            branch = replace(branch,metadata=dict(self.active.metadata),corrected=True)
            if self.candidate is not None:
                # A new fork may already be using this provisional Active.
                # Rebuild its frozen reference at the original fork watermark.
                corrected_reference = task.history.get(self.reference.watermark)
                if corrected_reference is None:
                    raise RuntimeError('missing frozen-reference reconciliation watermark')
                self.reference = corrected_reference
                self._reference_key = self.semantic_projection(corrected_reference.event_tokens).detach().clone()
                self._reference_evidence = self.heads.evidence(corrected_reference.event_tokens).detach().clone()
                self.shadow = branch
                self._accumulator = SequentialEvidence(self.evidence_config)
                for row,(key,evidence) in zip(self._episode.observations,self._candidate_records):
                    row['difference'] = float(self.heads.semantic(self._reference_key,key).sigmoid()[0])
                    row['physical_difference'] = float(physical_difference(self._reference_evidence,evidence,self.config.evidence_weights)[0])
                    revised = self._accumulator.update(row['difference'],row['stability'],row['physical_difference'],row['age_s'])
                    row['score'],row['reliability'] = revised.score,revised.reliability
            self.active = branch
            self.metrics['atomic_swaps'] += 1
            self._release_pin(task.pin)
            self._pending = None
        except Exception as error:
            self.failures.append(f'{type(error).__name__}: {error}')
            self.metrics['reconciliation_failures'] += 1
            self._release_pin(task.pin)
            self._pending = None

    @torch.inference_mode()
    def step_tokens(self, radar_tokens, radar_mask, timestamp, metadata=None):
        if self._closed:
            raise RuntimeError('controller is closed')
        if (radar_tokens.ndim != 3 or radar_tokens.shape[0] != 1 or radar_mask.shape != radar_tokens.shape[:2]
            or not radar_mask.bool().any() or not torch.isfinite(radar_tokens).all() or not math.isfinite(timestamp)):
            raise ValueError('finite single-recording radar batch/mask required')
        if self._previous_frame is not None and timestamp <= self._previous_frame.timestamp:
            raise ValueError('timestamps must be strictly increasing')
        self.poll()
        if self.failures:
            raise RuntimeError('controller failed; reset/recovery required: '+'; '.join(self.failures))
        tokens,mask = radar_tokens.detach().clone(),radar_mask.bool().clone()
        metadata = dict(metadata or {})
        self._sequence += 1
        local,radar,prototype = self._physical(tokens,mask,self.shadow if self.candidate is not None else self.active,timestamp)
        proposal = 1-(1-local)*(1-radar)
        decision,job,boundary = 'none',None,None
        difference = stability = score = None
        if self.candidate is None:
            self._proposal_streak = self._proposal_streak+1 if proposal >= self.config.proposal_threshold else 0
            if self.active is not None and self._proposal_streak >= self.config.proposal_persistence:
                self.reference = self.active
                self._reference_key = self.semantic_projection(self.reference.event_tokens).detach().clone()
                self._reference_evidence = self.heads.evidence(self.reference.event_tokens).detach().clone()
                self.shadow = self._advance(self.active,tokens,mask,timestamp,self._sequence,prototype=prototype)
                self.candidate = self._new_candidate(tokens,mask,timestamp,self._sequence,metadata)
                self._episode = Episode(self.candidate.event_id,self.active.event_id,timestamp)
                self._accumulator = SequentialEvidence(self.evidence_config)
                self.buffer.pin(self._episode.candidate_id,timestamp-self.config.delta_pre_s)
                self._proposal_streak = 0
                decision = 'proposal'
            else:
                self.active = self._advance(self.active,tokens,mask,timestamp,self._sequence,prototype=prototype)
                self.active = replace(self.active,metadata={**self.active.metadata,**metadata})
        else:
            self.shadow = self._advance(self.shadow,tokens,mask,timestamp,self._sequence,prototype=prototype)
            self.candidate = self._advance(self.candidate,tokens,mask,timestamp,self._sequence)
        online = self.shadow if self.candidate is not None else self.active
        frame = FeatureFrame(self._sequence,timestamp,tokens,mask,local,radar,
                             self.semantic_projection(online.event_tokens).detach().clone(),
                             self.heads.evidence(online.event_tokens).detach().clone())
        self.buffer.append(frame)
        self.archive.pin_start = min(self.buffer.pins.values()) if self.buffer.pins else None
        self.archive.append(frame)
        if self.candidate is not None:
            difference,stability,result = self._observe(timestamp)
            score = result.score
            if result.decision == 'confirm':
                self._queue_boundary(timestamp,result.score)
                self.active = replace(self.candidate,corrected=False)
                self._finish_episode('confirm',timestamp)
                decision = 'confirm'
            elif result.decision in ('reject','timeout_reject'):
                self.active = self.shadow
                self._release_pin(self._episode.candidate_id)
                self._finish_episode(result.decision,timestamp)
                decision = result.decision
        self._previous_frame = frame
        if self.buffer.pins:
            self.metrics['max_pinned_updates'] = max(self.metrics['max_pinned_updates'],len(self.buffer.frames))
        self.decision_trace.append(decision)
        return ControllerOutput(self.state,decision,proposal,difference,stability,score,boundary,job,
                                bool(self._boundaries) or self._pending is not None,tuple(self.failures))

    def flush(self, *, timeout_s=30.):
        """Explicit offline/EOF wait; never called by the streaming step."""
        deadline = time.monotonic()+timeout_s
        while self._boundaries or self._pending is not None:
            self.poll()
            if time.monotonic() >= deadline:
                raise TimeoutError('reconciliation still pending; pinned evidence retained')
            time.sleep(.001)
        if self.failures:
            raise RuntimeError('; '.join(self.failures))

    def finish(self, *, end_time=None, timeout_s=30.):
        """EOF tail is explicitly unconfirmed, never a fake Confirm."""
        if self.active is None:
            return None
        if self.candidate is not None:
            self.active = self.shadow
            self._release_pin(self._episode.candidate_id)
            self._finish_episode('eof_reject',self._previous_frame.timestamp)
        self.flush(timeout_s=timeout_s)
        end = self._previous_frame.timestamp+self.config.update_period_s if end_time is None else end_time
        if not math.isfinite(end) or end <= self._previous_frame.timestamp:
            raise ValueError('EOF end must follow last observation')
        tokens,mask = CommitArchive.tensors(self.archive.chunks)
        return FinalizationJob(self.active.event_id,self.active.start_time,end,end,0.,tokens,mask,
                               {**self.active.metadata,'eof_unconfirmed':True},False)

    def close(self):
        if not self._closed:
            self._closed = True
            if self._pending is not None:
                self._pending.future.cancel()
            for task in self._boundaries:
                task.future.cancel()
            self.buffer.pins.clear()
            self.buffer._evict()
            self._executor.shutdown(wait=False,cancel_futures=True)
            self._boundary_executor.shutdown(wait=False,cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
