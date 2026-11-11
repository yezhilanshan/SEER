"""Ordered background Commit → Memory, with independent generation-only QA."""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
import torch

from ..parrad_memory import MemoryUnit


class FrozenQA:
    """Adapt a copy of retrieved tokens; never modify tokens used by original P."""
    def __init__(self,qa_adapter,answerer):
        self.adapter,self.answerer = qa_adapter,answerer
        qa_adapter.eval().requires_grad_(False)

    @torch.inference_mode()
    def answer(self,context,query):
        parameter = next(self.adapter.parameters(),None)
        adapted_input = (context.to(device=parameter.device,dtype=parameter.dtype)
                         if parameter is not None else context)
        return self.answerer(self.adapter(adapted_input.clone()),query)

    @classmethod
    def from_native_reader(cls,qa_adapter,reader,tokenizer,*,max_new_tokens=96):
        from ..stages.parrad_stage4 import generate_from_memory
        reader.eval().requires_grad_(False)
        return cls(qa_adapter,lambda context,query:generate_from_memory(reader,tokenizer,context,query,max_new_tokens=max_new_tokens))


class V25Pipeline:
    def __init__(self,runtime,memory,*,qa=None,captioner=None,max_pending_jobs=64):
        if max_pending_jobs <= 0:
            raise ValueError('positive finalization queue bound required')
        self.runtime,self.memory,self.qa,self.captioner = runtime,memory,qa,captioner
        self.max_pending_jobs = max_pending_jobs
        self._executor = ThreadPoolExecutor(max_workers=1,thread_name_prefix='v25-commit')
        self._lock = threading.RLock()
        self._futures = []
        self.pending_finalizations = []
        self.retained_jobs = {}
        self.completed_events,self.failed_jobs,self.caption_failures = [],[],[]
        self._last_submitted_end = None
        self._closed,self._finished = False,False
        for module in (memory.compressor,memory.projection):
            module.eval().requires_grad_(False)

    @torch.inference_mode()
    def _finalize(self,job):
        controller = self.runtime.controller
        result = controller.event_former.commit_refine(job.radar_tokens,job.radar_mask,controller.prompt_cache)
        tokens = result.event_tokens[0].detach().clone()
        key = controller.semantic_projection(tokens[None])[0].detach().clone()
        caption = None
        if self.captioner is not None:
            try:
                caption = self.captioner(tokens.clone())
            except Exception as error:
                self.caption_failures.append(dict(event_id=job.event_id,error=str(error)))
        unit = MemoryUnit(job.event_id,tokens,job.start_time,job.end_time,job.confirmed_at,
                          job.confidence,key,caption=caption,status='captioned' if caption is not None else 'refined')
        with self._lock:
            # Compression/promotion also run in this background worker.
            self.memory.add(unit)
            self.completed_events.append(dict(event_id=job.event_id,start=job.start_time,end=job.end_time,
                confirmed_at=job.confirmed_at,confidence=job.confidence,confirmed=job.confirmed))
        return job.event_id

    def _guarded_finalize(self,job):
        if self.failed_jobs:
            self.failed_jobs.append(dict(event_id=job.event_id,error='blocked by preceding failed Commit'))
            raise RuntimeError('preceding Commit failed; later events are not silently committed')
        try:
            return self._finalize(job)
        except Exception as error:
            self.failed_jobs.append(dict(event_id=job.event_id,error=f'{type(error).__name__}: {error}'))
            raise

    def poll(self):
        self.runtime.controller.poll()
        pending = []
        for future in self._futures:
            if future.done():
                try:
                    completed_id = future.result()
                    self.retained_jobs.pop(completed_id,None)
                except Exception:
                    pass  # failure is retained with event ID by _guarded_finalize
            else:
                pending.append(future)
        self._futures = pending
        # Transfer ownership before pumping. Jobs accepted by the controller
        # must survive a full queue or a preceding worker failure.
        for job in self.runtime.controller.drain_finalizations():
            self._accept(job)
        self._pump()

    def _accept(self,job):
        if job.event_id in self.retained_jobs or any(e['event_id'] == job.event_id for e in self.completed_events):
            raise ValueError('duplicate Commit event')
        if self._last_submitted_end is not None and job.start_time < self._last_submitted_end-1e-9:
            raise ValueError('Commit jobs must be chronological and nonoverlapping')
        self.retained_jobs[job.event_id] = job
        self.pending_finalizations.append(job)
        self._last_submitted_end = job.end_time

    def _pump(self):
        if self.failed_jobs or self._closed:
            return
        while self.pending_finalizations and len(self._futures) < self.max_pending_jobs:
            job = self.pending_finalizations.pop(0)
            self._futures.append(self._executor.submit(self._guarded_finalize,job))

    def submit(self,job):
        if self._closed:
            raise RuntimeError('pipeline closed')
        self.poll()
        if self.failed_jobs:
            raise RuntimeError('Commit pipeline failed: '+str(self.failed_jobs))
        if len(self._futures)+len(self.pending_finalizations) >= self.max_pending_jobs:
            raise BufferError('Commit queue full; no event has been dropped')
        self._accept(job)
        self._pump()

    def step(self,*args,**kwargs):
        if self._finished or self._closed:
            raise RuntimeError('recording finished or pipeline closed')
        self.poll()
        if self.failed_jobs:
            raise RuntimeError('Commit pipeline failed: '+str(self.failed_jobs))
        if len(self._futures)+len(self.pending_finalizations) >= self.max_pending_jobs:
            raise BufferError('Commit queue full before consuming next radar update')
        result = self.runtime.step(*args,**kwargs)
        self.poll()
        return result

    def flush(self,*,timeout_s=30.):
        deadline = time.monotonic()+timeout_s
        try:
            self.runtime.controller.flush(timeout_s=timeout_s)
        except Exception:
            self.poll()  # preserve all completed intervals before propagating failure
            raise
        self.poll()
        while self._futures or self.pending_finalizations:
            self.poll()
            if self.failed_jobs:
                raise RuntimeError('Commit pipeline failed: '+str(self.failed_jobs))
            if time.monotonic() >= deadline:
                raise TimeoutError('Commit pending; no job discarded')
            time.sleep(.001)
        if self.failed_jobs:
            raise RuntimeError('Commit pipeline failed: '+str(self.failed_jobs))

    @torch.inference_mode()
    def context(self,query_key,*,now,include_active=True,**retrieval):
        c = self.runtime.controller
        branch = c.shadow if c.candidate is not None else c.active
        current = branch.event_tokens.clone() if include_active and not self._finished and branch is not None else None
        with self._lock:
            return self.memory.context(query_key,now=now,current=current,**retrieval)

    def answer(self,query,query_key,*,now,include_active=True):
        if self.qa is None:
            raise ValueError('attach an explicit independent QA reader')
        context,units = self.context(query_key,now=now,include_active=include_active)
        return dict(answer=self.qa.answer(context,query),selected_ids=[u.uid for u in units],
                    context_tokens=context.shape[1],includes_active=include_active and not self._finished)

    def finish(self,*,end_time=None,include_unconfirmed_tail=False,timeout_s=30.):
        if self._finished:
            raise RuntimeError('recording already finished')
        tail = self.runtime.controller.finish(end_time=end_time,timeout_s=timeout_s)
        if tail is not None and include_unconfirmed_tail:
            self.flush(timeout_s=timeout_s)
            self.submit(tail)
        self.flush(timeout_s=timeout_s)
        self._finished = True
        return dict(architecture='v25-sequential-evidence',status='interfaces-complete',
            confirmed_events=sum(e['confirmed'] for e in self.completed_events),
            eof_unconfirmed_events=sum(not e['confirmed'] for e in self.completed_events),
            events=list(self.completed_events),controller_metrics=dict(self.runtime.controller.metrics),
            failures=list(self.failed_jobs),caption_failures=list(self.caption_failures),
            formal_go=False,is_unseen_subject_test=False,realtime_validated=False)

    def snapshot(self):
        with self._lock:
            return self.memory.snapshot()

    def close(self):
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=False,cancel_futures=True)
            self.runtime.controller.close()

    def __enter__(self):
        return self

    def __exit__(self,*args):
        self.close()
