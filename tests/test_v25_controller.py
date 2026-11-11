import threading
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from continuous_mmwave.models.transition import PrototypeState, update_prototype
from continuous_mmwave.v25.evidence import EvidenceConfig
from continuous_mmwave.v25.controller import V25Controller, ControllerConfig
from continuous_mmwave.v25.buffer import FeatureBuffer, FeatureFrame, CommitArchive
from continuous_mmwave.v25.boundary import localize_boundary


class Former(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(radar_tokens_per_update=4, max_commit_radar_tokens=256)
        self.state_gate = SimpleNamespace(architecture='v25-age-free-tanh')
        self.calls = 0

    def initial_state(self, batch):
        return torch.zeros(batch, 8, 2)

    def build_prompt_cache(self, batch):
        return None

    def stream_update(self, previous, radar, maturity, mask, cache):
        assert maturity == 1.  # age cannot enter the trained gate
        self.calls += 1
        # A recurrence whose corrected origin can be checked exactly.
        return SimpleNamespace(event_tokens=previous*.5+radar.mean(1)[:,None])

    def commit_refine(self, radar, mask, cache=None):
        return SimpleNamespace(event_tokens=radar.mean(1)[:,None].expand(-1,8,-1))


class Projection(nn.Module):
    def forward(self, tokens):
        return F.normalize(tokens.mean(1),dim=-1)


class Heads(nn.Module):
    def __init__(self, difference=.99):
        super().__init__()
        self.difference = difference

    def local(self, previous, current, mask):
        assert mask.any()
        return torch.tensor([20. if not torch.equal(previous,current) else -20.])

    def radar(self, tokens, mask, prototype, period):
        value = tokens.mean(1)
        updated = PrototypeState.initialize(value) if prototype is None else update_prototype(prototype,value,period,2.)
        return torch.tensor([-20.]), updated

    def semantic(self, active, candidate):
        return torch.tensor([torch.logit(torch.tensor(self.difference))])

    def evidence(self, tokens):
        return tokens.mean(1).sigmoid()


def controller(difference=.99, **config):
    return V25Controller(Former(),Projection(),Heads(difference),
        ControllerConfig(proposal_persistence=1, boundary_local_weight=1.,boundary_radar_weight=0.,
                         boundary_semantic_weight=0.,**config),
        EvidenceConfig(h_plus=.1,h_minus=-.1,warmup_s=.4,timeout_s=1.,tau_stability=.1,evidence_weight=0.))


def step(c, t, value, mask=None):
    tokens = torch.tensor(value,dtype=torch.float32)[None,None].expand(1,4,-1).clone()
    return c.step_tokens(tokens,torch.ones(1,4,dtype=torch.bool) if mask is None else mask,t)


def sequence(c):
    step(c,0.,[1.,0.])
    assert step(c,.2,[0.,1.]).decision == 'proposal'
    step(c,.4,[0.,1.])
    return step(c,.6,[0.,1.])


def start_rebuild(c):
    c._boundaries[0].future.result(timeout=2)
    c.poll()


def test_confirm_rebuilds_from_refined_boundary_and_same_watermark():
    with controller() as c:
        output = sequence(c)
        assert output.decision == 'confirm'
        assert not c.active.corrected
        c.flush()
        jobs = c.drain_finalizations()
        assert jobs[0].end_time == c.active.start_time == .2
        assert c.episodes[0].refined_boundary == pytest.approx(.2)
        torch.testing.assert_close(c.active.event_tokens,torch.tensor([0.,1.75])[None,None].expand(1,8,2))
        assert c.active.prototype.count == 3
        assert c.active.watermark == 3
        assert c.active.corrected
        assert not c.buffer.pins
        assert c.metrics['atomic_swaps'] == 1


def test_reject_matches_unforked_shadow_and_never_replays():
    with controller(difference=.01) as c:
        output = sequence(c)
        assert output.decision == 'reject'
        expected = torch.zeros(1,8,2)
        for value in ([1.,0.],[0.,1.],[0.,1.],[0.,1.]):
            expected = expected*.5+torch.tensor(value)[None,None]
        torch.testing.assert_close(c.active.event_tokens,expected)
        assert c.active.prototype.count == 4
        assert c.metrics['rebuild_updates'] == 0
        assert not c.buffer.pins


def test_slow_rebuild_keeps_streaming_and_catches_tail():
    started, release = threading.Event(), threading.Event()
    with controller() as c:
        original = c._rebuild
        def blocked(*args):
            started.set()
            assert release.wait(5)
            return original(*args)
        c._rebuild = blocked
        sequence(c)
        start_rebuild(c)
        assert started.wait(2)
        step(c,.8,[0.,1.])
        step(c,1.,[0.,1.])
        assert c.active.watermark == 5
        assert not c.active.corrected
        assert c.buffer.pins
        release.set()
        c.flush()
        torch.testing.assert_close(c.active.event_tokens,torch.tensor([0.,1.9375])[None,None].expand(1,8,2))
        assert c.active.watermark == 5
        assert c.active.prototype.count == 5
        assert not c.buffer.pins


def test_reset_discards_running_result_and_releases_pins():
    started, release = threading.Event(), threading.Event()
    c = controller()
    original = c._rebuild
    def blocked(*args):
        started.set()
        release.wait(5)
        return original(*args)
    c._rebuild = blocked
    try:
        sequence(c)
        start_rebuild(c)
        assert started.wait(2)
        c.reset()
        assert not c.buffer.pins
        step(c,0.,[3.,0.])
        release.set()
        c.flush()
        torch.testing.assert_close(c.active.event_tokens,torch.tensor([3.,0.])[None,None].expand(1,8,2))
    finally:
        release.set()
        c.close()


def test_worker_failure_visible_and_no_fake_corrected_state():
    with controller() as c:
        def fail(*args):
            raise RuntimeError('injected failure')
        c._rebuild = fail
        sequence(c)
        with pytest.raises(RuntimeError,match='injected failure'):
            c.flush()
        assert not c.active.corrected
        assert not c.buffer.pins
        assert c.metrics['reconciliation_failures'] == 1


def test_continuous_confirm_discards_older_generation():
    started, release = threading.Event(), threading.Event()
    with controller() as c:
        original = c._rebuild
        calls = [0]
        def blocked(*args):
            calls[0] += 1
            if calls[0] == 1:
                started.set()
                release.wait(5)
            return original(*args)
        c._rebuild = blocked
        sequence(c)
        start_rebuild(c)
        assert started.wait(2)
        assert step(c,.8,[1.,0.]).decision == 'proposal'
        step(c,1.,[1.,0.])
        assert step(c,1.2,[1.,0.]).decision == 'confirm'
        current_id = c.active.event_id
        release.set()
        c.flush()
        assert c.active.event_id == current_id
        assert c.active.start_time == pytest.approx(.8)
        torch.testing.assert_close(c.active.event_tokens,torch.tensor([1.75,0.])[None,None].expand(1,8,2))
        assert not c.buffer.pins


def test_disjoint_masks_and_irregular_timestamps():
    with controller(difference=.01) as c:
        step(c,0.,[1.,0.],torch.tensor([[1,1,0,0]],dtype=torch.bool))
        output = step(c,.3,[0.,1.],torch.tensor([[0,0,1,1]],dtype=torch.bool))
        assert output.decision == 'none'
        with pytest.raises(ValueError,match='increasing'):
            step(c,.3,[0.,1.])


def frame(i,local=0.,key=(1.,0.)):
    return FeatureFrame(i,i*.2,torch.full((1,4,2),float(i)),torch.ones(1,4,dtype=torch.bool),local,0.,torch.tensor([key]))


def test_buffer_pinning_and_release_eviction():
    buffer = FeatureBuffer(retention_s=.5)
    buffer.append(frame(0))
    buffer.pin('candidate',0.)
    for i in range(1,20):
        buffer.append(frame(i))
    assert len(buffer.frames) == 20
    buffer.release('candidate')
    assert len(buffer.frames) <= 4
    assert not buffer.pins


@pytest.mark.parametrize('peak', [2,4])
def test_boundary_can_move_before_or_after_proposal(peak):
    rows = [frame(i,local=float(i == peak)) for i in range(6)]
    result = localize_boundary(rows,proposal_time=.6,confirm_time=1.,active_start=0.,
                               delta_pre_s=.4,weights=(1.,0.,0.))
    assert result.timestamp == pytest.approx(peak*.2)


def test_commit_archive_bounded_and_recent_boundary_split_exact():
    archive = CommitArchive(max_tokens=16, preserve_s=.6)
    for i in range(100):
        archive.append(frame(i))
    # Temporarily retains recent rewind window in addition to the pooled past.
    assert len(archive.chunks) <= 8
    before, after = archive.partition(19.6)
    assert before[-1].end < 19.6
    assert after[0].start == pytest.approx(19.6)


def test_boundary_refiner_is_background_and_confirm_never_waits(monkeypatch):
    import concurrent.futures
    import continuous_mmwave.v25.controller as module
    started,release = threading.Event(),threading.Event()
    original = module.localize_boundary
    def blocked(*args,**kwargs):
        started.set(); assert release.wait(5)
        return original(*args,**kwargs)
    monkeypatch.setattr(module,'localize_boundary',blocked)
    with controller() as c, concurrent.futures.ThreadPoolExecutor(1) as executor:
        try:
            future = executor.submit(sequence,c)
            assert started.wait(2)
            # The owning streaming call must return while localization is blocked.
            output = future.result(timeout=.2)
            assert output.decision == 'confirm'
            assert c.active.start_time == .2  # provisional proposal origin
            assert not c.active.corrected
            step(c,.8,[0.,1.])
        finally:
            release.set()
        c.flush()
        assert c.drain_finalizations()[0].end_time == .2
        assert c.active.watermark == 4


def test_archive_pin_is_released_without_another_stream_update():
    with controller() as c:
        sequence(c)
        c.flush()
        assert not c.buffer.pins
        assert c.archive.pin_start is None


def test_boundary_failure_stops_before_consuming_more_observations(monkeypatch):
    import continuous_mmwave.v25.controller as module
    def fail(*args,**kwargs):
        raise RuntimeError('boundary exploded')
    monkeypatch.setattr(module,'localize_boundary',fail)
    with controller() as c:
        sequence(c)
        with pytest.raises(RuntimeError,match='boundary exploded'):
            c.flush()
        before = c._sequence
        with pytest.raises(RuntimeError,match='boundary exploded'):
            step(c,.8,[0.,1.])
        assert c._sequence == before
        assert not c.active.corrected
