import pytest
import torch


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def make_model(mode='lif', **kwargs):
    from continuous_mmwave.event_gating.config import GateConfig
    from continuous_mmwave.event_gating.model import EventGate
    torch.manual_seed(31)
    return EventGate(GateConfig(feature_dim=4, hidden_dim=8, num_classes=3,
                                buffer_frames=8, recent_tau_s=.2, **kwargs), mode)


@pytest.mark.parametrize('mode', ['cif', 'lif'])
def test_stream_chunks_cover_frames_once_and_match_whole(mode):
    from continuous_mmwave.event_gating.streaming import EventStream
    model = make_model(mode).eval()
    x = torch.randn(47, 4)
    whole = EventStream(model, 'r')
    a = whole.push(x) + whole.finish()
    chunks = EventStream(model, 'r')
    b = []
    for start, stop in [(0, 1), (1, 14), (14, 25), (25, 47)]:
        b.extend(chunks.push(x[start:stop]))
    b.extend(chunks.finish())
    assert a == b
    assert a[0]['start_frame'] == 0 and a[-1]['stop_frame'] == 47
    assert sum(e['stop_frame']-e['start_frame'] for e in a) == 47
    for prev, nxt in zip(a, a[1:]):
        assert prev['stop_frame'] == nxt['start_frame']
    for event in a:
        expected = x[event['start_frame']:event['stop_frame']].mean(0)
        torch.testing.assert_close(torch.tensor(event['feature_mean']), expected, atol=1e-6, rtol=1e-5)
    assert chunks.finish() == []
    with pytest.raises(ValueError):
        chunks.push(x[:1])


def test_lif_training_without_hard_spikes_still_updates_gate_and_preserves_repeats():
    model = make_model('lif', theta=100.)
    x = torch.randn(30, 4)
    result = model.training_loss(x, [1, 1, 2])
    assert result['hard_count'] == 0
    assert torch.isfinite(result['loss'])
    result['loss'].backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.driver.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.classifier.parameters())
    with pytest.raises(ValueError, match='CTC'):
        model.training_loss(x[:2], [1, 1])


@pytest.mark.parametrize('mode', ['cif', 'lif'])
def test_padding_and_future_values_cannot_change_prefix(mode):
    model = make_model(mode)
    x = torch.randn(20, 4)
    a = model.training_loss(x, [1, 2], length=12)
    changed = x.clone()
    changed[12:] = float('nan')
    b = model.training_loss(changed, [1, 2], length=12)
    torch.testing.assert_close(a['loss'], b['loss'])
    first = model.drive_trace(x[:8])
    second = model.drive_trace(x)[:8]
    torch.testing.assert_close(first, second)


def test_backtracking_buffer_retains_old_summary_without_duplicating_frames():
    from continuous_mmwave.event_gating.streaming import FrameBuffer
    buffer = FrameBuffer(4, 2)
    x = torch.arange(20.).reshape(10, 2)
    for index, h in enumerate(x):
        buffer.append(index, h)
    assert buffer.stored_frames == 4
    total, count = buffer.split(8)
    assert count == 8
    torch.testing.assert_close(total / count, x[:8].mean(0))
    total, count = buffer.summary()
    assert count == 2
    torch.testing.assert_close(total / count, x[8:].mean(0))


def test_long_constant_stream_has_bounded_state_and_only_eof_event():
    from continuous_mmwave.event_gating.streaming import EventStream
    model = make_model().eval()
    stream = EventStream(model, 'long')
    for _ in range(20):
        assert stream.push(torch.ones(50, 4)) == []
        assert stream.stored_frames <= 8
    events = stream.finish()
    assert len(events) == 1 and events[0]['stop_frame'] == 1000
    assert events[0]['closure'] == 'eof'


def test_frame_baseline_debounce_is_causal_and_keeps_eof_frames():
    from continuous_mmwave.event_gating.streaming import EventStream
    model = make_model('frame-baseline').eval()
    model.classifier = torch.nn.Linear(4,3,bias=False)
    with torch.no_grad():
        model.classifier.weight.copy_(torch.tensor([[1.,0.,0.,0.],[0.,1.,0.,0.],[0.,0.,1.,0.]]))
    x = torch.tensor([[4.,0.,0.,0.]]*3+[[0.,4.,0.,0.]]*4)
    stream = EventStream(model,'r',minimum_segment_frames=2)
    assert stream.push(x[:4]) == []
    events = stream.push(x[4:])+stream.finish()
    assert [e['class_id'] for e in events] == [0,1]
    assert [(e['start_frame'],e['stop_frame']) for e in events] == [(0,3),(3,7)]
    assert events[0]['confirmed_at_frame'] == 5


def test_recent_variance_uses_elapsed_time_and_is_available_to_evidence_estimator():
    import math
    model = make_model('lif',theta=100.).eval()
    state,_ = model.lif_step(torch.zeros(4))
    state,_ = model.lif_step(torch.ones(4),state,dt=.1)
    rate=1-math.exp(-.1/.2)
    torch.testing.assert_close(state.recent_variance,torch.full((4,),rate*(1-rate)))
