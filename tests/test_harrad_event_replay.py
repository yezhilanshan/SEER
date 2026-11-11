import numpy as np
import pytest
import torch
from test_harrad_event import tiny_model


def replay(model, cells, instances, chunks):
    from continuous_mmwave.harrad.event_replay import OracleStartReplay
    engine = OracleStartReplay(model, instances)
    predictions = []
    first = 0
    for size in chunks:
        predictions.extend(engine.push(cells[first:first + size], first))
        first += size
    assert first == len(cells)
    return engine, predictions


def test_oracle_stream_matches_instances_across_odd_starts_ends_and_gaps():
    model = tiny_model().eval()
    cells = torch.randn(73, 4, 64)
    # Odd length, odd global start, adjacent equal-class instances, and gaps.
    instances = [dict(start=3, stop=28, label=1, uid='a'),
                 dict(start=28, stop=35, label=1, uid='b'),
                 dict(start=40, stop=70, label=4, uid='c')]
    with torch.no_grad():
        engine, predictions = replay(model, cells, instances, [1, 6, 22, 4, 1, 19, 20])
        for instance in instances:
            x = cells[instance['start']:instance['stop']][None]
            expected = model(x, torch.ones(1, len(x[0]), dtype=torch.bool))
            for p in [p for p in predictions if p['uid'] == instance['uid']]:
                logits = expected['final_logits'] if p['slot'] == 'end' else expected['prefix'][int(p['slot'][1:])]['logits']
                np.testing.assert_allclose(p['logits'], logits[0].numpy(), atol=1e-6, rtol=1e-6)
                assert p['observed_frames'] == p['requested_end'] - instance['start']
                assert p['actual_observation_end'] == p['requested_end']
    assert engine.resets == 3
    assert engine.position == 73 and engine.unlabelled_frames == 11
    assert engine.tail_readouts == 2
    assert len(predictions) == 10


def test_chunks_future_and_unlabelled_values_do_not_change_instance_readouts():
    model = tiny_model().eval()
    cells = torch.randn(45, 4, 64)
    instances = [dict(start=5, stop=12, label=2, uid='a'),
                 dict(start=19, stop=44, label=3, uid='b')]
    with torch.no_grad():
        _, single = replay(model, cells, instances, [45])
        _, chopped = replay(model, cells, instances, [1] * 45)
        altered = cells.clone()
        altered[:5] *= -100
        altered[12:19] += 100
        altered[44:] *= 100
        _, changed = replay(model, altered, instances, [11, 9, 25])
        future = cells.clone()
        future[11:] += 100
        _, perturbed = replay(model, future, instances, [45])
    for a, b, c in zip(single, chopped, changed):
        np.testing.assert_array_equal(a['logits'], b['logits'])
        np.testing.assert_array_equal(a['logits'], c['logits'])
    early = next(p for p in single if p['uid'] == 'a' and p['slot'] == 'u3')
    after = next(p for p in perturbed if p['uid'] == 'a' and p['slot'] == 'u3')
    np.testing.assert_array_equal(early['logits'], after['logits'])


def test_replay_rejects_overlap_and_nonsequential_input():
    from continuous_mmwave.harrad.event_replay import OracleStartReplay
    model = tiny_model().eval()
    with pytest.raises(ValueError, match='overlap'):
        OracleStartReplay(model, [dict(start=1, stop=8, uid='a'), dict(start=7, stop=9, uid='b')])
    engine = OracleStartReplay(model, [dict(start=1, stop=8, uid='a')])
    with pytest.raises(ValueError, match='sequential'):
        engine.push(torch.zeros(2, 4, 64), 1)
