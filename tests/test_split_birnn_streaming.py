from __future__ import annotations
import importlib
import numpy as np
import torch
import pytest


def modules():
    return (importlib.import_module('continuous_mmwave.split_birnn.model'),
            importlib.import_module('continuous_mmwave.split_birnn.streaming'))


def test_window_schedule_has_no_implicit_tail_flush():
    _, s = modules()
    assert s.correction_windows(12, 5, 2) == [(0,5), (2,7), (4,9), (6,11)]
    assert s.correction_windows(4, 5, 2) == []
    with pytest.raises(ValueError):
        s.correction_windows(12, 5, 6)


def test_first_final_counts_and_delays_match_independent_window_reference():
    m, s = modules(); torch.manual_seed(8)
    net = m.SplitBiGRU().eval(); x = torch.randn(1,12,128)
    with torch.no_grad():
        f, cache, _ = net.forward_chunk(x)
        first, final = f.clone(), f.clone()
        counts = np.zeros(12, dtype=int); first_end = np.full(12,-1); last_end = np.full(12,-1)
        for start, end in [(0,5),(2,7),(4,9),(6,11)]:
            corrected = net.correct(cache[:,start:end])
            for j in range(start,end):
                if counts[j] == 0:
                    first[:,j] = corrected[:,j-start]; first_end[j] = end-1
                final[:,j] = corrected[:,j-start]; counts[j] += 1; last_end[j] = end-1
        out = s.replay_split(net, x, window=5, hop=2, chunk_frames=3)
    torch.testing.assert_close(out['forward_logits'], f)
    torch.testing.assert_close(out['first_logits'], first)
    torch.testing.assert_close(out['final_logits'], final)
    assert out['correction_counts'].tolist() == [1,1,2,2,3,2,3,2,2,1,1,0]
    np.testing.assert_array_equal(out['first_end_frame'], first_end)
    np.testing.assert_array_equal(out['final_end_frame'], last_end)
    assert out['first_delay_s'][0] == pytest.approx(.36)
    assert out['first_delay_s'][4] == pytest.approx(0)
    assert np.isnan(out['first_delay_s'][11])
    assert out['final_delay_s'][4] == pytest.approx(.36)
    torch.testing.assert_close(out['final_logits'][:,11], f[:,11])


def test_short_recording_and_reset_never_drop_frames():
    m, s = modules(); net = m.SplitBiGRU().eval(); x = torch.randn(1,4,128)
    a = s.replay_split(net,x,window=5,hop=2)
    _ = s.replay_split(net,torch.randn(1,15,128),window=5,hop=2)
    b = s.replay_split(net,x,window=5,hop=2)
    torch.testing.assert_close(a['final_logits'], b['final_logits'])
    torch.testing.assert_close(a['final_logits'], a['forward_logits'])
    assert not a['correction_counts'].any()
    assert a['final_logits'].shape == (1,4,9)


def test_streaming_requires_eval_and_baseline_uses_identical_window_endpoints():
    m, s = modules(); x = torch.randn(1,12,128)
    with pytest.raises(ValueError):
        s.replay_split(m.SplitBiGRU().train(),x,window=5,hop=2)
    b = m.BiGRUBaseline().eval()
    result = s.replay_baseline(b,x,window=5,hop=2)
    assert result['events'] == [{'start_frame':i,'end_frame_exclusive':i+5,'emission_frame':i+4}
                                for i in [0,2,4,6]]
    assert result['available_mask'].tolist() == [True]*11+[False]
    assert torch.isnan(result['final_logits'][:,11]).all()
    with torch.no_grad():
        torch.testing.assert_close(result['final_logits'][:,6:11], b(x[:,6:11]))


def test_metric_views_include_uncorrected_tail_and_ignore_only_labels():
    _, s = modules()
    logits = torch.tensor([[[9.,0.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,9.,0.,0.,0.,0.,0.,0.,0.],
                           [0.,9.,0.,0.,0.,0.,0.,0.,0.]]])
    metrics = s.classification_metrics(logits, torch.tensor([[0,-100,0]]))
    assert metrics['valid_frames'] == 2 and metrics['accuracy'] == .5
    assert metrics['weighted_f1'] == pytest.approx(2/3)
    assert np.asarray(metrics['confusion_matrix'])[0,:2].tolist() == [1,1]
