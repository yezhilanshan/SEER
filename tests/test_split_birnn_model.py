from __future__ import annotations

import importlib
import torch
import pytest

torch.set_num_threads(1)


def model():
    torch.manual_seed(19)
    return importlib.import_module('continuous_mmwave.split_birnn.model').SplitBiGRU()


def test_whole_frame_and_irregular_chunk_forward_are_equivalent():
    m = model().eval()
    x = torch.randn(2, 19, 128)
    with torch.no_grad():
        whole, cache, _ = m.forward_chunk(x)
        for cuts in ([1]*19, [3, 7, 1, 8]):
            state, offset, logits, caches = None, 0, [], []
            for size in cuts:
                y, c, state = m.forward_chunk(x[:, offset:offset+size], state)
                logits.append(y); caches.append(c); offset += size
            torch.testing.assert_close(torch.cat(logits, 1), whole, atol=2e-6, rtol=1e-5)
            torch.testing.assert_close(torch.cat(caches, 1), cache, atol=2e-6, rtol=1e-5)
    assert cache.shape == (2, 19, 64)


def test_forward_is_causal_and_cached_backward_equals_joint_result():
    m = model().eval()
    x = torch.randn(1, 13, 128)
    changed = x.clone(); changed[:, 7:] += torch.randn_like(changed[:, 7:]) * 10
    with torch.no_grad():
        out = m.forward_sequence(x)
        other = m.forward_sequence(changed)
        torch.testing.assert_close(out['forward'][:, :7], other['forward'][:, :7])
        torch.testing.assert_close(m.correct(out['cache']), out['corrected'])
        assert out['corrected'].shape == (1, 13, 9)


def test_shared_head_and_average_match_independent_reverse_reference():
    m = model().eval(); x = torch.randn(1, 9, 128)
    with torch.no_grad():
        _, cache, _ = m.forward_chunk(x)
        forward_states = list(cache.split(16, dim=-1))
        backward = None
        for i, gru in enumerate(m.backward_grus):
            inp = forward_states[0] if i == 0 else torch.cat([forward_states[i], backward], -1)
            reverse, _ = gru(inp.flip(1))
            backward = reverse.flip(1)
        want = m.head((forward_states[3] + backward) / 2)
        torch.testing.assert_close(m.correct(cache), want)
    assert sum(p.numel() for p in m.forward_grus.parameters()) == 4896
    assert sum(p.numel() for p in m.backward_grus.parameters()) == 6432
    assert sum(p.numel() for p in m.head.parameters()) == 153
    assert len([n for n, p in m.named_parameters() if n.endswith('head.weight')]) == 1


def test_backward_only_loss_reaches_every_forward_layer_and_cnn():
    m = model().train()
    out = m.forward_sequence(torch.randn(2, 17, 128))
    out['corrected'].square().mean().backward()
    for component in [m.cnn, *m.forward_grus, *m.backward_grus, m.head]:
        grads = [p.grad for p in component.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(float(g.abs().sum()) for g in grads) > 0


def test_parameter_audit_separates_shared_head_and_cnn_mismatch():
    m = model()
    audit = importlib.import_module('continuous_mmwave.split_birnn.model').parameter_audit(m)
    assert audit['backward_with_shared_head'] == 6585
    assert audit['unique_split_total'] == audit['forward_branch'] + 6432
    assert audit['cnn_trainable'] == sum(p.numel() for p in m.cnn.parameters())
    assert audit['cnn_buffers'] > 0
    assert audit['cnn_exact_paper_match'] is (audit['cnn_trainable'] == 8301)


def test_standard_bigrus_use_past_and_future_at_upper_layers():
    mod = importlib.import_module('continuous_mmwave.split_birnn.model')
    b = mod.BiGRUBaseline().eval()
    assert b.gru.bidirectional and b.gru.num_layers == 3
    assert sum(p.numel() for p in b.gru.parameters()) == 12864
    assert b(torch.randn(2, 17, 128)).shape == (2, 17, 9)


def test_combo_matches_exact_smoothing_and_global_dice_with_ignore_mask():
    loss = importlib.import_module('continuous_mmwave.split_birnn.loss')
    probs = torch.tensor([[[.6]+[.05]*8, [.2,.4]+[.4/7]*7, [1/9]*9]])
    labels = torch.tensor([[0, 1, -100]])
    value, parts = loss.combo_loss(probs.log(), labels)
    logp = probs[0, :2].log()
    want_ce = -(.8*(logp[0,0]+logp[1,1]) + .025*(logp[0,1:].sum()+logp[1].sum()-logp[1,1]))/2
    want_dice = torch.tensor(1-(2*(.6+.4)+1)/(2*2+1))
    torch.testing.assert_close(parts['cce'], want_ce)
    torch.testing.assert_close(parts['dice'], want_dice)
    torch.testing.assert_close(value, want_ce + want_dice)
    changed = probs.log().clone(); changed[:, 2] += 100
    torch.testing.assert_close(loss.combo_loss(changed, labels)[0], value)


def test_loss_sensitivity_modes_are_explicit_and_all_ignored_is_differentiable():
    loss = importlib.import_module('continuous_mmwave.split_birnn.loss')
    x = torch.randn(1, 3, 9, requires_grad=True); y = torch.tensor([[0, 0, 1]])
    global_loss = loss.combo_loss(x, y)[0]
    class_loss = loss.combo_loss(x, y, dice_mode='per_class')[0]
    ce = loss.combo_loss(x, y, dice_mode='none')[0]
    assert not torch.isclose(global_loss, class_loss)
    assert global_loss > ce
    z = loss.combo_loss(x, torch.full_like(y, -100))[0]; z.backward()
    assert z.item() == 0 and torch.equal(x.grad, torch.zeros_like(x))
    with pytest.raises(ValueError):
        loss.combo_loss(x, y, dice_mode='undefined')
