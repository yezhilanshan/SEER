import numpy as np
import pytest
import torch


def test_boundary_targets_preserve_instances_and_unknown_masks():
    from continuous_mmwave.harrad.boundary import boundary_targets
    # Adjacent instances have the same class: still a distinct start/end.
    rows = [dict(start=3, stop=10, label=2), dict(start=10, stop=15, label=2)]
    data = boundary_targets(19, rows)
    assert data['cutoffs'].tolist() == list(range(2, 19, 2)) + [19]
    assert np.flatnonzero(data['targets'][:, 0]).tolist() == [1, 5]
    assert np.flatnonzero(data['targets'][:, 1]).tolist() == [4, 7]
    assert data['mask'][1, 0]  # mixed unknown/known Update can have a positive
    assert not data['mask'][0].any() and not data['mask'][-1].any()
    assert not data['mask'][2, 0]  # one-Update exclusion around a start
    assert data['mask'][3, 0] and not data['targets'][3, 0]
    assert data['truth']['start'].tolist() == [3, 10]
    assert data['truth']['end'].tolist() == [10, 15]
    assert data['known_frames'].sum() == 12


def test_features_and_head_are_causal_and_chunk_independent():
    from continuous_mmwave.harrad.boundary import continuous_inputs, BoundaryHead
    h = np.random.default_rng(5).normal(size=(19, 128)).astype('float32')
    x, cutoffs = continuous_inputs(h)
    altered = h.copy(); altered[8:] += 100
    y, _ = continuous_inputs(altered)
    np.testing.assert_array_equal(x[:4], y[:4])
    np.testing.assert_array_equal(x[0, 128:], np.zeros(128))
    np.testing.assert_allclose(x[1, 128:], h[3] - h[1])
    assert cutoffs[-1] == 19 and x.shape == (10, 256)
    torch.manual_seed(42); model = BoundaryHead().eval()
    assert sum(p.numel() for p in model.parameters()) == 33154
    with torch.no_grad():
        whole = model(torch.from_numpy(x))
        chunks = torch.cat([model(torch.from_numpy(x[:3])), model(torch.from_numpy(x[3:]))])
    torch.testing.assert_close(whole, chunks)


def test_loss_ignores_unknown_targets_and_has_valid_gradients():
    from continuous_mmwave.harrad.boundary import masked_boundary_loss
    logits = torch.zeros(4, 2, requires_grad=True)
    mask = torch.tensor([[1, 0], [0, 0], [1, 1], [0, 1]], dtype=torch.bool)
    target = torch.zeros_like(logits); target[0, 0] = 1
    a = masked_boundary_loss(logits, target, mask, torch.tensor([2., 3.]))
    changed = target.clone(); changed[~mask] = 1
    b = masked_boundary_loss(logits, changed, mask, torch.tensor([2., 3.]))
    torch.testing.assert_close(a, b)
    a.backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad[~mask].any()


def test_decoder_emits_now_on_rising_edge_without_future_peak_search():
    from continuous_mmwave.harrad.boundary import decode_boundaries
    p = np.array([.1, .8, .9, .1, .8, .1, .8])
    cutoffs = np.arange(2, 16, 2)
    assert decode_boundaries(p, cutoffs, .5).tolist() == [4, 10, 14]
    assert decode_boundaries(p[:3], cutoffs[:3], .5).tolist() == [4]
    assert decode_boundaries(np.ones(7), cutoffs, .5).tolist() == [2]


def test_matching_is_one_to_one_and_unknown_predictions_are_unscored():
    from continuous_mmwave.harrad.boundary import score_events, independent_event_counts
    known = np.zeros(35, dtype=bool); known[4:25] = True
    metrics = score_events(np.array([4, 8, 12, 20, 30]), np.array([6, 10, 21]), known, 2)
    assert (metrics['tp'], metrics['fp'], metrics['fn'], metrics['unscored_unknown']) == (3, 1, 0, 1)
    assert metrics['precision'] == .75 and metrics['recall'] == 1
    assert independent_event_counts(np.array([4, 8, 12, 20, 30]), np.array([6, 10, 21]), known, 2) == (3, 1, 0, 1)
    # The same prediction may not match both nearby true boundaries.
    result = score_events(np.array([10]), np.array([9, 11]), known, 2)
    assert result['tp'] == 1 and result['fn'] == 1
    with pytest.raises(ValueError, match='overlap'):
        from continuous_mmwave.harrad.boundary import boundary_targets
        boundary_targets(20, [dict(start=1, stop=10), dict(start=9, stop=12)])


def test_unresolvable_same_update_boundaries_are_not_silently_merged():
    from continuous_mmwave.harrad.boundary import boundary_targets
    with pytest.raises(ValueError, match='resolution'):
        boundary_targets(4, [dict(start=0, stop=1), dict(start=1, stop=2)])


def test_independent_matching_uses_same_earliest_pair_when_dp_cardinality_ties():
    from continuous_mmwave.harrad.boundary import boundary_targets, score_events, independent_event_counts
    known = boundary_targets(12, [dict(start=2, stop=7), dict(start=8, stop=9)])['known_frames']
    prediction = np.array([6, 10]); truth = np.array([2, 8])
    primary = score_events(prediction, truth, known, 2)
    assert (primary['tp'], primary['fp'], primary['fn'], primary['unscored_unknown']) == (1, 0, 1, 1)
    assert independent_event_counts(prediction, truth, known, 2) == (1, 0, 1, 1)
