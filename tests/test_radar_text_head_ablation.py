import numpy as np
import pytest
import torch


def test_factor_grid_changes_only_requested_factors_within_panel():
    from continuous_mmwave.radar_text.head_ablation import arm_specs
    for panel in ('original_optimizer', 'probe_replay'):
        specs = arm_specs(panel)
        assert [(s['epochs'], s['weighted'], s['standardized']) for s in specs] == [
            (3, False, False), (20, False, False), (3, True, False),
            (3, False, True), (20, True, True)]
        assert len({(s['lr'], s['weight_decay'], s['batch_size'], s['grad_clip'], s['initialization']) for s in specs}) == 1


def test_standardized_warm_start_preserves_raw_logits():
    from continuous_mmwave.radar_text.head_ablation import initialize_head
    torch.manual_seed(8)
    raw = torch.nn.Linear(4, 3)
    x = torch.randn(21, 4) * 3 + 2
    mean, std = x.mean(0), x.std(0, unbiased=False)
    spec = dict(initialization='checkpoint', standardized=True, seed=17)
    head = initialize_head(spec, raw.state_dict(), mean, std, 'cpu')
    torch.testing.assert_close(head((x-mean)/std), raw(x), atol=2e-6, rtol=2e-6)
    assert set(head.state_dict()) == {'weight', 'bias'}


def test_statistics_and_class_weights_use_only_valid_training_labels():
    from continuous_mmwave.radar_text.head_ablation import training_statistics
    x = np.array([[1, 10], [3, 10], [5, 10], [100, 999]], dtype=np.float32)
    y = np.array([0, 0, 1, -100])
    mean, std, weights = training_statistics(x, y, classes=3)
    np.testing.assert_array_equal(mean, [3, 10])
    assert std[1] == pytest.approx(1e-5)
    np.testing.assert_allclose(weights, [0.5, 1., 1.])


def test_training_prefix_identical_and_selects_validation_not_test(tmp_path):
    from continuous_mmwave.radar_text.head_ablation import arm_specs, fit_head
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    arrays = {s: (rng.normal(size=(50, 4)).astype('float32'), rng.integers(0, 3, 50))
              for s in ('train', 'validation')}
    reference = torch.nn.Linear(4, 3).state_dict()
    first, second = arm_specs('probe_replay')[:2]
    first.update(epochs=2, batch_size=16)
    second.update(epochs=4, batch_size=16)
    a = fit_head(arrays, first, reference, tmp_path/'a.pt', device='cpu', classes=3)
    b = fit_head(arrays, second, reference, tmp_path/'b.pt', device='cpu', classes=3)
    assert a['history'] == b['history'][:2]
    best = max(b['history'], key=lambda r: r['validation_macro_f1'])
    assert b['selected_epoch'] == best['epoch']
    assert a['trainable_parameters'] == 4*3+3


def test_original_head_gate_rejects_changed_prediction():
    from continuous_mmwave.radar_text.head_ablation import assert_reference
    logits = np.array([[1., 0.], [0., 2.]], dtype='float32')
    assert assert_reference(logits, logits.copy(), np.array([0, 1])) == 0
    with pytest.raises(AssertionError):
        assert_reference(logits[:, ::-1], logits, np.array([0, 1]))
