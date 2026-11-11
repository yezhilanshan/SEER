import numpy as np
import pytest
import torch


def test_pairing_rejects_missing_duplicate_or_wrong_reference():
    from continuous_mmwave.radar_text.sliding_baselines import pair_previous
    windows = [dict(sample_id='a', actions=[1, 2]), dict(sample_id='b', actions=[2])]
    rows = [dict(sample_id='b', reference_actions=[2]), dict(sample_id='a', reference_actions=[1, 2])]
    assert [r['sample_id'] for r in pair_previous(windows, rows)] == ['a', 'b']
    for bad in (rows[:1], [rows[0], rows[0]], [rows[0], dict(rows[1], reference_actions=[2, 1])]):
        with pytest.raises(ValueError): pair_previous(windows, bad)


def test_validation_selection_common_and_per_method_ignore_test():
    from continuous_mmwave.radar_text.sliding_baselines import choose_thresholds
    def candidate(median, score):
        return dict(median_seconds=median, minimum_seconds=0., metrics=dict(sequence_exact=score, order_lcs=.5, action_recall=.5))
    grid = {'B0': [candidate(0., .4), candidate(.3, .5)],
            'B1': [candidate(0., .8), candidate(.3, .5)],
            'B2': [candidate(0., .8), candidate(.3, .5)]}
    result = choose_thresholds(grid, split='validation')
    assert result['common']['median_seconds'] == 0
    assert result['individual']['B0']['median_seconds'] == .3
    with pytest.raises(ValueError): choose_thresholds(grid, split='test')
    for rows in grid.values():
        for r in rows:r['metrics']['sequence_exact']=.5
    assert choose_thresholds(grid, split='validation')['common']['median_seconds'] == 0


def test_saved_standardized_head_is_applied_without_refitting():
    from continuous_mmwave.radar_text.sliding_baselines import apply_saved_head
    x = torch.tensor([[1., 6.], [5., 10.]])
    saved = dict(head=dict(weight=torch.tensor([[1., -1.], [-1., 1.]]), bias=torch.zeros(2)),
                 mean=torch.tensor([1., 2.]), std=torch.tensor([2., 4.]))
    expected = torch.nn.functional.linear((x-saved['mean'])/saved['std'], **saved['head'])
    torch.testing.assert_close(apply_saved_head(x, saved), expected)


def test_smoothing_and_summary_do_not_force_reference_action_count():
    from continuous_mmwave.radar_text.sliding_baselines import score_windows, summarize, complete_window
    phrases = {0:'walks', 1:'sits', 2:'stands'}
    rows = [dict(sample_id='a', actions=[0], subject='s', sensor='r', full_window=True)]
    times = [np.arange(12)*.1]
    predictions = [np.repeat([0,1,2],4)]
    scores = score_windows(rows, predictions, times, phrases, dict(median_seconds=0., minimum_seconds=0.))
    assert scores[0]['predicted_actions']==[0,1,2]
    summary=summarize(scores)
    assert summary['count_excess_windows']==1 and summary['metrics']['sequence_exact']==0
    assert complete_window(11.97,.09) and not complete_window(8.19,.09)
