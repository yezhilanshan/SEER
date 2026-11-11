import pytest


def test_sequence_metrics_count_repeated_instances_instead_of_collapsing():
    from continuous_mmwave.event_gating.metrics import sequence_metrics
    a = sequence_metrics([1], [1, 1])
    assert a['sequence_exact'] == 0 and a['count_mae'] == 1 and a['edit_distance'] == 1
    assert sequence_metrics([1, 1], [1, 1])['sequence_exact'] == 1


def test_partial_annotations_keep_unknown_and_mixed_predictions_separate():
    from continuous_mmwave.event_gating.metrics import score_recording
    truth = [dict(start=2, stop=6, label=1)]
    pred = [dict(start_frame=0, stop_frame=2, class_id=0, closure='fire', confirmed_at_frame=2),
            dict(start_frame=2, stop_frame=4, class_id=1, closure='fire', confirmed_at_frame=5),
            dict(start_frame=4, stop_frame=6, class_id=0, closure='fire', confirmed_at_frame=6),
            dict(start_frame=6, stop_frame=10, class_id=2, closure='eof', confirmed_at_frame=10)]
    out = score_recording(pred, truth, 10)
    assert out['events']['tp'] == 1 and out['events']['fp'] == 1 and out['events']['fn'] == 0
    assert out['events']['unscored_unknown'] == 2
    assert out['events']['matches'][0]['iou'] == .5
    mixed = score_recording([dict(start_frame=0, stop_frame=10, class_id=0, closure='eof',
                                 confirmed_at_frame=10)], truth, 10)
    assert mixed['events']['unscored_mixed'] == 1 and mixed['events']['fn'] == 1
    assert out['end_boundary']['predicted'] == 3  # forced EOF excluded


def test_iou_matching_uses_original_interval_not_annotation_clipping():
    from continuous_mmwave.event_gating.metrics import score_recording
    pred = [dict(start_frame=0, stop_frame=100, class_id=1, closure='eof', confirmed_at_frame=100)]
    out = score_recording(pred, [dict(start=40, stop=45, label=1)], 100)
    assert out['events']['tp'] == 0 and out['events']['fn'] == 1


def test_adjacent_same_class_instance_recall_exposes_merged_events():
    from continuous_mmwave.event_gating.metrics import sequence_metrics
    merged=sequence_metrics([1,2],[1,1,2])
    assert merged['adjacent_same_class_instance_support']==2
    assert merged['adjacent_same_class_instance_recall']==.5
    assert sequence_metrics([1,1,2],[1,1,2])['adjacent_same_class_instance_recall']==1.
    assert sequence_metrics([1,2],[1,2])['adjacent_same_class_instance_recall'] is None
