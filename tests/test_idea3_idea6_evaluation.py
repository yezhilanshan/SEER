"""Regression cases for the shared partial-annotation evaluator."""
import importlib.util
from pathlib import Path

import numpy as np


spec = importlib.util.spec_from_file_location(
    'idea_comparison', Path(__file__).parents[1] / 'scripts/evaluate_idea3_idea6.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_unmatched_mixed_event_is_false_positive():
    known = np.zeros(40, bool)
    known[10:20] = True
    truth = [dict(start=10, stop=20, label=0)]
    events = [dict(start_frame=0, stop_frame=40, class_id=0)]
    result = evaluation.conservative_events(events, truth, known)
    assert (result['tp'], result['fp'], result['fn'], result['mixed_fp']) == (0, 1, 1, 1)


def test_unknown_prediction_is_separate_and_same_class_instances_stay_separate():
    known = np.zeros(40, bool)
    known[10:30] = True
    truth = [dict(start=10, stop=20, label=0), dict(start=20, stop=30, label=0)]
    events = [dict(start_frame=0, stop_frame=5, class_id=0),
              dict(start_frame=10, stop_frame=30, class_id=0)]
    result = evaluation.conservative_events(events, truth, known)
    assert (result['tp'], result['fp'], result['fn'], result['unknown']) == (1, 0, 1, 1)


def test_wrong_class_can_match_only_localization_metric():
    truth = [dict(start=0, stop=10, label=0)]
    events = [dict(start_frame=0, stop_frame=10, class_id=1)]
    known = np.ones(10, bool)
    assert evaluation.conservative_events(events, truth, known)['f1'] == 0
    assert evaluation.conservative_events(events, truth, known, require_class=False)['f1'] == 1
