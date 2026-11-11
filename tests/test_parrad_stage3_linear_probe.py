from __future__ import annotations

import numpy as np
import pytest


def test_relational_projection_matches_capacity_across_layer_dimensions() -> None:
    from continuous_mmwave.parrad_stage3_linear_probe import relational_features

    for dimension in (128, 1024, 7168):
        current = np.ones((3, dimension), np.float32)
        reference = np.zeros_like(current)
        first = relational_features(current, reference, 3407)
        repeated = relational_features(current, reference, 3407)
        second = relational_features(current, reference, 7211)
        assert first.shape == (3, 130)
        np.testing.assert_array_equal(first, repeated)
        assert not np.array_equal(first[:, :128], second[:, :128])
        assert first[0, -1] == pytest.approx(np.sqrt(dimension))


def test_probe_training_transform_and_coefficients_ignore_holdout_distribution() -> None:
    from continuous_mmwave.parrad_stage3_linear_probe import fit_probe

    train_x = np.array([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]])
    train_y = np.array([0., 0., 1., 1.])
    first = fit_probe(train_x, train_y, np.array([[-3., 1.], [3., 1.]]))
    second = fit_probe(train_x, train_y, np.array([[1000., 9.], [9000., 3.]]))
    np.testing.assert_array_equal(first["mean"], [0., 1.])
    np.testing.assert_array_equal(first["mean"], second["mean"])
    np.testing.assert_array_equal(first["coefficients"], second["coefficients"])
    assert first["scores"][0] < 0.5 < first["scores"][1]
    assert first["parameter_count"] == 3


def test_subject_folds_reject_nontrain_and_preserve_whole_subject() -> None:
    from continuous_mmwave.parrad_stage3_linear_probe import subject_fold_masks

    rows = [{"source_split": "train", "subject_id": subject} for subject in ("subject_02", "subject_03", "subject_04", "subject_02")]
    train, held = subject_fold_masks(rows, "subject_02")
    assert train.tolist() == [False, True, True, False]
    assert held.tolist() == [True, False, False, True]
    rows[1]["source_split"] = "validation"
    with pytest.raises(ValueError, match="train"):
        subject_fold_masks(rows, "subject_02")


def test_one_class_metrics_are_undefined_and_cutoff_is_fixed() -> None:
    from continuous_mmwave.parrad_stage3_linear_probe import binary_metrics

    result = binary_metrics([True, True], [0.8, 0.9])
    assert result["auroc"] is None
    assert result["balanced_accuracy"] is None
    result = binary_metrics([False, True], [0.49, 0.49])
    assert result["balanced_accuracy"] == 0.5
