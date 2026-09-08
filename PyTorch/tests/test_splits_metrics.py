from __future__ import annotations

import numpy as np

from cnn_all.classifiers.metrics import classification_metrics
from cnn_all.data.splits import repeated_stratified_splits


def test_repeated_splits_are_reproducible_and_stratified() -> None:
    labels = np.repeat(np.arange(5), 10)
    a = repeated_stratified_splits(labels, num_iterations=3, kfold=2, seed=123)
    b = repeated_stratified_splits(labels, num_iterations=3, kfold=2, seed=123)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]
    for split in a:
        train_counts = np.bincount(split.train_labels, minlength=5)
        test_counts = np.bincount(split.test_labels, minlength=5)
        np.testing.assert_array_equal(train_counts, np.full(5, 5))
        np.testing.assert_array_equal(test_counts, np.full(5, 5))
        assert set(split.train_indices).isdisjoint(split.test_indices)


def test_multiclass_metrics() -> None:
    result = classification_metrics(
        y_true=[0, 0, 1, 1, 2, 2],
        y_pred=[0, 1, 1, 1, 2, 0],
        class_names=["a", "b", "c"],
    )
    assert np.isclose(result["accuracy"], 4 / 6)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert np.isfinite(result["macro_f1"])
    assert result["num_misclassified"] == 2


def test_binary_metrics_preserve_legacy_fields_and_conventional_rates() -> None:
    result = classification_metrics(
        y_true=[0, 0, 0, 1, 1, 1],
        y_pred=[0, 1, 0, 1, 0, 1],
        class_names=["healthy", "all"],
    )
    assert result["TP"] == 2
    assert result["TN"] == 2
    assert result["FP"] == 1
    assert result["FN"] == 1
    assert np.isclose(result["sensitivity"], 2 / 3)
    assert np.isclose(result["specificity"], 2 / 3)
    # Historical CNN-ALL fields divide by the complete sample count.
    assert np.isclose(result["TPR"], 2 / 6)
    assert np.isclose(result["TNR"], 2 / 6)
