from __future__ import annotations

from typing import Sequence

import numpy as np


def cmc_train_test(
    distances: np.ndarray,
    train_labels: Sequence[int],
    test_labels: Sequence[int],
    pad_to: int = 30,
) -> dict:
    distances = np.asarray(distances)
    train = np.asarray(train_labels)
    test = np.asarray(test_labels)
    if distances.shape != (len(test), len(train)):
        raise ValueError("Distance matrix shape does not match train/test labels")

    ranks = np.empty(len(test), dtype=np.int64)
    for i in range(len(test)):
        order = np.argsort(distances[i], kind="stable")
        matches = np.flatnonzero(train[order] == test[i])
        ranks[i] = int(matches[0] + 1) if matches.size else len(train) + 1

    max_rank = max(int(ranks.max(initial=1)), 1)
    probabilities = np.bincount(ranks, minlength=max_rank + 1)[1:].astype(np.float64) / max(len(test), 1)
    curve = np.cumsum(probabilities)
    if curve.size < pad_to:
        curve = np.pad(curve, (0, pad_to - curve.size), constant_values=1.0)
    rank5 = float(curve[min(4, curve.size - 1)])
    return {
        "ranks": ranks.tolist(),
        "curve": curve.tolist(),
        "auc_sum": float(curve.sum()),
        "rank1": float(curve[0]),
        "rank5": rank5,
    }


def cmc_leave_one_out(
    distances: np.ndarray,
    labels: Sequence[int],
    pad_to: int = 30,
) -> dict:
    distances = np.asarray(distances).copy()
    labels_arr = np.asarray(labels)
    if distances.shape != (len(labels_arr), len(labels_arr)):
        raise ValueError("Leave-one-out distance matrix must be square")
    np.fill_diagonal(distances, np.inf)
    return cmc_train_test(distances, labels_arr, labels_arr, pad_to=pad_to)
