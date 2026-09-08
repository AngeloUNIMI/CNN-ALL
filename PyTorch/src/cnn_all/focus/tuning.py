from __future__ import annotations

from typing import Sequence

import numpy as np

from cnn_all.focus.curves import processed_focus_scores


def label_dependence(values: Sequence[float], labels: Sequence[int]) -> float:
    values_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    labels_arr = np.asarray(labels).reshape(-1)
    classes = np.unique(labels_arr)
    if len(values_arr) != len(labels_arr):
        raise ValueError("values and labels must have the same length")
    if len(classes) <= 2:
        if len(values_arr) < 2 or np.std(values_arr) == 0 or np.std(labels_arr.astype(float)) == 0:
            return 0.0
        dep = abs(float(np.corrcoef(values_arr, labels_arr.astype(float))[0, 1]))
        return 0.0 if not np.isfinite(dep) else dep

    overall = float(values_arr.mean())
    total = float(np.square(values_arr - overall).sum())
    if total <= np.finfo(np.float64).eps:
        return 0.0
    between = 0.0
    for cls in classes:
        subset = values_arr[labels_arr == cls]
        if subset.size:
            between += subset.size * float(subset.mean() - overall) ** 2
    return float(between / total)


def find_initial_focus_threshold(
    curve_scores: np.ndarray,
    labels: Sequence[int],
    start: float = 9.0,
    steps: int = 30,
    step: float = 0.1,
) -> tuple[float, list[dict[str, float]]]:
    """Port the 9.0, 8.9, ..., 6.1 label-dependence search."""
    labels_arr = np.asarray(labels)
    best_threshold = 1000.0
    best_dep = 1000.0
    trace: list[dict[str, float]] = []
    for i in range(steps):
        threshold = round(start - i * step, 10)
        processed = processed_focus_scores(curve_scores, threshold)
        dep = label_dependence(processed, labels_arr)
        trace.append({"threshold": threshold, "dependence": dep})
        if abs(dep) < best_dep:
            best_dep = abs(dep)
            best_threshold = threshold
    return best_threshold, trace


def candidate_focus_thresholds(initial: float, half_range: float, step: float) -> list[float]:
    start = round(initial - half_range, 10)
    end = round(initial + half_range, 10)
    if half_range == 0:
        return [round(initial, 10)]
    count = int(round((end - start) / step)) + 1
    return [round(start + i * step, 10) for i in range(count)]
