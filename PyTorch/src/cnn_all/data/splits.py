from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold


@dataclass
class SplitInfo:
    iteration: int
    seed: int
    train_indices: list[int]
    test_indices: list[int]
    train_labels: list[int]
    test_labels: list[int]
    initial_focus_threshold: float | None = None
    candidate_thresholds: list[float] | None = None
    candidate_accuracies: list[float] | None = None
    best_focus_threshold: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SplitInfo":
        return cls(**data)


def repeated_stratified_splits(
    labels: Sequence[int],
    num_iterations: int,
    kfold: int,
    seed: int,
) -> list[SplitInfo]:
    labels_arr = np.asarray(labels, dtype=np.int64)
    all_indices = np.arange(len(labels_arr))
    result: list[SplitInfo] = []

    for iteration in range(num_iterations):
        iteration_seed = seed + iteration
        skf = StratifiedKFold(n_splits=kfold, shuffle=True, random_state=iteration_seed)
        folds = list(skf.split(all_indices, labels_arr))
        # The MATLAB code selected one fold as training and the complement as
        # testing. For k=2 both sides are equal; for k>2 we preserve that
        # unusual orientation rather than silently changing the protocol.
        fold_index = iteration_seed % kfold
        conventional_train, conventional_test = folds[fold_index]
        train_indices = conventional_test
        test_indices = conventional_train
        result.append(
            SplitInfo(
                iteration=iteration,
                seed=iteration_seed,
                train_indices=train_indices.astype(int).tolist(),
                test_indices=test_indices.astype(int).tolist(),
                train_labels=labels_arr[train_indices].astype(int).tolist(),
                test_labels=labels_arr[test_indices].astype(int).tolist(),
            )
        )
    return result
