from __future__ import annotations

import numpy as np
import torch

from cnn_all.focus.curves import processed_focus_scores, select_variant_indices
from cnn_all.focus.fqpath import fqpath_score
from cnn_all.focus.tuning import candidate_focus_thresholds, label_dependence
from cnn_all.focus.unsharp import gaussian_blur_symmetric, unsharp_mask


def test_fqpath_is_batched_finite_and_responds_to_sharpening() -> None:
    torch.manual_seed(3)
    texture = torch.rand(2, 3, 96, 96)
    blurred = gaussian_blur_symmetric(texture, sigma=3.0)
    sharpened = unsharp_mask(blurred, radius=1.0, amount=0.8, lab_luminance_only=True)

    blurred_score = fqpath_score(blurred)
    sharpened_score = fqpath_score(sharpened)

    assert blurred_score.shape == (2,)
    assert torch.isfinite(blurred_score).all()
    assert torch.isfinite(sharpened_score).all()
    # FQPath uses larger values for more blur.
    assert torch.all(sharpened_score < blurred_score)


def test_adaptive_variant_selection_uses_original_for_each_radius() -> None:
    # col 0 is original; cols 1.. are independently sharpened radii.
    scores = np.asarray(
        [
            [8.0, 7.5, 7.0],   # already below threshold -> original
            [10.0, 8.0, 7.0],  # first radius reaches threshold
            [10.0, 9.5, 8.0],  # second radius reaches threshold
            [10.0, 9.5, 9.2],  # never reaches threshold -> last radius
        ],
        dtype=np.float32,
    )
    selected = select_variant_indices(scores, threshold=9.0)
    np.testing.assert_array_equal(selected, np.asarray([0, 1, 2, 2]))
    np.testing.assert_allclose(
        processed_focus_scores(scores, 9.0),
        np.asarray([8.0, 8.0, 8.0, 9.2]),
    )


def test_multiclass_label_dependence_is_invariant_to_label_numbering() -> None:
    values = [1.0, 1.1, 4.0, 4.1, 8.0, 8.1]
    labels_a = [0, 0, 1, 1, 2, 2]
    labels_b = [7, 7, 2, 2, 99, 99]
    assert np.isclose(label_dependence(values, labels_a), label_dependence(values, labels_b))


def test_candidate_thresholds_include_both_endpoints() -> None:
    assert candidate_focus_thresholds(7.3, 0.5, 0.1) == [
        6.8,
        6.9,
        7.0,
        7.1,
        7.2,
        7.3,
        7.4,
        7.5,
        7.6,
        7.7,
        7.8,
    ]
