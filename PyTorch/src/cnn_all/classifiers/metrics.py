from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def _safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    out = np.full_like(a, np.nan, dtype=np.float64)
    np.divide(a, b, out=out, where=b != 0)
    return out


def _mean_without_nan(values: np.ndarray) -> float:
    """MATLAB meanNoNaN semantics without RuntimeWarning on all-NaN input."""
    finite = values[~np.isnan(values)]
    return float(finite.mean()) if finite.size else float("nan")


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Compute the binary/multiclass metrics used by the MATLAB pipeline.

    For multiclass problems, sensitivity/recall, specificity, precision, and
    F1 are one-vs-rest macro averages.  For the binary case, the historical
    TPR/TNR/FPR/FNR fields retain the original CNN-ALL semantics (cell count
    divided by the total), while ``sensitivity`` and ``specificity`` expose
    the conventional conditional rates.
    """

    labels = np.arange(len(class_names), dtype=int)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    total = float(cm.sum())
    if total <= 0:
        raise ValueError("Cannot compute metrics from an empty confusion matrix")

    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    tn = total - tp - fn - fp
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    precision = _safe_divide(tp, tp + fp)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    macro_recall = _mean_without_nan(recall)
    macro_specificity = _mean_without_nan(specificity)
    macro_precision = _mean_without_nan(precision)
    macro_f1 = _mean_without_nan(f1)

    result = {
        "num_classes": len(class_names),
        "class_names": list(class_names),
        "confusion_matrix": cm.astype(int).tolist(),
        "num_misclassified": int(total - np.trace(cm)),
        "error_rate": float((total - np.trace(cm)) / total),
        "accuracy": float(np.trace(cm) / total),
        "macro_recall": macro_recall,
        "macro_sensitivity": macro_recall,
        "macro_specificity": macro_specificity,
        "macro_precision": macro_precision,
        "macro_f1": macro_f1,
        "balanced_accuracy": macro_recall,
        # MATLAB-compatible aggregate aliases.
        "sens": macro_recall,
        "spec": macro_specificity,
        "precision": macro_precision,
        "f1": macro_f1,
        "balancedAccuracy": macro_recall,
        "per_class": {},
    }
    for i, name in enumerate(class_names):
        result["per_class"][name] = {
            "tp": int(tp[i]),
            "tn": int(tn[i]),
            "fp": int(fp[i]),
            "fn": int(fn[i]),
            "recall": None if np.isnan(recall[i]) else float(recall[i]),
            "sensitivity": None if np.isnan(recall[i]) else float(recall[i]),
            "specificity": None if np.isnan(specificity[i]) else float(specificity[i]),
            "precision": None if np.isnan(precision[i]) else float(precision[i]),
            "f1": None if np.isnan(f1[i]) else float(f1[i]),
        }

    if len(class_names) == 2:
        positive = 1
        conventional_sensitivity = float(recall[positive])
        conventional_specificity = float(specificity[positive])
        conventional_precision = float(precision[positive])
        conventional_f1 = float(f1[positive])
        balanced = _mean_without_nan(
            np.asarray([conventional_sensitivity, conventional_specificity])
        )
        result.update(
            {
                "TP": int(tp[positive]),
                "TN": int(tn[positive]),
                "FP": int(fp[positive]),
                "FN": int(fn[positive]),
                "sensitivity": conventional_sensitivity,
                "specificity": conventional_specificity,
                "binary_precision": conventional_precision,
                "binary_f1": conventional_f1,
                "binary_balanced_accuracy": balanced,
                # Historical fields retained exactly as in computeErrorsFromCM.m.
                "TPR": float(tp[positive] / total),
                "TNR": float(tn[positive] / total),
                "FPR": float(fp[positive] / total),
                "FNR": float(fn[positive] / total),
            }
        )
    else:
        result.update(
            {
                "TPR": macro_recall,
                "TNR": macro_specificity,
                "FPR": float(1.0 - macro_specificity),
                "FNR": float(1.0 - macro_recall),
            }
        )
    return result
