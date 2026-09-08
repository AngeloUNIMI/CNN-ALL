from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch

from cnn_all.classifiers.cmc import cmc_leave_one_out
from cnn_all.classifiers.knn import knn_leave_one_out
from cnn_all.classifiers.metrics import classification_metrics
from cnn_all.config import AppConfig
from cnn_all.data.records import ImageRecord
from cnn_all.data.splits import SplitInfo, repeated_stratified_splits
from cnn_all.focus.curves import FocusCurves
from cnn_all.focus.tuning import candidate_focus_thresholds, find_initial_focus_threshold
from cnn_all.pcanet.io import extract_pcanet_features, make_pcanet_batch_factory
from cnn_all.pcanet.model import TorchPCANet
from cnn_all.utils import atomic_json_dump, empty_cuda_cache


def _split_path(shared_dir: Path, iteration: int) -> Path:
    return shared_dir / f"split_{iteration + 1:02d}.json"


def load_shared_splits(output_dir: str | Path) -> list[SplitInfo]:
    shared_dir = Path(output_dir) / "shared"
    paths = sorted(shared_dir.glob("split_*.json"))
    if not paths:
        raise FileNotFoundError(f"No shared split files found in {shared_dir}")
    import json

    result = []
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            result.append(SplitInfo.from_dict(json.load(f)))
    return result


def run_shared_phase(
    records: Sequence[ImageRecord],
    initial_curves: FocusCurves,
    application_curves: FocusCurves,
    cfg: AppConfig,
    device: torch.device,
    logger=None,
) -> list[SplitInfo]:
    output_root = Path(cfg.experiment.output_dir)
    shared_dir = output_root / "shared"
    shared_dir.mkdir(parents=True, exist_ok=True)

    labels = [r.label for r in records]
    splits = repeated_stratified_splits(
        labels,
        num_iterations=cfg.experiment.num_iterations,
        kfold=cfg.experiment.kfold,
        seed=cfg.runtime.seed,
    )

    selected_iterations = (
        set(int(i) for i in cfg.experiment.selected_iterations)
        if cfg.experiment.selected_iterations is not None
        else None
    )
    completed: list[SplitInfo] = []

    for split in splits:
        if selected_iterations is not None and split.iteration not in selected_iterations:
            continue
        path = _split_path(shared_dir, split.iteration)
        if cfg.runtime.resume and path.is_file():
            import json

            with path.open("r", encoding="utf-8") as f:
                loaded = SplitInfo.from_dict(json.load(f))
            if loaded.best_focus_threshold is not None:
                if logger:
                    logger.info(
                        "Shared iteration %d already complete; best threshold %.3f",
                        split.iteration + 1,
                        loaded.best_focus_threshold,
                    )
                completed.append(loaded)
                continue

        train_idx = split.train_indices
        train_labels = split.train_labels
        if logger:
            logger.info(
                "Shared iteration %d/%d: %d train, %d test",
                split.iteration + 1,
                cfg.experiment.num_iterations,
                len(split.train_indices),
                len(split.test_indices),
            )

        initial_threshold, dependence_trace = find_initial_focus_threshold(
            initial_curves.scores[train_idx],
            train_labels,
            start=cfg.focus.initial_threshold_start,
            steps=cfg.focus.initial_threshold_steps,
            step=cfg.focus.initial_threshold_step,
        )
        candidates = candidate_focus_thresholds(
            initial_threshold,
            cfg.focus.tune_half_range,
            cfg.focus.tune_step,
        )
        if logger:
            logger.info(
                "Iteration %d initial focus threshold %.3f; candidates %s",
                split.iteration + 1,
                initial_threshold,
                candidates,
            )

        candidate_rows: list[dict] = []
        best_accuracy = float("-inf")
        best_threshold = candidates[0]
        best_model: TorchPCANet | None = None

        if not cfg.pcanet.enabled:
            best_threshold = initial_threshold
            candidate_rows.append(
                {
                    "threshold": best_threshold,
                    "accuracy": float("nan"),
                    "macro_f1": float("nan"),
                    "rank5": float("nan"),
                    "seconds": 0.0,
                }
            )
        else:
            for threshold in candidates:
                start_time = time.perf_counter()
                model = TorchPCANet(cfg.pcanet, device)
                factory = make_pcanet_batch_factory(
                    records,
                    train_idx,
                    application_curves,
                    threshold,
                    cfg,
                    device,
                )
                model.fit(factory, logger=logger)
                features = extract_pcanet_features(
                    model,
                    records,
                    train_idx,
                    application_curves,
                    threshold,
                    cfg,
                    device,
                    desc=f"PCANet threshold {threshold:.2f}",
                )
                knn = knn_leave_one_out(features, train_labels, device, cfg.knn)
                metrics = classification_metrics(train_labels, knn.predictions, cfg.dataset.classes)
                cmc = cmc_leave_one_out(knn.distances, train_labels)
                elapsed = time.perf_counter() - start_time
                row = {
                    "threshold": float(threshold),
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "balanced_accuracy": metrics["balanced_accuracy"],
                    "rank5": cmc["rank5"],
                    "cmc_auc_sum": cmc["auc_sum"],
                    "seconds": elapsed,
                    "num_filters": model.summary.num_filters if model.summary else [],
                    "retained_variance": model.summary.retained_variance if model.summary else [],
                }
                candidate_rows.append(row)
                if logger:
                    logger.info(
                        "Iteration %d threshold %.3f: PCANet LOO accuracy %.4f, macro-F1 %.4f, %.1f s",
                        split.iteration + 1,
                        threshold,
                        metrics["accuracy"],
                        metrics["macro_f1"],
                        elapsed,
                    )
                # Highest threshold wins ties, matching the original sorted-last behavior.
                if metrics["accuracy"] > best_accuracy or (
                    metrics["accuracy"] == best_accuracy and threshold > best_threshold
                ):
                    best_accuracy = metrics["accuracy"]
                    best_threshold = threshold
                    best_model = model
                del features, knn, model
                empty_cuda_cache()

        split.initial_focus_threshold = float(initial_threshold)
        split.candidate_thresholds = [float(r["threshold"]) for r in candidate_rows]
        split.candidate_accuracies = [float(r["accuracy"]) for r in candidate_rows]
        split.best_focus_threshold = float(best_threshold)
        atomic_json_dump(split.to_dict(), path)
        atomic_json_dump(dependence_trace, shared_dir / f"dependence_{split.iteration + 1:02d}.json")
        pd.DataFrame(candidate_rows).to_csv(
            shared_dir / f"focus_tuning_{split.iteration + 1:02d}.csv",
            index=False,
        )
        if cfg.pcanet.save_best_filters and best_model is not None:
            best_model.save(shared_dir / f"pcanet_best_{split.iteration + 1:02d}.pt")
        completed.append(split)
        if logger:
            logger.info(
                "Shared iteration %d complete: best threshold %.3f",
                split.iteration + 1,
                best_threshold,
            )

    atomic_json_dump(
        {
            "num_splits": len(completed),
            "best_focus_thresholds": [s.best_focus_threshold for s in completed],
            "splits": [s.to_dict() for s in completed],
        },
        shared_dir / "summary.json",
    )
    return completed
