from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Sequence

import torch

from cnn_all.classifiers.cmc import cmc_train_test
from cnn_all.classifiers.knn import knn_train_test
from cnn_all.classifiers.metrics import classification_metrics
from cnn_all.cnn.backbones import build_backbone, normalize_backbone_name
from cnn_all.cnn.features import extract_features_from_model
from cnn_all.cnn.gradcam import save_gradcam_comparisons
from cnn_all.cnn.training import train_finetuned_model
from cnn_all.config import AppConfig
from cnn_all.data.records import ImageRecord
from cnn_all.data.splits import SplitInfo
from cnn_all.focus.curves import FocusCurves
from cnn_all.pipeline.reporting import save_average_cmc
from cnn_all.utils import atomic_json_dump, empty_cuda_cache


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pretrained_feature_experiment(
    model,
    spec,
    backbone: str,
    split: SplitInfo,
    records: Sequence[ImageRecord],
    curves: FocusCurves,
    cfg: AppConfig,
    device: torch.device,
    output_dir: Path,
    logger=None,
    offload_after: bool = False,
) -> dict:
    result_path = output_dir / "pretrained_features.json"
    if cfg.runtime.resume and result_path.is_file():
        if logger:
            logger.info("Using existing pretrained-feature result: %s", result_path)
        return _load_json(result_path)

    model = model.to(device)
    if cfg.cnn.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.eval()

    threshold = float(split.best_focus_threshold)
    result: dict[str, dict] = {}
    for view in ("original", "unsharp"):
        train_features, train_order = extract_features_from_model(
            model,
            spec,
            backbone,
            records,
            split.train_indices,
            curves,
            threshold,
            view,
            cfg,
            device,
            logger,
        )
        test_features, test_order = extract_features_from_model(
            model,
            spec,
            backbone,
            records,
            split.test_indices,
            curves,
            threshold,
            view,
            cfg,
            device,
            logger,
        )
        train_labels = [records[i].label for i in train_order]
        test_labels = [records[i].label for i in test_order]
        knn = knn_train_test(
            train_features,
            test_features,
            train_labels,
            device,
            cfg.knn,
        )
        metrics = classification_metrics(test_labels, knn.predictions, cfg.dataset.classes)
        cmc = cmc_train_test(knn.distances, train_labels, test_labels)
        result[view] = {
            "metrics": metrics,
            "cmc": cmc,
            "train_indices": train_order,
            "test_indices": test_order,
            "predictions": knn.predictions.tolist(),
            "feature_dimension": int(train_features.shape[1]),
        }
        # Feature caches are useful for downstream analysis and cost much less
        # than re-running VGG/ResNet inference.
        torch.save(
            {
                "train_features": train_features,
                "test_features": test_features,
                "train_indices": train_order,
                "test_indices": test_order,
                "train_labels": train_labels,
                "test_labels": test_labels,
            },
            output_dir / f"features_{view}.pt",
        )
        if logger:
            logger.info(
                "%s iteration %d pretrained %s: accuracy %.4f, rank-5 %.4f",
                backbone,
                split.iteration + 1,
                view,
                metrics["accuracy"],
                cmc["rank5"],
            )
        del train_features, test_features, knn
    atomic_json_dump(result, result_path)
    if offload_after and device.type == "cuda":
        model.to("cpu")
        empty_cuda_cache()
    return result


def _aggregate(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": float("nan"), "std": float("nan")}
    return {
        "count": len(values),
        "mean": float(mean(values)),
        "std": float(pstdev(values)) if len(values) > 1 else 0.0,
    }


def _aggregate_network_results(rows: list[dict]) -> dict:
    summary: dict[str, dict] = {}
    for family in ("pretrained", "finetuned"):
        summary[family] = {}
        for view in ("original", "unsharp"):
            metrics_rows = [r[family][view] for r in rows if family in r and view in r[family]]
            summary[family][view] = {
                "accuracy": _aggregate([m["accuracy"] for m in metrics_rows]),
                "macro_f1": _aggregate([m["macro_f1"] for m in metrics_rows]),
                "balanced_accuracy": _aggregate([m["balanced_accuracy"] for m in metrics_rows]),
                "macro_specificity": _aggregate([m["macro_specificity"] for m in metrics_rows]),
            }
    if rows and all("pretrained_rank5" in r for r in rows):
        summary["pretrained_rank5"] = {
            view: _aggregate([r["pretrained_rank5"][view] for r in rows])
            for view in ("original", "unsharp")
        }
    return summary


def run_network_phase(
    records: Sequence[ImageRecord],
    curves: FocusCurves,
    splits: Sequence[SplitInfo],
    cfg: AppConfig,
    device: torch.device,
    logger=None,
) -> None:
    backbones = cfg.experiment.selected_backbones or cfg.experiment.backbones
    backbones = [normalize_backbone_name(b) for b in backbones]
    selected_iterations = (
        set(int(i) for i in cfg.experiment.selected_iterations)
        if cfg.experiment.selected_iterations is not None
        else None
    )

    for backbone_position, backbone in enumerate(backbones):
        network_root = Path(cfg.experiment.output_dir) / backbone
        network_root.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        feature_model = None
        feature_spec = None

        if cfg.experiment.run_pretrained_features:
            feature_model, feature_spec = build_backbone(
                backbone,
                pretrained=cfg.cnn.pretrained,
                allow_download=cfg.cnn.download_weights,
            )
            # Keep the feature-only model on CPU when fine-tuning is enabled.
            # It is moved to CUDA only for feature extraction, then offloaded, so
            # VGG/ResNet feature weights do not occupy VRAM during optimization.
            if not cfg.experiment.run_finetuning:
                feature_model = feature_model.to(device)
                if cfg.cnn.channels_last and device.type == "cuda":
                    feature_model = feature_model.to(memory_format=torch.channels_last)
            feature_model.eval()

        for split in splits:
            if selected_iterations is not None and split.iteration not in selected_iterations:
                continue
            if split.best_focus_threshold is None:
                raise ValueError(f"Split {split.iteration} has no best_focus_threshold")
            iteration_dir = network_root / f"iteration_{split.iteration + 1:02d}"
            iteration_dir.mkdir(parents=True, exist_ok=True)
            # Original and unsharp runs use identical initialization, sample
            # order, and augmentation RNG streams for a controlled comparison.
            paired_seed = (
                int(cfg.runtime.seed)
                + 100_000 * int(backbone_position)
                + 1_000 * int(split.iteration)
            )
            row: dict = {
                "iteration": split.iteration,
                "focus_threshold": float(split.best_focus_threshold),
            }

            if cfg.experiment.run_pretrained_features:
                assert feature_model is not None and feature_spec is not None
                pretrained = _pretrained_feature_experiment(
                    feature_model,
                    feature_spec,
                    backbone,
                    split,
                    records,
                    curves,
                    cfg,
                    device,
                    iteration_dir,
                    logger,
                    offload_after=cfg.experiment.run_finetuning,
                )
                row["pretrained"] = {
                    "original": pretrained["original"]["metrics"],
                    "unsharp": pretrained["unsharp"]["metrics"],
                }
                row["pretrained_rank5"] = {
                    "original": pretrained["original"]["cmc"]["rank5"],
                    "unsharp": pretrained["unsharp"]["cmc"]["rank5"],
                }
                row["pretrained_cmc"] = {
                    "original": pretrained["original"]["cmc"]["curve"],
                    "unsharp": pretrained["unsharp"]["cmc"]["curve"],
                }

            original_model = None
            unsharp_model = None
            original_spec = None
            unsharp_spec = None
            if cfg.experiment.run_finetuning:
                original_model, original_spec, original_metrics, _ = train_finetuned_model(
                    backbone,
                    records,
                    split.train_indices,
                    split.test_indices,
                    curves,
                    float(split.best_focus_threshold),
                    "original",
                    cfg,
                    device,
                    iteration_dir / "finetune_original",
                    logger,
                    run_seed=paired_seed,
                )

                # Never retain two large fine-tuned CNNs in VRAM. The original
                # model is offloaded before the unsharp model is created/trained.
                if cfg.experiment.run_gradcam:
                    original_model = original_model.to("cpu")
                else:
                    del original_model
                    original_model = None
                    original_spec = None
                empty_cuda_cache()

                unsharp_model, unsharp_spec, unsharp_metrics, _ = train_finetuned_model(
                    backbone,
                    records,
                    split.train_indices,
                    split.test_indices,
                    curves,
                    float(split.best_focus_threshold),
                    "unsharp",
                    cfg,
                    device,
                    iteration_dir / "finetune_unsharp",
                    logger,
                    run_seed=paired_seed,
                )
                row["finetuned"] = {
                    "original": original_metrics,
                    "unsharp": unsharp_metrics,
                }

                if cfg.experiment.run_gradcam:
                    unsharp_model = unsharp_model.to("cpu")
                else:
                    del unsharp_model
                    unsharp_model = None
                    unsharp_spec = None
                empty_cuda_cache()

            if (
                cfg.experiment.run_gradcam
                and original_model is not None
                and unsharp_model is not None
                and original_spec is not None
                and unsharp_spec is not None
            ):
                save_gradcam_comparisons(
                    original_model,
                    original_spec,
                    unsharp_model,
                    unsharp_spec,
                    records,
                    split.test_indices,
                    curves,
                    float(split.best_focus_threshold),
                    cfg,
                    device,
                    iteration_dir / "gradcam",
                    logger,
                )

            atomic_json_dump(row, iteration_dir / "summary.json")
            rows.append(row)
            del original_model, unsharp_model
            empty_cuda_cache()

        if feature_model is not None:
            del feature_model
            empty_cuda_cache()
        cmc_rows = [r["pretrained_cmc"] for r in rows if "pretrained_cmc" in r]
        if cmc_rows:
            save_average_cmc(
                [r["original"] for r in cmc_rows],
                [r["unsharp"] for r in cmc_rows],
                network_root,
            )

        atomic_json_dump(
            {
                "backbone": backbone,
                "iterations": rows,
                "aggregate": _aggregate_network_results(rows),
            },
            network_root / "summary.json",
        )
        if logger:
            logger.info("Completed network: %s", backbone)
