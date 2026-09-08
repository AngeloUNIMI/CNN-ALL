from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from cnn_all.config import AppConfig
from cnn_all.data.datasets import build_loader
from cnn_all.data.records import ImageRecord
from cnn_all.data.splits import SplitInfo
from cnn_all.focus.curves import FocusCurves, apply_adaptive_unsharp, select_variant_indices
from cnn_all.utils import atomic_json_dump


def representative_threshold(splits: Sequence[SplitInfo]) -> float:
    values = np.asarray(
        [float(s.best_focus_threshold) for s in splits if s.best_focus_threshold is not None],
        dtype=np.float64,
    )
    if values.size == 0:
        raise ValueError("No tuned focus thresholds are available for export")
    median = float(np.median(values))
    return float(values[np.argmin(np.abs(values - median))])


def export_unsharpened_dataset(
    records: Sequence[ImageRecord],
    curves: FocusCurves,
    splits: Sequence[SplitInfo],
    cfg: AppConfig,
    device: torch.device,
    logger=None,
) -> dict:
    threshold = representative_threshold(splits)
    output_root = Path(cfg.export.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for class_name in cfg.dataset.classes:
        (output_root / class_name).mkdir(parents=True, exist_ok=True)

    loader = build_loader(
        records,
        subset=None,
        batch_size=cfg.export.batch_size,
        runtime=cfg.runtime,
        shuffle=False,
    )
    rows: list[dict] = []
    num_sharpened = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc="Exporting adaptive unsharp DB", unit="batch"):
            images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
            indices = batch["index"].to(device)
            output, selected = apply_adaptive_unsharp(
                images,
                indices,
                curves,
                threshold,
                cfg.focus.amount,
                cfg.focus.edge_threshold,
                cfg.focus.matlab_rgb_lab_sharpen,
            )
            output_u8 = (output.clamp(0, 1) * 255.0).round().to(torch.uint8).cpu()
            selected_np = selected.cpu().numpy().astype(int)
            global_indices = batch["index"].numpy().astype(int)

            for local, global_index in enumerate(global_indices):
                record = records[int(global_index)]
                variant = int(selected_np[local])
                radius = 0.0 if variant == 0 else float(curves.radii[variant - 1])
                num_sharpened += int(variant > 0)
                output_path = output_root / record.class_name / record.original_filename
                if cfg.export.overwrite or not output_path.exists():
                    array = output_u8[local].permute(1, 2, 0).numpy()
                    Image.fromarray(array, mode="RGB").save(output_path)
                rows.append(
                    {
                        "index": int(global_index),
                        "class_name": record.class_name,
                        "label": record.label,
                        "source_path": record.path,
                        "output_path": str(output_path.resolve()),
                        "original_filename": record.original_filename,
                        "threshold": threshold,
                        "sharpened": bool(variant > 0),
                        "variant_column": variant,
                        "radius": radius,
                        "iterations": variant,
                        "focus_before": float(curves.scores[global_index, 0]),
                        "focus_after": float(curves.scores[global_index, variant]),
                    }
                )

    summary = {
        "threshold": threshold,
        "num_images": len(records),
        "num_sharpened": num_sharpened,
        "fraction_sharpened": num_sharpened / max(len(records), 1),
        "output_dir": str(output_root.resolve()),
        "class_names": cfg.dataset.classes,
        "thresholds_by_iteration": [s.best_focus_threshold for s in splits],
    }
    if cfg.export.save_metadata_csv:
        pd.DataFrame(rows).to_csv(output_root / "export_info.csv", index=False)
    atomic_json_dump(summary, output_root / "export_info.json")
    (output_root / "README.txt").write_text(
        "CNN-ALL-PyTorch adaptively unsharpened database\n"
        f"Focus threshold: {threshold:.8f}\n"
        f"Images: {len(records)}\n"
        f"Images sharpened: {num_sharpened}\n"
        f"Fraction sharpened: {summary['fraction_sharpened']:.6f}\n"
        "Every selected radius was applied to the original image, not cumulatively.\n",
        encoding="utf-8",
    )
    if logger:
        logger.info(
            "Export complete: %d/%d images sharpened; threshold %.3f; %s",
            num_sharpened,
            len(records),
            threshold,
            output_root,
        )
    return summary
