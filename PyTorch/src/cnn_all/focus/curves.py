from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from cnn_all.config import AppConfig
from cnn_all.data.datasets import build_loader
from cnn_all.data.records import ImageRecord
from cnn_all.focus.fqpath import fqpath_score
from cnn_all.focus.unsharp import unsharp_mask


@dataclass
class FocusCurves:
    scores: np.ndarray  # N x (1 + len(radii)); column 0 is original
    radii: np.ndarray
    image_size: tuple[int, int]
    paths: list[str]
    amount: float
    edge_threshold: float
    lab_luminance_only: bool
    cache_version: int = 2

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            scores=self.scores.astype(np.float32),
            radii=self.radii.astype(np.float32),
            image_size=np.asarray(self.image_size, dtype=np.int64),
            paths=np.asarray(self.paths, dtype=str),
            amount=np.asarray(self.amount, dtype=np.float64),
            edge_threshold=np.asarray(self.edge_threshold, dtype=np.float64),
            lab_luminance_only=np.asarray(self.lab_luminance_only, dtype=np.bool_),
            cache_version=np.asarray(self.cache_version, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "FocusCurves":
        data = np.load(path, allow_pickle=False)
        return cls(
            scores=np.asarray(data["scores"], dtype=np.float32),
            radii=np.asarray(data["radii"], dtype=np.float32),
            image_size=tuple(int(v) for v in data["image_size"].tolist()),
            paths=[str(v) for v in data["paths"].tolist()],
            amount=float(data["amount"]) if "amount" in data else float("nan"),
            edge_threshold=(
                float(data["edge_threshold"])
                if "edge_threshold" in data
                else float("nan")
            ),
            lab_luminance_only=(
                bool(data["lab_luminance_only"])
                if "lab_luminance_only" in data
                else False
            ),
            cache_version=int(data["cache_version"]) if "cache_version" in data else 1,
        )

    def validate(
        self,
        records: Sequence[ImageRecord],
        image_size: Sequence[int],
        cfg: AppConfig,
    ) -> None:
        expected_paths = [str(Path(r.path).resolve()) for r in records]
        if self.paths != expected_paths:
            raise ValueError("Focus cache does not match the current manifest paths")
        if tuple(int(v) for v in image_size) != self.image_size:
            raise ValueError("Focus cache image size does not match the requested size")
        expected_radii = np.asarray(cfg.focus.radii, dtype=np.float32)
        if self.scores.shape != (len(records), len(expected_radii) + 1):
            raise ValueError("Focus cache has an unexpected score matrix shape")
        if not np.array_equal(self.radii.astype(np.float32), expected_radii):
            raise ValueError("Focus cache radii do not match the configuration")
        if not np.isclose(self.amount, cfg.focus.amount):
            raise ValueError("Focus cache amount does not match the configuration")
        if not np.isclose(self.edge_threshold, cfg.focus.edge_threshold):
            raise ValueError("Focus cache edge threshold does not match the configuration")
        if self.lab_luminance_only != cfg.focus.matlab_rgb_lab_sharpen:
            raise ValueError("Focus cache RGB sharpening mode does not match the configuration")
        if self.cache_version != 2:
            raise ValueError("Focus cache version is obsolete")


def focus_cache_path(cfg: AppConfig, image_size: Sequence[int]) -> Path:
    h, w = int(image_size[0]), int(image_size[1])
    return Path(cfg.focus.cache_dir) / f"focus_curves_{h}x{w}.npz"


def compute_focus_curves(
    records: Sequence[ImageRecord],
    cfg: AppConfig,
    device: torch.device,
    image_size: Sequence[int],
    logger=None,
) -> FocusCurves:
    cache = focus_cache_path(cfg, image_size)
    if cache.is_file() and not cfg.focus.force_recompute_cache:
        try:
            curves = FocusCurves.load(cache)
            curves.validate(records, image_size, cfg)
        except (ValueError, KeyError, OSError) as exc:
            if logger:
                logger.warning("Ignoring stale focus cache %s: %s", cache, exc)
        else:
            if logger:
                logger.info("Loaded focus-curve cache: %s", cache)
            return curves

    loader = build_loader(
        records,
        subset=None,
        batch_size=cfg.focus.batch_size,
        runtime=cfg.runtime,
        shuffle=False,
    )
    scores = np.zeros((len(records), len(cfg.focus.radii) + 1), dtype=np.float32)
    h, w = int(image_size[0]), int(image_size[1])

    iterator = tqdm(loader, desc=f"FQPath curves {h}x{w}", unit="batch")
    with torch.inference_mode():
        for batch in iterator:
            indices = batch["index"].numpy()
            images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
            images = F.interpolate(images, size=(h, w), mode="bilinear", align_corners=False, antialias=True)
            scores[indices, 0] = fqpath_score(images).detach().cpu().numpy()
            for column, radius in enumerate(cfg.focus.radii, start=1):
                sharpened = unsharp_mask(
                    images,
                    radius=float(radius),
                    amount=cfg.focus.amount,
                    threshold=cfg.focus.edge_threshold,
                    lab_luminance_only=cfg.focus.matlab_rgb_lab_sharpen,
                )
                scores[indices, column] = fqpath_score(sharpened).detach().cpu().numpy()

    curves = FocusCurves(
        scores=scores,
        radii=np.asarray(cfg.focus.radii, dtype=np.float32),
        image_size=(h, w),
        paths=[str(Path(r.path).resolve()) for r in records],
        amount=float(cfg.focus.amount),
        edge_threshold=float(cfg.focus.edge_threshold),
        lab_luminance_only=bool(cfg.focus.matlab_rgb_lab_sharpen),
        cache_version=2,
    )
    curves.save(cache)
    if logger:
        logger.info("Saved focus-curve cache: %s", cache)
    return curves


def select_variant_indices(scores: np.ndarray, threshold: float) -> np.ndarray:
    """Select original/radius columns using the original sequential rule."""
    if scores.ndim != 2 or scores.shape[1] < 1:
        raise ValueError("scores must be N x (1 + radii)")
    selected = np.zeros(scores.shape[0], dtype=np.int64)
    active = scores[:, 0] > float(threshold)
    for column in range(1, scores.shape[1]):
        selected[active] = column
        active = active & (scores[:, column] > float(threshold))
    return selected


def processed_focus_scores(scores: np.ndarray, threshold: float) -> np.ndarray:
    selected = select_variant_indices(scores, threshold)
    return scores[np.arange(scores.shape[0]), selected]


def apply_adaptive_unsharp(
    images: torch.Tensor,
    global_indices: torch.Tensor,
    curves: FocusCurves,
    threshold: float,
    amount: float,
    edge_threshold: float,
    lab_luminance_only: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply each selected radius to the original image batch on GPU."""
    indices_np = global_indices.detach().cpu().numpy().astype(np.int64)
    selected_np = select_variant_indices(curves.scores[indices_np], threshold)
    selected = torch.as_tensor(selected_np, device=images.device, dtype=torch.long)
    output = images.clone()
    for column in torch.unique(selected).tolist():
        column = int(column)
        if column == 0:
            continue
        mask = selected == column
        radius = float(curves.radii[column - 1])
        # Crucially, every radius is applied to `images` (the original batch),
        # never to the result from a previous radius.
        output[mask] = unsharp_mask(
            images[mask],
            radius=radius,
            amount=amount,
            threshold=edge_threshold,
            lab_luminance_only=lab_luminance_only,
        )
    return output, selected
