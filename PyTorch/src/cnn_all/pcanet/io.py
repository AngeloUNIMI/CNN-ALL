from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm

from cnn_all.config import AppConfig
from cnn_all.data.datasets import build_loader
from cnn_all.data.records import ImageRecord
from cnn_all.focus.curves import FocusCurves, apply_adaptive_unsharp
from cnn_all.focus.fqpath import rgb_to_gray
from cnn_all.pcanet.model import TorchPCANet


def make_pcanet_batch_factory(
    records: Sequence[ImageRecord],
    subset: Sequence[int],
    curves: FocusCurves,
    threshold: float,
    cfg: AppConfig,
    device: torch.device,
) -> Callable[[], Iterable[torch.Tensor]]:
    def factory() -> Iterable[torch.Tensor]:
        loader = build_loader(
            records,
            subset=subset,
            batch_size=cfg.pcanet.fit_batch_size,
            runtime=cfg.runtime,
            shuffle=False,
        )
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
            indices = batch["index"].to(device)
            images, _ = apply_adaptive_unsharp(
                images,
                indices,
                curves,
                threshold,
                cfg.focus.amount,
                cfg.focus.edge_threshold,
                cfg.focus.matlab_rgb_lab_sharpen,
            )
            gray = rgb_to_gray(images)
            if gray.ndim == 3:
                gray = gray.unsqueeze(1)
            gray = F.interpolate(
                gray,
                size=tuple(cfg.pcanet.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            yield gray

    return factory


def extract_pcanet_features(
    model: TorchPCANet,
    records: Sequence[ImageRecord],
    subset: Sequence[int],
    curves: FocusCurves,
    threshold: float,
    cfg: AppConfig,
    device: torch.device,
    desc: str = "PCANet features",
) -> torch.Tensor:
    loader = build_loader(
        records,
        subset=subset,
        batch_size=cfg.pcanet.feature_batch_size,
        runtime=cfg.runtime,
        shuffle=False,
    )
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for batch in tqdm(loader, desc=desc, leave=False):
            images = batch["image"].to(device, non_blocking=True).float().div_(255.0)
            indices = batch["index"].to(device)
            images, _ = apply_adaptive_unsharp(
                images,
                indices,
                curves,
                threshold,
                cfg.focus.amount,
                cfg.focus.edge_threshold,
                cfg.focus.matlab_rgb_lab_sharpen,
            )
            gray = rgb_to_gray(images)
            if gray.ndim == 3:
                gray = gray.unsqueeze(1)
            gray = F.interpolate(
                gray,
                size=tuple(cfg.pcanet.image_size),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            outputs.append(model.transform(gray).detach().cpu().float())
    if not outputs:
        raise RuntimeError("No PCANet features were extracted")
    return torch.cat(outputs, dim=0)
