from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cnn_all.config import AppConfig
from cnn_all.focus.curves import FocusCurves, apply_adaptive_unsharp


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _random_flip_batch(x: torch.Tensor, horizontal: bool, vertical: bool) -> torch.Tensor:
    b = x.shape[0]
    if horizontal:
        mask = torch.rand(b, device=x.device) < 0.5
        x[mask] = torch.flip(x[mask], dims=[3])
    if vertical:
        mask = torch.rand(b, device=x.device) < 0.5
        x[mask] = torch.flip(x[mask], dims=[2])
    return x


def _random_rotate_batch(x: torch.Tensor, max_degrees: float) -> torch.Tensor:
    if max_degrees <= 0:
        return x
    angles = (torch.rand(x.shape[0], device=x.device, dtype=x.dtype) * 2.0 - 1.0) * max_degrees
    radians = angles * math.pi / 180.0
    cos = torch.cos(radians)
    sin = torch.sin(radians)
    theta = torch.zeros((x.shape[0], 2, 3), device=x.device, dtype=x.dtype)
    theta[:, 0, 0] = cos
    theta[:, 0, 1] = -sin
    theta[:, 1, 0] = sin
    theta[:, 1, 1] = cos
    grid = F.affine_grid(theta, x.size(), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def normalize_cnn_input(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "imagenet":
        mean = torch.tensor(IMAGENET_MEAN, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
        return (x - mean) / std
    if mode == "legacy_per_image_center":
        return x - x.mean(dim=(1, 2, 3), keepdim=True)
    raise ValueError(f"Unknown CNN preprocessing mode: {mode}")


def prepare_cnn_batch(
    raw_uint8: torch.Tensor,
    global_indices: torch.Tensor,
    cfg: AppConfig,
    curves: FocusCurves,
    focus_threshold: float,
    view: str,
    device: torch.device,
    train: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = raw_uint8.to(device, non_blocking=True).float().div_(255.0)
    indices = global_indices.to(device, non_blocking=True)
    if view == "unsharp":
        images, selected = apply_adaptive_unsharp(
            images,
            indices,
            curves,
            focus_threshold,
            cfg.focus.amount,
            cfg.focus.edge_threshold,
            cfg.focus.matlab_rgb_lab_sharpen,
        )
    elif view == "original":
        selected = torch.zeros(images.shape[0], device=device, dtype=torch.long)
    else:
        raise ValueError("view must be 'original' or 'unsharp'")

    if train:
        images = _random_flip_batch(
            images,
            horizontal=cfg.cnn.horizontal_flip,
            vertical=cfg.cnn.vertical_flip,
        )
        images = _random_rotate_batch(images, cfg.cnn.rotation_degrees)

    images = F.interpolate(
        images,
        size=(cfg.cnn.input_size, cfg.cnn.input_size),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    images = normalize_cnn_input(images, cfg.cnn.preprocess_mode)
    if cfg.cnn.channels_last and device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images, selected
