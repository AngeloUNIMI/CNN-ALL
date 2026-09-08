from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from tqdm import tqdm

from cnn_all.cnn.backbones import BackboneSpec, build_backbone, get_submodule
from cnn_all.cnn.preprocessing import prepare_cnn_batch
from cnn_all.config import AppConfig
from cnn_all.data.datasets import build_loader
from cnn_all.data.records import ImageRecord
from cnn_all.focus.curves import FocusCurves


def extract_features_from_model(
    model: nn.Module,
    spec: BackboneSpec,
    backbone_name: str,
    records: Sequence[ImageRecord],
    subset: Sequence[int],
    curves: FocusCurves,
    focus_threshold: float,
    view: str,
    cfg: AppConfig,
    device: torch.device,
    logger=None,
) -> tuple[torch.Tensor, list[int]]:
    model.eval()
    feature_module = get_submodule(model, spec.feature_layer)
    captured: list[torch.Tensor] = []

    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            output = output[0]
        captured.append(output.detach())

    handle = feature_module.register_forward_hook(hook)
    loader = build_loader(
        records,
        subset=subset,
        batch_size=cfg.cnn.feature_batch_size,
        runtime=cfg.runtime,
        shuffle=False,
    )
    features: list[torch.Tensor] = []
    ordered_indices: list[int] = []
    amp_enabled = cfg.cnn.amp and device.type == "cuda"

    try:
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"{backbone_name} {view} features", leave=False):
                images, _ = prepare_cnn_batch(
                    batch["image"],
                    batch["index"],
                    cfg,
                    curves,
                    focus_threshold,
                    view,
                    device,
                    train=False,
                )
                captured.clear()
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    _ = model(images)
                if len(captured) != 1:
                    raise RuntimeError(
                        f"Expected one feature hook output, received {len(captured)}"
                    )
                value = captured[0]
                value = value.flatten(start_dim=1)
                features.append(value.float().cpu())
                ordered_indices.extend(int(i) for i in batch["index"].tolist())
    finally:
        handle.remove()

    if not features:
        raise RuntimeError("No CNN features were extracted")
    result = torch.cat(features, dim=0)
    if logger:
        logger.info(
            "%s %s feature matrix: %d x %d",
            backbone_name,
            view,
            result.shape[0],
            result.shape[1],
        )
    return result, ordered_indices


def extract_backbone_features(
    backbone_name: str,
    records: Sequence[ImageRecord],
    subset: Sequence[int],
    curves: FocusCurves,
    focus_threshold: float,
    view: str,
    cfg: AppConfig,
    device: torch.device,
    logger=None,
) -> tuple[torch.Tensor, list[int]]:
    model, spec = build_backbone(
        backbone_name,
        pretrained=cfg.cnn.pretrained,
        allow_download=cfg.cnn.download_weights,
    )
    model = model.to(device)
    if cfg.cnn.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    try:
        return extract_features_from_model(
            model,
            spec,
            backbone_name,
            records,
            subset,
            curves,
            focus_threshold,
            view,
            cfg,
            device,
            logger,
        )
    finally:
        del model
