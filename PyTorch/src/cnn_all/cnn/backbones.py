from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from urllib.parse import urlparse

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    constructor: Callable
    weights_enum: type
    feature_layer: str
    gradcam_layer: str
    head_path: str


BACKBONES: dict[str, BackboneSpec] = {
    "alexnet": BackboneSpec(
        "alexnet", models.alexnet, models.AlexNet_Weights, "classifier.1", "features.10", "classifier.6"
    ),
    "vgg16": BackboneSpec(
        "vgg16", models.vgg16, models.VGG16_Weights, "classifier.0", "features.28", "classifier.6"
    ),
    "vgg19": BackboneSpec(
        "vgg19", models.vgg19, models.VGG19_Weights, "classifier.0", "features.34", "classifier.6"
    ),
    "resnet18": BackboneSpec(
        "resnet18", models.resnet18, models.ResNet18_Weights, "fc", "layer4.1", "fc"
    ),
    "resnet50": BackboneSpec(
        "resnet50", models.resnet50, models.ResNet50_Weights, "fc", "layer4.2.conv3", "fc"
    ),
    "resnet101": BackboneSpec(
        "resnet101", models.resnet101, models.ResNet101_Weights, "fc", "layer4.2.conv3", "fc"
    ),
    "densenet201": BackboneSpec(
        "densenet201",
        models.densenet201,
        models.DenseNet201_Weights,
        "classifier",
        "features.denseblock4.denselayer32.conv2",
        "classifier",
    ),
}


def normalize_backbone_name(name: str) -> str:
    key = name.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "alexnet": "alexnet",
        "vgg16": "vgg16",
        "vgg19": "vgg19",
        "resnet18": "resnet18",
        "resnet50": "resnet50",
        "resnet101": "resnet101",
        "densenet201": "densenet201",
    }
    if key not in aliases:
        raise KeyError(f"Unsupported backbone: {name}. Available: {sorted(BACKBONES)}")
    return aliases[key]


def get_submodule(module: nn.Module, path: str) -> nn.Module:
    current: nn.Module = module
    for part in path.split("."):
        if part.isdigit():
            current = current[int(part)]  # type: ignore[index]
        else:
            current = getattr(current, part)
    return current


def set_submodule(module: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)  # type: ignore[index]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = value  # type: ignore[index]
    else:
        setattr(parent, last, value)


def _cached_weight_path(weights) -> Path:
    filename = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def build_backbone(
    name: str,
    pretrained: bool = True,
    allow_download: bool = True,
) -> tuple[nn.Module, BackboneSpec]:
    key = normalize_backbone_name(name)
    spec = BACKBONES[key]
    weights = spec.weights_enum.DEFAULT if pretrained else None
    if pretrained and not allow_download and not _cached_weight_path(weights).is_file():
        raise FileNotFoundError(
            f"Pretrained weights for {key} are not present in the local Torch Hub cache: "
            f"{_cached_weight_path(weights)}. Set cnn.download_weights=true for the first "
            "run or pre-populate TORCH_HOME/checkpoints."
        )
    try:
        model = spec.constructor(weights=weights)
    except Exception as exc:
        if pretrained:
            raise RuntimeError(
                f"Could not load pretrained weights for {key}. Ensure the server has "
                "internet access for the first run or pre-populate TORCH_HOME."
            ) from exc
        raise
    return model, spec


def replace_classifier(model: nn.Module, spec: BackboneSpec, num_classes: int) -> nn.Module:
    old = get_submodule(model, spec.head_path)
    if not isinstance(old, nn.Linear):
        raise TypeError(f"Expected a Linear classifier at {spec.head_path}, got {type(old).__name__}")
    new = nn.Linear(old.in_features, num_classes, bias=old.bias is not None)
    nn.init.normal_(new.weight, mean=0.0, std=0.01)
    if new.bias is not None:
        nn.init.zeros_(new.bias)
    set_submodule(model, spec.head_path, new)
    return new


def model_parameter_groups(
    model: nn.Module,
    head: nn.Module,
    base_lr: float,
    head_multiplier: float,
) -> list[dict]:
    head_ids = {id(p) for p in head.parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids and p.requires_grad]
    head_params = [p for p in head.parameters() if p.requires_grad]
    return [
        {"params": backbone_params, "lr": base_lr},
        {"params": head_params, "lr": base_lr * head_multiplier},
    ]
