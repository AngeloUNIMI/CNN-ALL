from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml


@dataclass
class RuntimeConfig:
    device: str = "auto"
    seed: int = 1337
    deterministic: bool = False
    num_workers: int = 16
    pin_memory: bool = True
    persistent_workers: bool = True
    prefetch_factor: int = 2
    cudnn_benchmark: bool = True
    allow_tf32: bool = True
    log_level: str = "INFO"
    resume: bool = True


@dataclass
class DatasetConfig:
    type: str = "raabin_wbc"
    root: str = "./imgs/orig/RAABIN-WBC"
    prepared_root: str = "./cache/prepared/RAABIN-WBC"
    manifest: str = "./cache/prepared/RAABIN-WBC/manifest.csv"
    classes: list[str] = field(
        default_factory=lambda: [
            "Basophil",
            "Eosinophil",
            "Lymphocyte",
            "Monocyte",
            "Neutrophil",
        ]
    )
    extensions: list[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"]
    )
    roi_size: list[int] = field(default_factory=lambda: [256, 256])
    use_existing_crops: bool = True
    apply_color_normalization: bool = False
    overwrite_prepared: bool = False
    # ALL-IDB2-only options
    all_idb_image_subdir: str = "ALL-IDB/ALL_IDB2/img"
    legacy_segmentation: bool = False
    segmentation_axis_scale: float = 1.5


@dataclass
class FocusConfig:
    enabled: bool = True
    cache_dir: str = "./cache/focus"
    batch_size: int = 32
    initial_image_size: list[int] = field(default_factory=lambda: [128, 128])
    application_image_size: list[int] = field(default_factory=lambda: [256, 256])
    radii: list[int] = field(default_factory=lambda: [1, 3, 5, 7, 9, 11, 13, 15, 17])
    amount: float = 0.8
    edge_threshold: float = 0.0
    initial_threshold_start: float = 9.0
    initial_threshold_steps: int = 30
    initial_threshold_step: float = 0.1
    tune_half_range: float = 0.5
    tune_step: float = 0.1
    matlab_rgb_lab_sharpen: bool = True
    force_recompute_cache: bool = False


@dataclass
class PCANetConfig:
    enabled: bool = True
    image_size: list[int] = field(default_factory=lambda: [128, 128])
    num_stages: int = 1
    patch_sizes: list[int] = field(default_factory=lambda: [15])
    max_filters: list[int] = field(default_factory=lambda: [11])
    dynamic_filters: bool = True
    retained_variance: list[float] = field(default_factory=lambda: [0.92])
    hist_block_size: list[int] = field(default_factory=lambda: [23, 23])
    block_overlap_ratio: float = 0.0
    fit_batch_size: int = 4
    feature_batch_size: int = 16
    covariance_patch_chunk: int = 8192
    dtype: str = "float32"
    max_training_images: int = 100000
    save_best_filters: bool = False


@dataclass
class KNNConfig:
    neighbors: int = 1
    distance: str = "euclidean"
    query_chunk_size: int = 128
    gallery_chunk_size: int = 2048


@dataclass
class ExperimentConfig:
    output_dir: str = "./Results_PyTorch/RAABIN_WBC"
    num_iterations: int = 10
    kfold: int = 2
    run_shared_phase: bool = True
    run_pretrained_features: bool = True
    run_finetuning: bool = True
    run_gradcam: bool = True
    backbones: list[str] = field(
        default_factory=lambda: [
            "alexnet",
            "vgg16",
            "vgg19",
            "resnet18",
            "resnet50",
            "resnet101",
            "densenet201",
        ]
    )
    selected_iterations: list[int] | None = None
    selected_backbones: list[str] | None = None


@dataclass
class CNNConfig:
    pretrained: bool = True
    download_weights: bool = True
    input_size: int = 224
    batch_size: int = 20
    feature_batch_size: int = 64
    epochs: int = 100
    learning_rate: float = 1e-4
    head_lr_multiplier: float = 20.0
    momentum: float = 0.9
    weight_decay: float = 1e-4
    label_smoothing: float = 0.0
    amp: bool = True
    compile: bool = False
    channels_last: bool = True
    class_weighted_loss: bool = False
    balanced_sampler: bool = False
    horizontal_flip: bool = True
    vertical_flip: bool = True
    rotation_degrees: float = 180.0
    log_every_steps: int = 20
    checkpoint_every_epochs: int = 1
    preprocess_mode: str = "imagenet"
    gradcam_max_samples: int = 10
    gradcam_only_sharpened: bool = True


@dataclass
class ExportConfig:
    enabled: bool = True
    output_dir: str = "./imgs/unsharp/RAABIN-WBC"
    batch_size: int = 32
    overwrite: bool = True
    save_metadata_csv: bool = True


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    focus: FocusConfig = field(default_factory=FocusConfig)
    pcanet: PCANetConfig = field(default_factory=PCANetConfig)
    knn: KNNConfig = field(default_factory=KNNConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    cnn: CNNConfig = field(default_factory=CNNConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


T = TypeVar("T")


def _dataclass_from_dict(cls: type[T], data: dict[str, Any] | None) -> T:
    data = data or {}
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        value = data[f.name]
        hinted = hints.get(f.name)
        if isinstance(hinted, type) and is_dataclass(hinted) and isinstance(value, dict):
            kwargs[f.name] = _dataclass_from_dict(hinted, value)
        else:
            kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = _dataclass_from_dict(AppConfig, raw)
    validate_config(cfg)
    return cfg


def save_config(cfg: AppConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)


def validate_config(cfg: AppConfig) -> None:
    if cfg.dataset.type not in {"raabin_wbc", "all_idb2"}:
        raise ValueError("dataset.type must be 'raabin_wbc' or 'all_idb2'")
    if cfg.experiment.kfold < 2:
        raise ValueError("experiment.kfold must be >= 2")
    if cfg.pcanet.num_stages != len(cfg.pcanet.patch_sizes):
        raise ValueError("pcanet.num_stages must match len(pcanet.patch_sizes)")
    if cfg.pcanet.num_stages != len(cfg.pcanet.max_filters):
        raise ValueError("pcanet.num_stages must match len(pcanet.max_filters)")
    if cfg.pcanet.num_stages != len(cfg.pcanet.retained_variance):
        raise ValueError("pcanet.num_stages must match len(pcanet.retained_variance)")
    for p in cfg.pcanet.patch_sizes:
        if p <= 0 or p % 2 == 0:
            raise ValueError("Every PCANet patch size must be a positive odd integer")
    if any(r <= 0 for r in cfg.focus.radii):
        raise ValueError("focus.radii must contain positive radii")
    if sorted(cfg.focus.radii) != cfg.focus.radii:
        raise ValueError("focus.radii must be sorted in increasing order")
    if cfg.cnn.preprocess_mode not in {"imagenet", "legacy_per_image_center"}:
        raise ValueError("cnn.preprocess_mode must be 'imagenet' or 'legacy_per_image_center'")
