from __future__ import annotations

from pathlib import Path

import torch

from cnn_all.config import AppConfig, save_config
from cnn_all.data.records import load_manifest, prepare_dataset
from cnn_all.focus.curves import compute_focus_curves
from cnn_all.pipeline.export import export_unsharpened_dataset
from cnn_all.pipeline.networks import run_network_phase
from cnn_all.pipeline.shared import load_shared_splits, run_shared_phase
from cnn_all.utils import configure_runtime, resolve_device, setup_logging


def initialize(cfg: AppConfig):
    output_dir = Path(cfg.experiment.output_dir)
    logger = setup_logging(output_dir, cfg.runtime.log_level)
    configure_runtime(
        seed=cfg.runtime.seed,
        deterministic=cfg.runtime.deterministic,
        cudnn_benchmark=cfg.runtime.cudnn_benchmark,
        allow_tf32=cfg.runtime.allow_tf32,
    )
    device = resolve_device(cfg.runtime.device)
    save_config(cfg, output_dir / "resolved_config.yaml")
    logger.info("Device: %s", device)
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        logger.info(
            "GPU: %s | %.1f GiB | CUDA %s | PyTorch %s",
            props.name,
            props.total_memory / (1024**3),
            torch.version.cuda,
            torch.__version__,
        )
    else:
        logger.info("PyTorch %s | CPU threads %d", torch.__version__, torch.get_num_threads())
    return logger, device


def get_records(cfg: AppConfig, logger):
    return prepare_dataset(cfg.dataset, logger=logger)


def get_focus_curves(records, cfg: AppConfig, device, logger):
    initial = compute_focus_curves(
        records,
        cfg,
        device,
        cfg.focus.initial_image_size,
        logger=logger,
    )
    if tuple(cfg.focus.initial_image_size) == tuple(cfg.focus.application_image_size):
        application = initial
    else:
        application = compute_focus_curves(
            records,
            cfg,
            device,
            cfg.focus.application_image_size,
            logger=logger,
        )
    return initial, application


def run_prepare(cfg: AppConfig) -> None:
    logger, _ = initialize(cfg)
    records = get_records(cfg, logger)
    logger.info("Preparation complete: %d records", len(records))


def run_focus_cache(cfg: AppConfig) -> None:
    logger, device = initialize(cfg)
    records = get_records(cfg, logger)
    get_focus_curves(records, cfg, device, logger)
    logger.info("Focus caches complete")


def run_shared(cfg: AppConfig) -> None:
    logger, device = initialize(cfg)
    records = get_records(cfg, logger)
    initial, application = get_focus_curves(records, cfg, device, logger)
    run_shared_phase(records, initial, application, cfg, device, logger)


def run_networks(cfg: AppConfig) -> None:
    logger, device = initialize(cfg)
    records = get_records(cfg, logger)
    _, application = get_focus_curves(records, cfg, device, logger)
    splits = load_shared_splits(cfg.experiment.output_dir)
    run_network_phase(records, application, splits, cfg, device, logger)


def run_export(cfg: AppConfig) -> None:
    logger, device = initialize(cfg)
    records = get_records(cfg, logger)
    _, application = get_focus_curves(records, cfg, device, logger)
    splits = load_shared_splits(cfg.experiment.output_dir)
    export_unsharpened_dataset(records, application, splits, cfg, device, logger)


def run_all(cfg: AppConfig) -> None:
    logger, device = initialize(cfg)
    records = get_records(cfg, logger)
    initial, application = get_focus_curves(records, cfg, device, logger)
    if cfg.experiment.run_shared_phase:
        splits = run_shared_phase(records, initial, application, cfg, device, logger)
    else:
        splits = load_shared_splits(cfg.experiment.output_dir)
    if cfg.experiment.run_pretrained_features or cfg.experiment.run_finetuning:
        run_network_phase(records, application, splits, cfg, device, logger)
    if cfg.export.enabled:
        export_unsharpened_dataset(records, application, splits, cfg, device, logger)
    logger.info("Entire pipeline complete")
