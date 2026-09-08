from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

import torch
import torchvision

from cnn_all.cnn.backbones import BACKBONES, build_backbone
from cnn_all.config import AppConfig, load_config
from cnn_all.pipeline.run import (
    run_all,
    run_export,
    run_focus_cache,
    run_networks,
    run_prepare,
    run_shared,
)
from cnn_all.smoke import make_smoke_dataset


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--device", default=None, help="Override runtime.device, e.g. cuda, cuda:1, cpu")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=None,
        help="Run only these backbones (network phase)",
    )
    parser.add_argument(
        "--iterations",
        nargs="+",
        type=int,
        default=None,
        help="Run only these 1-based iteration numbers",
    )
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse completed outputs")


def _load_with_overrides(args) -> AppConfig:
    cfg = load_config(args.config)
    if args.device is not None:
        cfg.runtime.device = args.device
    if args.backbones:
        cfg.experiment.selected_backbones = list(args.backbones)
    if args.iterations:
        converted = [i - 1 for i in args.iterations]
        if any(i < 0 for i in converted):
            raise ValueError("--iterations uses positive, 1-based values")
        cfg.experiment.selected_iterations = converted
    if args.no_resume:
        cfg.runtime.resume = False
    return cfg


def doctor() -> int:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        info["gpus"] = [
            {
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_gib": round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 2),
            }
            for i in range(torch.cuda.device_count())
        ]
    print(json.dumps(info, indent=2))
    failures = []
    for name in BACKBONES:
        try:
            model, _ = build_backbone(name, pretrained=False)
            del model
            print(f"[OK] {name}")
        except Exception as exc:
            failures.append((name, str(exc)))
            print(f"[FAIL] {name}: {exc}")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cnn-all",
        description="GPU-optimized PyTorch CNN-ALL / RAABIN-WBC pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("run", "Run preparation, shared focus/PCANet phase, CNN phase, and export"),
        ("prepare", "Prepare/crop/resize the dataset and write the manifest"),
        ("focus-cache", "Precompute FQPath scores for original and all radii"),
        ("shared", "Run split generation and PCANet focus-threshold tuning"),
        ("networks", "Run pretrained features, fine-tuning, and Grad-CAM"),
        ("export", "Export the final adaptively unsharpened database"),
    ]:
        p = sub.add_parser(name, help=help_text)
        _add_common(p)
    sub.add_parser("doctor", help="Print environment/GPU details and construct all backbones")
    smoke = sub.add_parser("make-smoke-data", help="Generate a tiny five-class synthetic dataset")
    smoke.add_argument("--output", default="./smoke_data")
    smoke.add_argument("--images-per-class", type=int, default=4)
    smoke.add_argument("--size", type=int, default=64)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "make-smoke-data":
        path = make_smoke_dataset(args.output, args.images_per_class, args.size)
        print(path.resolve())
        return 0

    cfg = _load_with_overrides(args)
    actions = {
        "run": run_all,
        "prepare": run_prepare,
        "focus-cache": run_focus_cache,
        "shared": run_shared,
        "networks": run_networks,
        "export": run_export,
    }
    actions[args.command](cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
