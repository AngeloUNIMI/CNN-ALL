from __future__ import annotations

import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

from cnn_all.config import DatasetConfig
from cnn_all.legacy.preprocessing import LegacySegmentationConfig, prepare_legacy_roi


@dataclass(frozen=True)
class ImageRecord:
    index: int
    path: str
    label: int
    class_name: str
    original_filename: str
    source_path: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _iter_image_files(root: Path, extensions: Iterable[str]) -> list[Path]:
    valid = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    return sorted(
        [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in valid],
        key=lambda p: p.name.lower(),
    )


def scan_raabin(cfg: DatasetConfig) -> list[tuple[Path, int, str]]:
    root = Path(cfg.root)
    if not root.is_dir():
        raise FileNotFoundError(f"RAABIN-WBC root does not exist: {root}")
    rows: list[tuple[Path, int, str]] = []
    for label, class_name in enumerate(cfg.classes):
        class_dir = root / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"RAABIN-WBC class directory not found: {class_dir}. "
                "Linux paths are case-sensitive."
            )
        files = _iter_image_files(class_dir, cfg.extensions)
        if not files:
            raise RuntimeError(f"No supported images found in {class_dir}")
        rows.extend((path, label, class_name) for path in files)
    return rows


def _parse_all_idb_label(filename: str) -> int:
    match = re.search(r"_(\d+)(?:\.[^.]+)$", filename)
    if not match:
        raise ValueError(f"Cannot parse ALL-IDB2 label from filename: {filename}")
    return int(match.group(1))


def scan_all_idb2(cfg: DatasetConfig) -> list[tuple[Path, int, str]]:
    root = Path(cfg.root) / cfg.all_idb_image_subdir
    if not root.is_dir():
        raise FileNotFoundError(f"ALL-IDB2 image directory does not exist: {root}")
    rows: list[tuple[Path, int, str]] = []
    for path in _iter_image_files(root, cfg.extensions):
        label = _parse_all_idb_label(path.name)
        if label >= len(cfg.classes):
            raise ValueError(f"Label {label} in {path.name} has no corresponding class name")
        rows.append((path, label, cfg.classes[label]))
    if not rows:
        raise RuntimeError(f"No supported ALL-IDB2 images found in {root}")
    return rows


def scan_source_dataset(cfg: DatasetConfig) -> list[tuple[Path, int, str]]:
    if cfg.type == "raabin_wbc":
        return scan_raabin(cfg)
    if cfg.type == "all_idb2":
        return scan_all_idb2(cfg)
    raise ValueError(f"Unsupported dataset type: {cfg.type}")


def prepare_dataset(cfg: DatasetConfig, logger=None) -> list[ImageRecord]:
    prepared_root = Path(cfg.prepared_root)
    manifest_path = Path(cfg.manifest)

    if manifest_path.is_file() and not cfg.overwrite_prepared:
        records = load_manifest(manifest_path)
        missing = [r.path for r in records if not Path(r.path).is_file()]
        if not missing:
            if logger:
                logger.info("Using existing prepared dataset: %s", manifest_path)
            return records
        if logger:
            logger.warning("Prepared manifest has %d missing files; rebuilding", len(missing))

    source_rows = scan_source_dataset(cfg)
    if cfg.overwrite_prepared and prepared_root.exists():
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    records: list[ImageRecord] = []
    roi_h, roi_w = int(cfg.roi_size[0]), int(cfg.roi_size[1])
    legacy_cfg = LegacySegmentationConfig(
        axis_scale=cfg.segmentation_axis_scale,
        roi_size=(roi_h, roi_w),
    )

    iterator = tqdm(source_rows, desc="Preparing dataset", unit="image")
    for index, (source_path, label, class_name) in enumerate(iterator):
        class_out = prepared_root / class_name
        class_out.mkdir(parents=True, exist_ok=True)
        # Keep the original name inside class folders. This preserves external
        # references and avoids collisions because each class has its own dir.
        output_path = class_out / source_path.name

        try:
            with Image.open(source_path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                use_legacy_roi = (
                    not cfg.use_existing_crops
                    or (cfg.type == "all_idb2" and cfg.legacy_segmentation)
                )
                if use_legacy_roi:
                    prepared = prepare_legacy_roi(
                        im,
                        legacy_cfg,
                        apply_colour_normalization=cfg.apply_color_normalization,
                    )
                else:
                    prepared = im.resize((roi_w, roi_h), resample=Image.Resampling.BILINEAR)
                prepared.save(output_path)
        except Exception as exc:
            if logger:
                logger.warning("Skipping %s: %s", source_path, exc)
            continue

        records.append(
            ImageRecord(
                index=len(records),
                path=str(output_path.resolve()),
                label=int(label),
                class_name=class_name,
                original_filename=source_path.name,
                source_path=str(source_path.resolve()),
            )
        )

    if not records:
        raise RuntimeError("Dataset preparation produced no usable images")

    present_classes = {r.class_name for r in records}
    missing_classes = [name for name in cfg.classes if name not in present_classes]
    if missing_classes:
        raise RuntimeError(
            "Dataset preparation produced no usable images for classes: "
            + ", ".join(missing_classes)
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([r.as_dict() for r in records]).to_csv(manifest_path, index=False)
    if logger:
        logger.info("Prepared %d images at %s", len(records), prepared_root)
        logger.info("Manifest: %s", manifest_path)
    return records


def load_manifest(path: str | Path) -> list[ImageRecord]:
    frame = pd.read_csv(path)
    required = {
        "index",
        "path",
        "label",
        "class_name",
        "original_filename",
        "source_path",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")
    records: list[ImageRecord] = []
    for row in frame.to_dict(orient="records"):
        records.append(
            ImageRecord(
                index=int(row["index"]),
                path=str(row["path"]),
                label=int(row["label"]),
                class_name=str(row["class_name"]),
                original_filename=str(row["original_filename"]),
                source_path=str(row["source_path"]),
            )
        )
    return records
