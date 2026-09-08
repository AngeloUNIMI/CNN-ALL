from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset, Sampler, WeightedRandomSampler
from torchvision.transforms.functional import pil_to_tensor

from cnn_all.config import RuntimeConfig
from cnn_all.data.records import ImageRecord
from cnn_all.utils import worker_seed_fn


class ImageRecordDataset(Dataset[dict[str, object]]):
    """Dataset that returns uint8 CHW tensors and stable global indices."""

    def __init__(self, records: Sequence[ImageRecord], subset: Sequence[int] | None = None):
        self.records = list(records)
        self.indices = list(range(len(records))) if subset is None else [int(i) for i in subset]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, object]:
        global_index = self.indices[item]
        record = self.records[global_index]
        with Image.open(record.path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = pil_to_tensor(image)  # uint8, CxHxW
        return {
            "image": tensor,
            "label": record.label,
            "index": global_index,
            "path": record.path,
            "filename": record.original_filename,
            "class_name": record.class_name,
        }


def _collate(batch: list[dict[str, object]]) -> dict[str, object]:
    images = torch.stack([item["image"] for item in batch])
    labels = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
    indices = torch.tensor([int(item["index"]) for item in batch], dtype=torch.long)
    return {
        "image": images,
        "label": labels,
        "index": indices,
        "path": [str(item["path"]) for item in batch],
        "filename": [str(item["filename"]) for item in batch],
        "class_name": [str(item["class_name"]) for item in batch],
    }


def build_loader(
    records: Sequence[ImageRecord],
    subset: Sequence[int] | None,
    batch_size: int,
    runtime: RuntimeConfig,
    shuffle: bool = False,
    sampler: Sampler[int] | None = None,
    drop_last: bool = False,
) -> DataLoader:
    dataset = ImageRecordDataset(records, subset=subset)
    kwargs: dict[str, object] = {
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle and sampler is None),
        "sampler": sampler,
        "num_workers": int(runtime.num_workers),
        "pin_memory": bool(runtime.pin_memory),
        "drop_last": drop_last,
        "collate_fn": _collate,
        "worker_init_fn": worker_seed_fn,
    }
    if runtime.num_workers > 0:
        kwargs["persistent_workers"] = bool(runtime.persistent_workers)
        kwargs["prefetch_factor"] = int(runtime.prefetch_factor)
    return DataLoader(dataset, **kwargs)


def make_balanced_sampler(labels: Sequence[int]) -> WeightedRandomSampler:
    y = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(y)
    weights_per_class = torch.zeros_like(counts, dtype=torch.float64)
    nonzero = counts > 0
    weights_per_class[nonzero] = 1.0 / counts[nonzero].double()
    sample_weights = weights_per_class[y]
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
