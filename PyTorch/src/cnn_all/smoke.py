from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


CLASSES = ["Basophil", "Eosinophil", "Lymphocyte", "Monocyte", "Neutrophil"]


def make_smoke_dataset(root: str | Path, images_per_class: int = 4, size: int = 64) -> Path:
    root = Path(root)
    rng = random.Random(7)
    for class_index, class_name in enumerate(CLASSES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for i in range(images_per_class):
            image = Image.new("RGB", (size, size), (230, 208, 205))
            draw = ImageDraw.Draw(image, "RGBA")
            # RBC-like background objects.
            for _ in range(9):
                x = rng.randint(-5, size - 5)
                y = rng.randint(-5, size - 5)
                r = rng.randint(5, 9)
                draw.ellipse((x - r, y - r, x + r, y + r), fill=(100, 70, 125, 130))
            cx = size // 2 + rng.randint(-3, 3)
            cy = size // 2 + rng.randint(-3, 3)
            radius = 13 + class_index
            draw.ellipse(
                (cx - radius - 3, cy - radius - 3, cx + radius + 3, cy + radius + 3),
                fill=(205, 135, 180, 120),
            )
            # Class-dependent lobed nucleus pattern.
            lobes = 1 + (class_index % 3)
            for lobe in range(lobes):
                angle = 2 * math.pi * lobe / lobes
                lx = cx + int(math.cos(angle) * 5)
                ly = cy + int(math.sin(angle) * 5)
                rr = max(6, radius - 4)
                draw.ellipse((lx - rr, ly - rr, lx + rr, ly + rr), fill=(110, 15, 125, 235))
            if i % 2:
                image = image.filter(ImageFilter.GaussianBlur(radius=0.8 + 0.2 * class_index))
            image.save(class_dir / f"sample_{i:03d}.png")
    return root
