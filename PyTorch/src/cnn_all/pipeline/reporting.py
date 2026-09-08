from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_average_cmc(
    original_curves: Sequence[Sequence[float]],
    unsharp_curves: Sequence[Sequence[float]],
    output_dir: str | Path,
) -> None:
    """Save headless average CMC CSV/PNG outputs matching stampaAvgCMC intent."""

    if not original_curves or not unsharp_curves:
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width = max(
        max(len(curve) for curve in original_curves),
        max(len(curve) for curve in unsharp_curves),
        30,
    )

    def padded_mean(curves: Sequence[Sequence[float]]) -> np.ndarray:
        rows: list[np.ndarray] = []
        for curve in curves:
            arr = np.asarray(curve, dtype=np.float64)
            if arr.size < width:
                arr = np.pad(arr, (0, width - arr.size), constant_values=1.0)
            else:
                arr = arr[:width]
            rows.append(arr)
        return np.mean(np.stack(rows, axis=0), axis=0)

    original = padded_mean(original_curves)
    unsharp = padded_mean(unsharp_curves)
    ranks = np.arange(1, width + 1)
    pd.DataFrame(
        {
            "rank": ranks,
            "original": original,
            "unsharp": unsharp,
        }
    ).to_csv(output_dir / "cmc_average.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ranks, original, marker="o", markevery=max(1, width // 10), label="Original")
    ax.plot(ranks, unsharp, marker="s", markevery=max(1, width // 10), label="Unsharp")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Identification rate")
    ax.set_xlim(1, min(width, 30))
    ax.set_ylim(0.0, 1.01)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "cmc_average.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
