from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from cnn_all.config import KNNConfig


@dataclass
class KNNResult:
    predictions: np.ndarray
    distances: np.ndarray
    nearest_indices: np.ndarray


def pairwise_distance_chunked(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    device: torch.device,
    cfg: KNNConfig,
) -> torch.Tensor:
    """Compute a full distance matrix with bounded accelerator memory."""
    q = queries.detach().float().cpu()
    g = gallery.detach().float().cpu()
    result = torch.empty((q.shape[0], g.shape[0]), dtype=torch.float32)

    for qs in range(0, q.shape[0], cfg.query_chunk_size):
        qe = min(qs + cfg.query_chunk_size, q.shape[0])
        q_chunk = q[qs:qe].to(device, non_blocking=True)
        row_parts: list[torch.Tensor] = []
        for gs in range(0, g.shape[0], cfg.gallery_chunk_size):
            ge = min(gs + cfg.gallery_chunk_size, g.shape[0])
            g_chunk = g[gs:ge].to(device, non_blocking=True)
            if cfg.distance == "euclidean":
                distances = torch.cdist(q_chunk, g_chunk, p=2)
            elif cfg.distance in {"chisq", "chi-square", "chi_square"}:
                eps = 1e-12
                a = q_chunk[:, None, :]
                b = g_chunk[None, :, :]
                distances = 0.5 * ((a - b).pow(2) / (a + b + eps)).sum(dim=2)
            else:
                raise ValueError(f"Unsupported distance: {cfg.distance}")
            row_parts.append(distances.cpu())
        result[qs:qe] = torch.cat(row_parts, dim=1)
    return result


def knn_train_test(
    train_features: torch.Tensor,
    test_features: torch.Tensor,
    train_labels: Sequence[int],
    device: torch.device,
    cfg: KNNConfig,
) -> KNNResult:
    if cfg.neighbors != 1:
        raise NotImplementedError("The original pipeline uses 1-NN; only neighbors=1 is currently supported")
    distances = pairwise_distance_chunked(test_features, train_features, device, cfg)
    nearest = torch.argmin(distances, dim=1)
    labels = torch.as_tensor(train_labels, dtype=torch.long)
    predictions = labels[nearest]
    return KNNResult(
        predictions=predictions.numpy(),
        distances=distances.numpy(),
        nearest_indices=nearest.numpy(),
    )


def knn_leave_one_out(
    features: torch.Tensor,
    labels: Sequence[int],
    device: torch.device,
    cfg: KNNConfig,
) -> KNNResult:
    distances = pairwise_distance_chunked(features, features, device, cfg)
    distances.fill_diagonal_(float("inf"))
    nearest = torch.argmin(distances, dim=1)
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    predictions = label_tensor[nearest]
    return KNNResult(
        predictions=predictions.numpy(),
        distances=distances.numpy(),
        nearest_indices=nearest.numpy(),
    )
