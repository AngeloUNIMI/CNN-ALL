from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator

import torch
import torch.nn.functional as F

from cnn_all.config import PCANetConfig


BatchFactory = Callable[[], Iterable[torch.Tensor]]


@dataclass
class PCANetFitSummary:
    num_filters: list[int]
    retained_variance: list[float]
    eigenvalues: list[list[float]]


class TorchPCANet:
    """GPU-oriented PCANet implementation matching the active MATLAB path.

    PCA filters are learned from per-patch mean-centred vectors. Filter
    responses are computed from zero-padded, mean-centred patches. The final
    response maps are binary-hashed and summarized by local histograms.
    """

    def __init__(self, cfg: PCANetConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.filters: list[torch.Tensor] = []  # each K x D
        self.summary: PCANetFitSummary | None = None
        if cfg.dtype == "float64":
            self.dtype = torch.float64
        elif cfg.dtype == "float32":
            self.dtype = torch.float32
        else:
            raise ValueError("pcanet.dtype must be float32 or float64")

    @staticmethod
    def _mean_centered_patches(
        x: torch.Tensor,
        patch_size: int,
        padding: int = 0,
    ) -> torch.Tensor:
        patches = F.unfold(x, kernel_size=patch_size, stride=1, padding=padding)
        return patches - patches.mean(dim=1, keepdim=True)

    def _apply_filter_bank(self, x: torch.Tensor, stage: int) -> torch.Tensor:
        patch_size = int(self.cfg.patch_sizes[stage])
        patches = self._mean_centered_patches(x, patch_size, padding=patch_size // 2)
        filters = self.filters[stage].to(device=x.device, dtype=x.dtype)
        # filters: K x D, patches: B x D x L -> B x K x L
        responses = torch.einsum("kd,bdl->bkl", filters, patches)
        return responses.reshape(x.shape[0], filters.shape[0], x.shape[-2], x.shape[-1])

    def _input_for_stage(self, base: torch.Tensor, stage: int) -> torch.Tensor:
        current = base
        for previous in range(stage):
            response = self._apply_filter_bank(current, previous)
            current = response.reshape(-1, 1, response.shape[-2], response.shape[-1])
        return current

    def _accumulate_covariance(
        self,
        batch_factory: BatchFactory,
        stage: int,
    ) -> tuple[torch.Tensor, int]:
        patch_size = int(self.cfg.patch_sizes[stage])
        covariance: torch.Tensor | None = None
        total_patches = 0
        seen_base_images = 0

        for base in batch_factory():
            if seen_base_images >= self.cfg.max_training_images:
                break
            base = base.to(self.device, dtype=self.dtype, non_blocking=True)
            if seen_base_images + base.shape[0] > self.cfg.max_training_images:
                base = base[: self.cfg.max_training_images - seen_base_images]
            seen_base_images += base.shape[0]

            current = self._input_for_stage(base, stage)
            patches = self._mean_centered_patches(current, patch_size, padding=0)
            # D x (B*L), matching MATLAB's im2col_mean_removal output.
            patches = patches.permute(1, 0, 2).reshape(patches.shape[1], -1)
            if covariance is None:
                covariance = torch.zeros(
                    (patches.shape[0], patches.shape[0]),
                    device=self.device,
                    dtype=self.dtype,
                )
            chunk_size = max(1, int(self.cfg.covariance_patch_chunk))
            for start in range(0, patches.shape[1], chunk_size):
                block = patches[:, start : start + chunk_size]
                covariance.addmm_(block, block.transpose(0, 1))
                total_patches += block.shape[1]
            del patches, current, base

        if covariance is None or total_patches == 0:
            raise RuntimeError("PCANet received no patches for PCA fitting")
        covariance /= float(total_patches)
        return covariance, total_patches

    def _select_num_filters(self, eigenvalues_desc: torch.Tensor, stage: int) -> tuple[int, float]:
        maximum = min(int(self.cfg.max_filters[stage]), eigenvalues_desc.numel())
        total = eigenvalues_desc.clamp_min(0).sum()
        if float(total) <= torch.finfo(eigenvalues_desc.dtype).eps:
            return maximum, 0.0
        if not self.cfg.dynamic_filters:
            retained = float(eigenvalues_desc[:maximum].sum() / total)
            return maximum, retained

        target = float(self.cfg.retained_variance[stage])
        cumulative = torch.cumsum(eigenvalues_desc.clamp_min(0), dim=0) / total
        candidates = torch.nonzero(cumulative >= target, as_tuple=False)
        first = int(candidates[0].item()) + 1 if candidates.numel() else maximum
        first = min(max(1, first), maximum)

        # Preserve the MATLAB choice between the first component count that
        # reaches the target and one additional component, whichever is closer.
        possible = min(first + 1, maximum)
        diff_first = abs(float(cumulative[first - 1]) - target)
        diff_possible = abs(float(cumulative[possible - 1]) - target)
        chosen = first if diff_first < diff_possible else possible
        retained = float(cumulative[chosen - 1])
        return chosen, retained

    def fit(self, batch_factory: BatchFactory, logger=None) -> PCANetFitSummary:
        self.filters.clear()
        num_filters: list[int] = []
        retained_variance: list[float] = []
        eigenvalue_lists: list[list[float]] = []

        for stage in range(self.cfg.num_stages):
            if logger:
                logger.info("PCANet stage %d/%d: covariance", stage + 1, self.cfg.num_stages)
            covariance, total_patches = self._accumulate_covariance(batch_factory, stage)
            if logger:
                logger.info(
                    "PCANet stage %d: eigendecomposition of %dx%d covariance (%d patches)",
                    stage + 1,
                    covariance.shape[0],
                    covariance.shape[1],
                    total_patches,
                )
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            order = torch.argsort(eigenvalues, descending=True)
            eigenvalues = eigenvalues[order]
            eigenvectors = eigenvectors[:, order]
            count, retained = self._select_num_filters(eigenvalues, stage)
            filters = eigenvectors[:, :count].transpose(0, 1).contiguous()
            self.filters.append(filters.detach())
            num_filters.append(count)
            retained_variance.append(retained)
            eigenvalue_lists.append(eigenvalues.detach().cpu().float().tolist())
            if logger:
                logger.info(
                    "PCANet stage %d: selected %d filters, retained variance %.4f",
                    stage + 1,
                    count,
                    retained,
                )
            del covariance, eigenvalues, eigenvectors

        self.summary = PCANetFitSummary(
            num_filters=num_filters,
            retained_variance=retained_variance,
            eigenvalues=eigenvalue_lists,
        )
        return self.summary

    def _final_responses(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        current = x
        groups = 1
        for stage in range(self.cfg.num_stages):
            response = self._apply_filter_bank(current, stage)
            k = response.shape[1]
            if stage == self.cfg.num_stages - 1:
                return response.reshape(batch_size, groups, k, response.shape[-2], response.shape[-1])
            current = response.reshape(-1, 1, response.shape[-2], response.shape[-1])
            groups *= k
        raise AssertionError("Unreachable")

    def _hash_histogram(self, responses: torch.Tensor) -> torch.Tensor:
        # B x G x K x H x W
        b, groups, k, h, w = responses.shape
        if k > 20:
            raise ValueError(
                "Hashing 2^K bins becomes impractical for K>20; reduce retained filters"
            )
        weights = (2 ** torch.arange(k - 1, -1, -1, device=responses.device)).to(torch.long)
        codes = ((responses > 0).to(torch.long) * weights.view(1, 1, k, 1, 1)).sum(dim=2)
        codes = codes.reshape(b * groups, 1, h, w).float()

        block_h, block_w = int(self.cfg.hist_block_size[0]), int(self.cfg.hist_block_size[1])
        stride_h = max(1, int(round((1.0 - self.cfg.block_overlap_ratio) * block_h)))
        stride_w = max(1, int(round((1.0 - self.cfg.block_overlap_ratio) * block_w)))
        if h < block_h or w < block_w:
            pad_h = max(0, block_h - h)
            pad_w = max(0, block_w - w)
            codes = F.pad(codes, (0, pad_w, 0, pad_h), value=0)

        blocks = F.unfold(
            codes,
            kernel_size=(block_h, block_w),
            stride=(stride_h, stride_w),
        ).long()
        # BG x area x num_blocks -> BG x num_blocks x area
        blocks = blocks.transpose(1, 2)
        num_blocks = blocks.shape[1]
        bins = 2**k
        histogram = torch.zeros(
            (b * groups, num_blocks, bins),
            device=responses.device,
            dtype=responses.dtype,
        )
        histogram.scatter_add_(2, blocks, torch.ones_like(blocks, dtype=responses.dtype))
        histogram *= bins / histogram.sum(dim=2, keepdim=True).clamp_min(1.0)
        histogram = histogram.reshape(b, groups * num_blocks, bins)
        # MATLAB vectorizes transpose([Bhist{:}]), i.e. bin-major ordering.
        return histogram.transpose(1, 2).reshape(b, -1)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if len(self.filters) != self.cfg.num_stages:
            raise RuntimeError("PCANet must be fitted before transform")
        x = x.to(self.device, dtype=self.dtype, non_blocking=True)
        responses = self._final_responses(x)
        return self._hash_histogram(responses)

    def save(self, path: str | Path) -> None:
        if self.summary is None:
            raise RuntimeError("Cannot save an unfitted PCANet")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.cfg),
                "filters": [f.detach().cpu() for f in self.filters],
                "summary": asdict(self.summary),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device) -> "TorchPCANet":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        cfg = PCANetConfig(**payload["config"])
        model = cls(cfg, device=device)
        model.filters = [f.to(device=device, dtype=model.dtype) for f in payload["filters"]]
        model.summary = PCANetFitSummary(**payload["summary"])
        return model
