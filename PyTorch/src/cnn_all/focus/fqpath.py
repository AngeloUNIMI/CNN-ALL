from __future__ import annotations

import math
from functools import lru_cache
from importlib.resources import files

import numpy as np
import torch
import torch.nn.functional as F


def _as_bchw(image: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if image.ndim == 3:
        return image.unsqueeze(0), True
    if image.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(image.shape)}")
    return image, False


def symmetric_indices(length: int, before: int, after: int, device: torch.device) -> torch.Tensor:
    """Indices implementing MATLAB-style symmetric padding (edge repeated)."""
    if length <= 0:
        raise ValueError("length must be positive")
    raw = torch.arange(-before, length + after, device=device, dtype=torch.long)
    period = 2 * length
    mapped = torch.remainder(raw, period)
    mapped = torch.where(mapped < length, mapped, period - 1 - mapped)
    return mapped


def symmetric_pad2d(
    x: torch.Tensor,
    left: int,
    right: int,
    top: int,
    bottom: int,
) -> torch.Tensor:
    if top or bottom:
        idx_h = symmetric_indices(x.shape[-2], top, bottom, x.device)
        x = x.index_select(-2, idx_h)
    if left or right:
        idx_w = symmetric_indices(x.shape[-1], left, right, x.device)
        x = x.index_select(-1, idx_w)
    return x


@lru_cache(maxsize=1)
def _kernel_numpy() -> np.ndarray:
    path = files("cnn_all.focus.assets").joinpath("fqpath_kernel.npy")
    with path.open("rb") as f:
        return np.load(f).astype(np.float64)


def fqpath_kernel(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(_kernel_numpy(), device=device, dtype=dtype)


def generalized_gaussian_kernel(
    device: torch.device,
    dtype: torch.dtype,
    half_width: int = 16,
    amplitude: float = 1.0,
    sigma: float = 2.0,
    beta: float = 1.5,
) -> torch.Tensor:
    x = torch.arange(-half_width, half_width + 1, device=device, dtype=dtype)
    gamma_1 = math.gamma(1.0 / beta)
    gamma_3 = math.gamma(3.0 / beta)
    a = sigma * math.sqrt(gamma_1 / gamma_3)
    denom = 2.0 * math.gamma(1.0 + 1.0 / beta) * a
    kernel = amplitude / denom * torch.exp(-torch.abs(x / a).pow(beta))
    return kernel / kernel.sum()


def rgb_to_gray(image: torch.Tensor) -> torch.Tensor:
    x, squeezed = _as_bchw(image)
    if x.shape[1] == 1:
        gray = x
    elif x.shape[1] >= 3:
        weights = torch.tensor(
            [0.29893602, 0.58704307, 0.11402090],
            device=x.device,
            dtype=x.dtype,
        ).view(1, 3, 1, 1)
        gray = (x[:, :3] * weights).sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported channel count: {x.shape[1]}")
    return gray.squeeze(0) if squeezed else gray


def _conv_horizontal(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    radius = kernel.numel() // 2
    padded = symmetric_pad2d(x, radius, radius, 0, 0)
    weight = kernel.view(1, 1, 1, -1)
    return F.conv2d(padded, weight)


def _conv_vertical(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    radius = kernel.numel() // 2
    padded = symmetric_pad2d(x, 0, 0, radius, radius)
    weight = kernel.view(1, 1, -1, 1)
    return F.conv2d(padded, weight)


def fqpath_score(image: torch.Tensor) -> torch.Tensor:
    """Compute the FQPath non-reference focus score for a batch.

    Input must be float data in a consistent numeric range, normally [0, 1].
    The implementation ports the supplied MATLAB FQPath algorithm and uses
    batched GPU convolutions; the small histogram/moment reduction is done
    per image because the positive-response masks have variable length.

    Higher scores indicate more blur, matching the MATLAB implementation.
    """

    x, squeezed = _as_bchw(image)
    if not x.is_floating_point():
        x = x.float() / 255.0
    gray = rgb_to_gray(x)
    if gray.ndim == 3:
        gray = gray.unsqueeze(0)

    work_dtype = torch.float64 if gray.device.type == "cpu" else torch.float32
    gray = gray.to(dtype=work_dtype)
    low = generalized_gaussian_kernel(gray.device, gray.dtype)
    band = fqpath_kernel(gray.device, gray.dtype)

    vertical_response = _conv_horizontal(_conv_horizontal(gray, low), band)
    horizontal_response = _conv_vertical(_conv_vertical(gray, low), band)

    scores: list[torch.Tensor] = []
    eps = torch.finfo(gray.dtype).tiny

    for i in range(gray.shape[0]):
        v = vertical_response[i, 0]
        h = horizontal_response[i, 0]
        mask = (v > 0) & (h > 0)
        if not bool(mask.any()):
            scores.append(torch.tensor(120.0, device=gray.device, dtype=gray.dtype))
            continue

        vv = v[mask].abs()
        hh = h[mask].abs()
        combined = torch.cat([vv, hh], dim=0)
        min_val = combined.min()
        max_val = combined.max()
        if not torch.isfinite(max_val) or float(max_val.abs()) <= eps:
            scores.append(torch.tensor(120.0, device=gray.device, dtype=gray.dtype))
            continue

        # MATLAB ``hist(v(:), 50)`` uses 50 equally spaced *centres* and
        # assigns samples according to the midpoints between adjacent centres.
        # ``torch.histc`` uses edges instead, so bucketize explicitly for parity.
        centres = torch.linspace(min_val, max_val, 50, device=gray.device, dtype=gray.dtype)
        if float(max_val - min_val) <= eps:
            hist = torch.zeros(50, device=gray.device, dtype=gray.dtype)
            hist[24] = combined.numel()
        else:
            boundaries = 0.5 * (centres[:-1] + centres[1:])
            bin_indices = torch.bucketize(combined, boundaries, right=False)
            hist = torch.bincount(bin_indices, minlength=50).to(dtype=gray.dtype)
        cdf = hist.cumsum(dim=0) / hist.sum().clamp_min(eps)
        cutoff = cdf.min() + 0.95 * (cdf.max() - cdf.min())
        count_below = int((cdf < cutoff).sum().item())
        centre_index = max(0, min(49, count_below - 1))
        sigma_approx = centres[centre_index] / centres[-1].clamp_min(eps)
        fraction = (1.0 - torch.tanh(60.0 * (sigma_approx - 0.095))) / 4.0 + 0.09

        # p_norm = 1/2 in the original code.
        feature = (torch.sqrt(vv) + torch.sqrt(hh)).pow(2)
        number = int(torch.round(fraction * feature.numel()).clamp(1, feature.numel()).item())
        top = torch.topk(feature, k=number, largest=True, sorted=False).values
        mean = top.mean()
        moment4 = (top - mean).pow(4).mean().abs()
        if float(moment4) <= eps:
            score = torch.tensor(120.0, device=gray.device, dtype=gray.dtype)
        else:
            score = -torch.log10(moment4)
            score = torch.nan_to_num(score, nan=120.0, posinf=120.0, neginf=0.0)
        scores.append(score)

    result = torch.stack(scores).to(dtype=torch.float32)
    return result.squeeze(0) if squeezed else result
