from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from cnn_all.focus.fqpath import symmetric_pad2d


def _as_bchw(x: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if x.ndim == 3:
        return x.unsqueeze(0), True
    if x.ndim != 4:
        raise ValueError(f"Expected CHW or BCHW tensor, got {tuple(x.shape)}")
    return x, False


def _srgb_to_linear(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).pow(2.4))


def _linear_to_srgb(rgb: torch.Tensor) -> torch.Tensor:
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * rgb.clamp_min(0).pow(1.0 / 2.4) - 0.055,
    )


def rgb_to_lab(rgb: torch.Tensor) -> torch.Tensor:
    x, squeezed = _as_bchw(rgb)
    if x.shape[1] != 3:
        raise ValueError("rgb_to_lab expects exactly three channels")
    linear = _srgb_to_linear(x.clamp(0, 1))
    matrix = torch.tensor(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        device=x.device,
        dtype=x.dtype,
    )
    xyz = torch.einsum("ij,bjhw->bihw", matrix, linear)
    white = torch.tensor([0.95047, 1.0, 1.08883], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    xyz = xyz / white
    delta = 6.0 / 29.0
    f = torch.where(xyz > delta**3, xyz.clamp_min(0).pow(1.0 / 3.0), xyz / (3 * delta**2) + 4.0 / 29.0)
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    lab = torch.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], dim=1)
    return lab.squeeze(0) if squeezed else lab


def lab_to_rgb(lab: torch.Tensor) -> torch.Tensor:
    x, squeezed = _as_bchw(lab)
    l, a, b = x[:, 0], x[:, 1], x[:, 2]
    fy = (l + 16.0) / 116.0
    fx = fy + a / 500.0
    fz = fy - b / 200.0
    f = torch.stack([fx, fy, fz], dim=1)
    delta = 6.0 / 29.0
    xyz = torch.where(f > delta, f.pow(3), 3 * delta**2 * (f - 4.0 / 29.0))
    white = torch.tensor([0.95047, 1.0, 1.08883], device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    xyz = xyz * white
    inv = torch.tensor(
        [
            [3.2404542, -1.5371385, -0.4985314],
            [-0.9692660, 1.8760108, 0.0415560],
            [0.0556434, -0.2040259, 1.0572252],
        ],
        device=x.device,
        dtype=x.dtype,
    )
    linear = torch.einsum("ij,bjhw->bihw", inv, xyz)
    rgb = _linear_to_srgb(linear).clamp(0.0, 1.0)
    return rgb.squeeze(0) if squeezed else rgb


def gaussian_kernel1d(sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(math.ceil(2.0 * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma).pow(2))
    return kernel / kernel.sum()


def gaussian_blur_symmetric(image: torch.Tensor, sigma: float) -> torch.Tensor:
    x, squeezed = _as_bchw(image)
    kernel = gaussian_kernel1d(sigma, x.device, x.dtype)
    radius = kernel.numel() // 2
    channels = x.shape[1]

    padded = symmetric_pad2d(x, radius, radius, 0, 0)
    horizontal = kernel.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    x = F.conv2d(padded, horizontal, groups=channels)

    padded = symmetric_pad2d(x, 0, 0, radius, radius)
    vertical = kernel.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    x = F.conv2d(padded, vertical, groups=channels)
    return x.squeeze(0) if squeezed else x


def _unsharp_scalar_channel(
    channel: torch.Tensor,
    radius: float,
    amount: float,
    threshold: float,
    value_range: float,
) -> torch.Tensor:
    blurred = gaussian_blur_symmetric(channel, sigma=radius)
    detail = channel - blurred
    if threshold > 0:
        detail = torch.where(detail.abs() >= threshold * value_range, detail, torch.zeros_like(detail))
    return channel + amount * detail


def unsharp_mask(
    image: torch.Tensor,
    radius: float,
    amount: float = 0.8,
    threshold: float = 0.0,
    lab_luminance_only: bool = True,
) -> torch.Tensor:
    """MATLAB-compatible adaptive unsharp operation.

    Radius is the Gaussian standard deviation and amount defaults to 0.8,
    matching ``imsharpen``. For RGB images, the default sharpens only Lab L*,
    mirroring recent MATLAB behavior. The operation always starts from the
    supplied original tensor; callers should not feed the previous radius's
    output when reproducing the CNN-ALL loop.
    """

    x, squeezed = _as_bchw(image)
    original_dtype = x.dtype
    if not x.is_floating_point():
        x = x.float() / 255.0
    x = x.clamp(0.0, 1.0)

    if x.shape[1] == 3 and lab_luminance_only:
        lab = rgb_to_lab(x)
        l = lab[:, 0:1]
        l_sharp = _unsharp_scalar_channel(l, radius, amount, threshold, value_range=100.0).clamp(0, 100)
        out = lab_to_rgb(torch.cat([l_sharp, lab[:, 1:]], dim=1))
    else:
        out = _unsharp_scalar_channel(x, radius, amount, threshold, value_range=1.0).clamp(0, 1)

    if not original_dtype.is_floating_point:
        out = (out * 255.0).round().to(original_dtype)
    return out.squeeze(0) if squeezed else out
