from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage import color, exposure, filters, measure, morphology, segmentation, transform
from sklearn.cluster import KMeans


def comprehensive_colour_normalization(
    image: np.ndarray,
    threshold: float = 1e-12,
    max_iterations: int = 1000,
) -> np.ndarray:
    """Port of the Finlayson comprehensive colour normalization loop.

    The MATLAB code alternates per-pixel chromaticity normalization and
    per-channel illumination normalization until the L2-norm change falls
    below ``threshold``. A finite iteration cap and epsilon guards are added
    for numerical safety.
    """

    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 3:
        raise ValueError("Expected an HxWxC image")
    previous = x.copy()
    eps = np.finfo(np.float64).eps
    for _ in range(max_iterations):
        denom_pixel = previous.sum(axis=2, keepdims=True)
        normalized = previous / np.maximum(denom_pixel, eps)
        denom_channel = normalized.sum(axis=(0, 1), keepdims=True)
        normalized = normalized.shape[0] * normalized.shape[1] * normalized / np.maximum(
            denom_channel, eps
        )
        difference = np.linalg.norm(previous.ravel()) - np.linalg.norm(normalized.ravel())
        previous = normalized
        if difference <= threshold:
            break
    out = previous
    out -= np.nanmin(out)
    maxv = np.nanmax(out)
    if maxv > 0:
        out /= maxv
    return np.clip(out, 0.0, 1.0)


@dataclass
class LegacySegmentationConfig:
    axis_scale: float = 1.5
    roi_size: tuple[int, int] = (256, 256)
    morphology_radius: int = 5
    chan_vese_iterations: int = 200
    kmeans_clusters: int = 3


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels = measure.label(mask, connectivity=2)
    if labels.max() == 0:
        return mask.astype(bool)
    counts = np.bincount(labels.ravel())
    counts[0] = 0
    return labels == counts.argmax()


def segment_wbc_legacy_approx(
    image: np.ndarray,
    cfg: LegacySegmentationConfig,
) -> np.ndarray:
    """Approximate the legacy Otsu + fuzzy-colour + Chan-Vese segmentation.

    This is intentionally isolated as a compatibility module. The active
    RAABIN-WBC configuration uses existing cell crops and does not call it.
    """

    rgb = np.asarray(image, dtype=np.float32)
    if rgb.max() > 1.0:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0.0, 1.0)

    norm = comprehensive_colour_normalization(rgb)
    gray = color.rgb2gray(norm)
    try:
        otsu = filters.threshold_otsu(gray)
    except ValueError:
        otsu = float(gray.mean())
    mask_intensity = gray < otsu

    hsv = color.rgb2hsv(rgb)
    pixels = hsv.reshape(-1, 3)
    km = KMeans(n_clusters=cfg.kmeans_clusters, n_init=5, random_state=0)
    lab = km.fit_predict(pixels).reshape(hsv.shape[:2])
    centers = km.cluster_centers_
    # Cell material is typically more saturated and/or darker than the slide
    # background. Keep the two most cell-like clusters.
    cell_score = centers[:, 1] + (1.0 - centers[:, 2])
    selected = np.argsort(cell_score)[-min(2, cfg.kmeans_clusters) :]
    mask_colour = np.isin(lab, selected)

    mask = mask_intensity | mask_colour
    mask = ndi.binary_fill_holes(mask)
    selem = morphology.disk(max(1, cfg.morphology_radius))
    mask = morphology.binary_closing(mask, selem)
    mask = morphology.binary_opening(mask, selem)
    mask = _largest_component(mask)

    # Morphological Chan-Vese is a robust headless counterpart of MATLAB's
    # Chan-Vese activecontour call.
    try:
        mask = segmentation.morphological_chan_vese(
            gray,
            num_iter=cfg.chan_vese_iterations,
            init_level_set=mask,
            smoothing=1,
            lambda1=1,
            lambda2=1,
        )
    except Exception as exc:  # pragma: no cover - fallback for pathological images
        warnings.warn(f"Chan-Vese refinement failed; using morphology mask: {exc}")
    mask = ndi.binary_fill_holes(_largest_component(mask))
    return mask.astype(bool)


def extract_centered_roi_legacy_approx(
    image: np.ndarray,
    mask: np.ndarray,
    cfg: LegacySegmentationConfig,
) -> np.ndarray:
    """Extract a centered square ROI using the component minor-axis length.

    The original MATLAB code deliberately centers the crop at the image
    centre rather than at the fitted ellipse centre. That behavior is kept.
    """

    rgb = np.asarray(image)
    props = measure.regionprops(measure.label(mask.astype(np.uint8)))
    if not props:
        raise ValueError("No connected component found for ROI extraction")
    region = max(props, key=lambda p: p.area)
    minor = max(float(region.axis_minor_length), 2.0)
    radius = max(1, int(round((minor / 2.0) * cfg.axis_scale)))

    h, w = rgb.shape[:2]
    cy, cx = h // 2, w // 2
    y0, y1 = cy - radius, cy + radius + 1
    x0, x1 = cx - radius, cx + radius + 1
    if y0 < 0 or x0 < 0 or y1 > h or x1 > w:
        radius = max(1, int(round(minor / 2.0)))
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
    crop = rgb[y0:y1, x0:x1]
    if crop.size == 0:
        raise ValueError("Empty ROI")
    resized = transform.resize(
        crop,
        cfg.roi_size,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    )
    return np.clip(resized, 0, 255).astype(np.uint8)


def prepare_legacy_roi(
    image: Image.Image,
    cfg: LegacySegmentationConfig,
    apply_colour_normalization: bool = False,
) -> Image.Image:
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    mask = segment_wbc_legacy_approx(arr, cfg)
    roi = extract_centered_roi_legacy_approx(arr, mask, cfg)
    if apply_colour_normalization:
        roi = (comprehensive_colour_normalization(roi) * 255.0).round().astype(np.uint8)
    return Image.fromarray(roi, mode="RGB")


def stain_deconvolution_ica_approx(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compact ICA-based stain decomposition compatibility implementation.

    It preserves the optical-density/ICA idea used by SCD_MA but omits the
    original multi-resolution wavelet band selection. It is not used by the
    default RAABIN-WBC path.
    """

    from sklearn.decomposition import FastICA

    rgb = np.asarray(image, dtype=np.float64)
    if rgb.max() <= 1.0:
        rgb *= 255.0
    h, w, _ = rgb.shape
    od = -np.log((rgb.reshape(-1, 3).T + 1.0) / 255.0)
    ica = FastICA(n_components=3, whiten="unit-variance", random_state=0, max_iter=1000)
    sources = ica.fit_transform(od.T).T
    vectors = np.abs(ica.mixing_)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=0, keepdims=True), 1e-12)
    reconstructed = []
    for i in range(3):
        stain = 255.0 * np.exp(-vectors[:, i : i + 1] @ sources[i : i + 1])
        reconstructed.append(np.clip(stain.T.reshape(h, w, 3), 0, 255).astype(np.uint8))
    return tuple(reconstructed)  # type: ignore[return-value]
