from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from cnn_all.cnn.backbones import BackboneSpec, get_submodule
from cnn_all.cnn.preprocessing import prepare_cnn_batch
from cnn_all.config import AppConfig
from cnn_all.data.records import ImageRecord
from cnn_all.focus.curves import FocusCurves, select_variant_indices
from cnn_all.utils import empty_cuda_cache


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        # Do not use Module.register_full_backward_hook here. Torchvision
        # backbones contain in-place ReLUs and PyTorch wraps outputs of
        # modules with full backward hooks in a view. A later in-place op can
        # then trigger: "BackwardHookFunctionBackward is a view and is being
        # modified inplace". Instead, attach the gradient hook directly to
        # the target activation tensor during the forward pass.
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self.gradients = grad.detach()

    def _forward_hook(self, _module, _inputs, output):
        activation = output if torch.is_tensor(output) else output[0]
        # Save a detached copy for CAM construction while leaving the actual
        # forward tensor untouched. Registering a Tensor hook captures dL/dA
        # without introducing the problematic module-backward-hook view.
        self.activations = activation.detach().clone()
        if activation.requires_grad:
            activation.register_hook(self._save_gradient)

    def close(self) -> None:
        self.forward_handle.remove()

    def __call__(self, image: torch.Tensor) -> tuple[torch.Tensor, int, float]:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image)
        probs = torch.softmax(logits, dim=1)
        cls = int(probs.argmax(dim=1).item())
        score = probs[0, cls]
        score.backward()
        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients")
        activations = self.activations
        gradients = self.gradients
        weights = gradients.mean(dim=(-2, -1), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bicubic", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min()).clamp_min(1e-12)
        return cam.detach(), cls, float(score.detach())


def _load_raw(record: ImageRecord) -> torch.Tensor:
    arr = np.asarray(Image.open(record.path).convert("RGB"), dtype=np.uint8).copy()
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def save_gradcam_comparisons(
    original_model: nn.Module,
    original_spec: BackboneSpec,
    unsharp_model: nn.Module,
    unsharp_spec: BackboneSpec,
    records: Sequence[ImageRecord],
    test_indices: Sequence[int],
    curves: FocusCurves,
    focus_threshold: float,
    cfg: AppConfig,
    device: torch.device,
    output_dir: str | Path,
    logger=None,
) -> None:
    """Save original/unsharp Grad-CAM pairs with bounded accelerator memory.

    Only one fine-tuned network is placed on the accelerator at a time. This
    is important for VGG-16/VGG-19 and avoids keeping the original model, the
    unsharp model, and optimizer-era allocations in VRAM simultaneously.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_all = select_variant_indices(curves.scores[np.asarray(test_indices)], focus_threshold)
    candidates = [
        (int(idx), int(selected))
        for idx, selected in zip(test_indices, selected_all.tolist())
        if (not cfg.cnn.gradcam_only_sharpened) or selected > 0
    ][: cfg.cnn.gradcam_max_samples]

    if not candidates:
        if logger:
            logger.info("Saved 0 Grad-CAM comparisons to %s", output_dir)
        return

    # Keep only a small, bounded set of raw images in host memory.
    raw_by_index = {idx: _load_raw(records[idx]) for idx, _ in candidates}
    original_results: dict[int, tuple[torch.Tensor, int, float]] = {}

    # Pass 1: original model. It is moved back to CPU before pass 2.
    original_model = original_model.to(device)
    if cfg.cnn.channels_last and device.type == "cuda":
        original_model = original_model.to(memory_format=torch.channels_last)
    original_model.eval()
    cam_original = GradCAM(
        original_model,
        get_submodule(original_model, original_spec.gradcam_layer),
    )
    try:
        for global_index, _ in candidates:
            raw = raw_by_index[global_index]
            index_tensor = torch.tensor([global_index], dtype=torch.long)
            original_input, _ = prepare_cnn_batch(
                raw,
                index_tensor,
                cfg,
                curves,
                focus_threshold,
                "original",
                device,
                train=False,
            )
            original_map, original_cls, original_score = cam_original(original_input)
            original_results[global_index] = (
                original_map.detach().cpu(),
                original_cls,
                original_score,
            )
    finally:
        cam_original.close()
        original_model.to("cpu")
        empty_cuda_cache()

    # Pass 2: unsharp model. Build the paired figures while this model is on
    # the accelerator and the original model is safely resident on the CPU.
    unsharp_model = unsharp_model.to(device)
    if cfg.cnn.channels_last and device.type == "cuda":
        unsharp_model = unsharp_model.to(memory_format=torch.channels_last)
    unsharp_model.eval()
    cam_unsharp = GradCAM(
        unsharp_model,
        get_submodule(unsharp_model, unsharp_spec.gradcam_layer),
    )
    try:
        for global_index, selected_variant in candidates:
            record = records[global_index]
            raw = raw_by_index[global_index]
            index_tensor = torch.tensor([global_index], dtype=torch.long)
            unsharp_input, _ = prepare_cnn_batch(
                raw,
                index_tensor,
                cfg,
                curves,
                focus_threshold,
                "unsharp",
                device,
                train=False,
            )
            unsharp_map, unsharp_cls, unsharp_score = cam_unsharp(unsharp_input)
            original_map, original_cls, original_score = original_results[global_index]

            raw_float = raw[0].float().div(255.0).permute(1, 2, 0).numpy()
            from cnn_all.focus.curves import apply_adaptive_unsharp

            unsharp_display, _ = apply_adaptive_unsharp(
                raw.float().div(255.0).to(device),
                index_tensor.to(device),
                curves,
                focus_threshold,
                cfg.focus.amount,
                cfg.focus.edge_threshold,
                cfg.focus.matlab_rgb_lab_sharpen,
            )
            unsharp_display_np = (
                unsharp_display[0].permute(1, 2, 0).detach().cpu().numpy()
            )

            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes[0, 0].imshow(raw_float)
            axes[0, 0].set_title(
                f"Original | true={record.class_name} | "
                f"pred={cfg.dataset.classes[original_cls]} ({original_score:.3f})"
            )
            axes[0, 1].imshow(raw_float)
            axes[0, 1].imshow(original_map.numpy(), cmap="jet", alpha=0.5)
            axes[0, 1].set_title("Original Grad-CAM")
            axes[1, 0].imshow(unsharp_display_np)
            radius = 0 if selected_variant == 0 else curves.radii[selected_variant - 1]
            axes[1, 0].set_title(
                f"Unsharp r={radius:g} | true={record.class_name} | "
                f"pred={cfg.dataset.classes[unsharp_cls]} ({unsharp_score:.3f})"
            )
            axes[1, 1].imshow(unsharp_display_np)
            axes[1, 1].imshow(unsharp_map.cpu().numpy(), cmap="jet", alpha=0.5)
            axes[1, 1].set_title("Unsharp Grad-CAM")
            for ax in axes.ravel():
                ax.axis("off")
            fig.tight_layout()
            stem = Path(record.original_filename).stem
            fig.savefig(output_dir / f"{stem}_gradcam.png", dpi=160, bbox_inches="tight")
            plt.close(fig)
    finally:
        cam_unsharp.close()
        unsharp_model.to("cpu")
        empty_cuda_cache()

    if logger:
        logger.info("Saved %d Grad-CAM comparisons to %s", len(candidates), output_dir)

