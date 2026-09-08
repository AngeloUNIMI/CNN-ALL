from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
from torch import nn

from cnn_all.classifiers.metrics import classification_metrics
from cnn_all.cnn.backbones import (
    BackboneSpec,
    build_backbone,
    model_parameter_groups,
    replace_classifier,
)
from cnn_all.cnn.preprocessing import prepare_cnn_batch
from cnn_all.config import AppConfig
from cnn_all.data.datasets import build_loader, make_balanced_sampler
from cnn_all.data.records import ImageRecord
from cnn_all.focus.curves import FocusCurves
from cnn_all.utils import atomic_json_dump


def _class_weights(labels: Sequence[int], num_classes: int, device: torch.device) -> torch.Tensor:
    y = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.bincount(y, minlength=num_classes).float()
    weights = counts.sum() / (counts.clamp_min(1.0) * num_classes)
    return weights.to(device)


def evaluate_model(
    model: nn.Module,
    records: Sequence[ImageRecord],
    subset: Sequence[int],
    curves: FocusCurves,
    focus_threshold: float,
    view: str,
    cfg: AppConfig,
    device: torch.device,
) -> tuple[dict, list[int], list[int], list[int], list[int]]:
    loader = build_loader(
        records,
        subset=subset,
        batch_size=cfg.cnn.feature_batch_size,
        runtime=cfg.runtime,
        shuffle=False,
    )
    model.eval()
    true: list[int] = []
    pred: list[int] = []
    ordered_indices: list[int] = []
    selected_variants: list[int] = []
    amp_enabled = cfg.cnn.amp and device.type == "cuda"
    with torch.inference_mode():
        for batch in loader:
            images, selected = prepare_cnn_batch(
                batch["image"],
                batch["index"],
                cfg,
                curves,
                focus_threshold,
                view,
                device,
                train=False,
            )
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = model(images)
            predictions = logits.argmax(dim=1)
            true.extend(int(v) for v in batch["label"].tolist())
            pred.extend(int(v) for v in predictions.cpu().tolist())
            ordered_indices.extend(int(v) for v in batch["index"].tolist())
            selected_variants.extend(int(v) for v in selected.cpu().tolist())
    metrics = classification_metrics(true, pred, cfg.dataset.classes)
    return metrics, true, pred, ordered_indices, selected_variants


def train_finetuned_model(
    backbone_name: str,
    records: Sequence[ImageRecord],
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    curves: FocusCurves,
    focus_threshold: float,
    view: str,
    cfg: AppConfig,
    device: torch.device,
    output_dir: str | Path,
    logger=None,
    run_seed: int | None = None,
) -> tuple[nn.Module, BackboneSpec, dict, list[dict]]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "model_final.pt"
    checkpoint_path = output_dir / "checkpoint_latest.pt"

    if run_seed is not None:
        torch.manual_seed(run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(run_seed)

    model, spec = build_backbone(
        backbone_name,
        pretrained=cfg.cnn.pretrained,
        allow_download=cfg.cnn.download_weights,
    )
    head = replace_classifier(model, spec, len(cfg.dataset.classes))
    model = model.to(device)
    if cfg.cnn.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    if cfg.runtime.resume and final_path.is_file():
        payload = torch.load(final_path, map_location=device, weights_only=False)
        model.load_state_dict(payload["model"])
        if logger:
            logger.info("Loaded completed fine-tuned model: %s", final_path)
        metrics, true, pred, ordered, selected = evaluate_model(
            model,
            records,
            test_indices,
            curves,
            focus_threshold,
            view,
            cfg,
            device,
        )
        atomic_json_dump(
            {
                "metrics": metrics,
                "true": true,
                "predictions": pred,
                "indices": ordered,
                "selected_variants": selected,
            },
            output_dir / "evaluation.json",
        )
        return model, spec, metrics, payload.get("history", [])

    train_labels = [records[i].label for i in train_indices]
    sampler = make_balanced_sampler(train_labels) if cfg.cnn.balanced_sampler else None
    loader = build_loader(
        records,
        subset=train_indices,
        batch_size=cfg.cnn.batch_size,
        runtime=cfg.runtime,
        shuffle=sampler is None,
        sampler=sampler,
    )

    parameters = model_parameter_groups(
        model,
        head,
        base_lr=cfg.cnn.learning_rate,
        head_multiplier=cfg.cnn.head_lr_multiplier,
    )
    optimizer = torch.optim.SGD(
        parameters,
        lr=cfg.cnn.learning_rate,
        momentum=cfg.cnn.momentum,
        weight_decay=cfg.cnn.weight_decay,
    )
    weights = (
        _class_weights(train_labels, len(cfg.dataset.classes), device)
        if cfg.cnn.class_weighted_loss
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.cnn.label_smoothing)
    amp_enabled = cfg.cnn.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    train_model: nn.Module = model
    if cfg.cnn.compile and hasattr(torch, "compile"):
        if logger:
            logger.info("Compiling %s with torch.compile", backbone_name)
        train_model = torch.compile(model)

    history: list[dict] = []
    global_step = 0
    start_epoch = 1
    if cfg.runtime.resume and checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        history = list(checkpoint.get("history", []))
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        if logger:
            logger.info(
                "Resuming %s/%s from epoch %d (%s)",
                backbone_name,
                view,
                start_epoch,
                checkpoint_path,
            )

    if logger:
        logger.info(
            "Fine-tuning %s/%s: %d images, %d epochs, batch size %d",
            backbone_name,
            view,
            len(train_indices),
            cfg.cnn.epochs,
            cfg.cnn.batch_size,
        )

    for epoch in range(start_epoch, cfg.cnn.epochs + 1):
        train_model.train()
        start = time.perf_counter()
        running_loss = 0.0
        correct = 0
        seen = 0

        for step, batch in enumerate(loader, start=1):
            labels = batch["label"].to(device, non_blocking=True)
            images, _ = prepare_cnn_batch(
                batch["image"],
                batch["index"],
                cfg,
                curves,
                focus_threshold,
                view,
                device,
                train=True,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits = train_model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = labels.shape[0]
            running_loss += float(loss.detach()) * batch_size
            correct += int((logits.detach().argmax(dim=1) == labels).sum())
            seen += batch_size
            global_step += 1
            if logger and (step % cfg.cnn.log_every_steps == 0 or step == len(loader)):
                logger.info(
                    "%s/%s epoch %d/%d step %d/%d loss %.5f accuracy %.4f",
                    backbone_name,
                    view,
                    epoch,
                    cfg.cnn.epochs,
                    step,
                    len(loader),
                    running_loss / max(seen, 1),
                    correct / max(seen, 1),
                )

        epoch_row = {
            "epoch": epoch,
            "loss": running_loss / max(seen, 1),
            "accuracy": correct / max(seen, 1),
            "seconds": time.perf_counter() - start,
            "global_step": global_step,
        }
        history.append(epoch_row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

        if epoch % cfg.cnn.checkpoint_every_epochs == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step,
                    "history": history,
                    "backbone": backbone_name,
                    "view": view,
                    "focus_threshold": focus_threshold,
                },
                output_dir / "checkpoint_latest.pt",
            )

    metrics, true, pred, ordered, selected = evaluate_model(
        model,
        records,
        test_indices,
        curves,
        focus_threshold,
        view,
        cfg,
        device,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "history": history,
            "backbone": backbone_name,
            "view": view,
            "focus_threshold": focus_threshold,
            "class_names": cfg.dataset.classes,
        },
        final_path,
    )
    atomic_json_dump(
        {
            "metrics": metrics,
            "true": true,
            "predictions": pred,
            "indices": ordered,
            "selected_variants": selected,
        },
        output_dir / "evaluation.json",
    )
    if logger:
        logger.info(
            "Fine-tuned %s/%s accuracy %.4f macro-F1 %.4f",
            backbone_name,
            view,
            metrics["accuracy"],
            metrics["macro_f1"],
        )
    return model, spec, metrics, history
