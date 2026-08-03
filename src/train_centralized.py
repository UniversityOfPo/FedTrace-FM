"""Centralized CIFAR-100 ResNet-18 baseline training entry point.

Run from the project root with:
    python -m src.train_centralized --config configs/cifar100_resnet18_baseline.yaml

This script wires together the reusable data, model, training-engine,
metrics, and checkpoint modules for a single-machine (non-federated)
baseline run. It contains no federated, attack, or attribution logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.cifar100 import (  # noqa: E402
    build_dataloaders,
    build_datasets,
    get_class_names,
    split_train_validation,
)
from src.evaluation.metrics import (  # noqa: E402
    compute_balanced_accuracy,
    compute_confusion_matrix,
    macro_precision_recall_f1,
    per_class_accuracy,
)
from src.models.resnet18_cifar import count_parameters, resnet18_cifar  # noqa: E402
from src.training.engine import (  # noqa: E402
    build_grad_scaler,
    build_optimizer,
    build_scheduler,
    evaluate,
    resolve_device,
    train_one_epoch,
)
from src.utils.checkpoint import (  # noqa: E402
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from src.utils.config import ConfigError, TrainingConfig, load_data_config, load_training_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/cifar100_resnet18_baseline.yaml")
MODEL_NAME = "resnet18_cifar"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training entry point."""
    parser = argparse.ArgumentParser(description="Train the centralized CIFAR-100 ResNet-18 baseline.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the training YAML configuration.",
    )
    return parser.parse_args()


def _configure_logging(log_path: Path) -> None:
    """Configure root logging to write to both the console and ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")],
        force=True,
    )


def _serializable_config(config: TrainingConfig) -> dict[str, Any]:
    """Convert a ``TrainingConfig`` into a plain, JSON/pickle-safe dict (Path -> str)."""
    raw = asdict(config)
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in raw.items()}


def save_history_csv(history: list[dict[str, float]], path: Path) -> None:
    """Save per-epoch training history as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)
    logger.info("Saved training history to %s", path)


def save_loss_curve(history: list[dict[str, float]], path: Path) -> None:
    """Save a training-vs-validation loss curve plot."""
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row["train_loss"] for row in history], label="train")
    ax.plot(epochs, [row["validation_loss"] for row in history], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training vs Validation Loss")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved loss curve to %s", path)


def save_top1_curve(history: list[dict[str, float]], path: Path) -> None:
    """Save a training-vs-validation top-1 accuracy curve plot."""
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row["train_top1"] * 100 for row in history], label="train")
    ax.plot(epochs, [row["validation_top1"] * 100 for row in history], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title("Training vs Validation Top-1 Accuracy")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved top-1 accuracy curve to %s", path)


def save_lr_curve(history: list[dict[str, float]], path: Path) -> None:
    """Save a learning-rate-over-epochs plot."""
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epochs, [row["learning_rate"] for row in history])
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved learning rate curve to %s", path)


def save_confusion_matrix_plot(confusion, path: Path) -> None:
    """Save a confusion matrix heatmap (matplotlib only, default colormap)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(confusion, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix plot to %s", path)


def save_per_class_accuracy_csv(class_names: list[str], class_acc, path: Path) -> None:
    """Save per-class accuracy as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "class_name", "accuracy"])
        for idx, name in enumerate(class_names):
            writer.writerow([idx, name, float(class_acc[idx])])
    logger.info("Saved per-class accuracy to %s", path)


def _save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a plain dictionary as an indented JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved %s", path)


def main() -> None:
    """Run centralized training end to end: data, model, train/eval, outputs."""
    args = parse_args()

    try:
        config = load_training_config(args.config)
    except ConfigError as exc:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    output_dir = config.output_dir
    checkpoint_dir = config.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _configure_logging(output_dir / "run.log")
    logger.info("Starting experiment '%s'", config.experiment_name)

    set_seed(config.seed)

    device = resolve_device(config.device)
    gpu_name: str | None = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Using GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
    else:
        logger.info("CUDA not available; using CPU.")

    data_config = load_data_config(config.dataset_config)
    # The official test set is loaded here and used only once, for the final
    # evaluation after training. Per-epoch model selection uses a held-out
    # validation split carved out of the training set instead (see below),
    # so the test set is never touched during training or checkpoint selection.
    _, test_dataset = build_datasets(data_config.data_root, download=data_config.download)
    train_subset, val_subset = split_train_validation(
        data_config.data_root,
        download=data_config.download,
        validation_fraction=config.validation_fraction,
        seed=config.seed,
    )
    # Training config's batch_size/num_workers take precedence over the
    # dataset YAML's defaults; only data_root/download/pin_memory come from
    # the dataset config.
    train_loader, val_loader = build_dataloaders(
        train_subset,
        val_subset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=data_config.pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=data_config.pin_memory,
    )
    class_names = get_class_names(test_dataset)
    if len(class_names) != config.num_classes:
        raise ValueError(
            f"Configured num_classes ({config.num_classes}) does not match the dataset's "
            f"actual class count ({len(class_names)})."
        )

    model = resnet18_cifar(num_classes=config.num_classes).to(device)
    total_params, trainable_params = count_parameters(model)
    logger.info("Model parameters: total=%d trainable=%d", total_params, trainable_params)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = build_optimizer(model, config.optimizer, config.learning_rate, config.momentum, config.weight_decay)
    scheduler = build_scheduler(optimizer, config.scheduler, config.epochs, config.warmup_epochs)
    scaler = build_grad_scaler(device, config.mixed_precision)

    best_ckpt_path = checkpoint_dir / "best.pt"
    last_ckpt_path = checkpoint_dir / "last.pt"

    start_epoch = 0
    best_val_top1 = 0.0
    best_epoch = -1

    if config.resume_from is not None:
        checkpoint = load_checkpoint(config.resume_from, map_location=device)
        restore_training_state(checkpoint, model, optimizer, scheduler, scaler)
        start_epoch = checkpoint["epoch"] + 1
        best_val_top1 = checkpoint["best_val_top1"]
        logger.info(
            "Resumed from %s (starting at epoch %d, best_val_top1=%.4f)",
            config.resume_from,
            start_epoch + 1,
            best_val_top1,
        )

    history: list[dict[str, float]] = []
    training_start = time.perf_counter()

    for epoch in range(start_epoch, config.epochs):
        epoch_start = time.perf_counter()

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            scaler=scaler,
            use_amp=config.mixed_precision,
            grad_clip_norm=config.gradient_clip_norm,
            log_interval=config.log_interval,
        )
        val_metrics, _ = evaluate(model, val_loader, criterion, device, use_amp=config.mixed_precision)

        if scheduler is not None:
            scheduler.step()

        epoch_seconds = time.perf_counter() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d - train_loss=%.4f train_top1=%.4f val_loss=%.4f val_top1=%.4f "
            "val_top5=%.4f lr=%.6f (%.1fs)",
            epoch + 1,
            config.epochs,
            train_metrics.loss,
            train_metrics.top1,
            val_metrics.loss,
            val_metrics.top1,
            val_metrics.top5,
            current_lr,
            epoch_seconds,
        )

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics.loss,
                "train_top1": train_metrics.top1,
                "validation_loss": val_metrics.loss,
                "validation_top1": val_metrics.top1,
                "validation_top5": val_metrics.top5,
                "learning_rate": current_lr,
                "epoch_seconds": epoch_seconds,
            }
        )

        is_best = val_metrics.top1 > best_val_top1
        if is_best:
            best_val_top1 = val_metrics.top1
            best_epoch = epoch + 1

        serializable_config = _serializable_config(config)
        if config.save_last_checkpoint:
            save_checkpoint(
                build_checkpoint(
                    epoch, model, optimizer, scheduler, scaler, best_val_top1,
                    serializable_config, config.seed, MODEL_NAME, config.num_classes,
                ),
                last_ckpt_path,
            )
        if config.save_best_checkpoint and is_best:
            save_checkpoint(
                build_checkpoint(
                    epoch, model, optimizer, scheduler, scaler, best_val_top1,
                    serializable_config, config.seed, MODEL_NAME, config.num_classes,
                ),
                best_ckpt_path,
            )

    total_training_seconds = time.perf_counter() - training_start

    if config.save_best_checkpoint and best_ckpt_path.exists():
        best_checkpoint = load_checkpoint(best_ckpt_path, map_location=device)
        restore_training_state(best_checkpoint, model)
        logger.info(
            "Reloaded best checkpoint (epoch %d, val_top1=%.4f) for final evaluation",
            best_checkpoint["epoch"] + 1,
            best_checkpoint["best_val_top1"],
        )

    final_metrics, predictions = evaluate(
        model, test_loader, criterion, device, use_amp=config.mixed_precision, return_predictions=True
    )
    targets_arr, preds_arr = predictions

    confusion = compute_confusion_matrix(targets_arr, preds_arr, num_classes=config.num_classes)
    class_acc = per_class_accuracy(confusion)
    macro_precision, macro_recall, macro_f1 = macro_precision_recall_f1(
        targets_arr, preds_arr, config.num_classes
    )
    balanced_acc = compute_balanced_accuracy(targets_arr, preds_arr)

    print(f"Final test loss: {final_metrics.loss:.4f}")
    print(f"Final test top-1: {final_metrics.top1:.4f}")
    print(f"Final test top-5: {final_metrics.top5:.4f}")

    if history:
        save_history_csv(history, output_dir / "history.csv")
        save_loss_curve(history, output_dir / "training_loss.png")
        save_top1_curve(history, output_dir / "top1_accuracy.png")
        save_lr_curve(history, output_dir / "learning_rate.png")

    save_confusion_matrix_plot(confusion, output_dir / "confusion_matrix.png")
    save_per_class_accuracy_csv(class_names, class_acc, output_dir / "per_class_accuracy.csv")

    classification_metrics = {
        "num_classes": config.num_classes,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
        "final_test_top1": final_metrics.top1,
        "final_test_top5": final_metrics.top5,
    }
    _save_json(classification_metrics, output_dir / "classification_metrics.json")

    summary = {
        "experiment_name": config.experiment_name,
        "model_name": MODEL_NAME,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "device": str(device),
        "gpu_name": gpu_name,
        "seed": config.seed,
        "validation_fraction": config.validation_fraction,
        "configured_epochs": config.epochs,
        "completed_epochs": start_epoch + len(history),
        # Model selection during training: best epoch by validation top-1
        # accuracy (validation split held out from the training set).
        "best_validation": {
            "epoch": best_epoch,
            "top1": best_val_top1,
        },
        # One-time evaluation on the official held-out test set, using the
        # best checkpoint reloaded above. Never used for model selection.
        "final_test": {
            "loss": final_metrics.loss,
            "top1": final_metrics.top1,
            "top5": final_metrics.top5,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
        },
        "total_training_seconds": total_training_seconds,
        "best_checkpoint_path": str(best_ckpt_path) if config.save_best_checkpoint else None,
        "last_checkpoint_path": str(last_ckpt_path) if config.save_last_checkpoint else None,
    }
    _save_json(summary, output_dir / "summary.json")

    logger.info("Experiment '%s' complete.", config.experiment_name)


if __name__ == "__main__":
    main()
