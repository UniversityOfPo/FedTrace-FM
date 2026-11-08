"""Checkpoint save/load utilities for training runs.

Checkpoints are written atomically: the payload is saved to a temporary
file in the destination directory and then moved into place with
``os.replace``, so a crash or interrupt mid-save can never leave a
corrupted or half-written checkpoint at the destination path.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


def build_checkpoint(
    epoch: int,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    best_val_top1: float,
    config: dict[str, Any],
    seed: int,
    model_name: str,
    num_classes: int,
) -> dict[str, Any]:
    """Assemble a full checkpoint payload dictionary.

    Args:
        epoch: The last completed epoch index (caller's own convention;
            this module does not interpret it).
        model: The model whose ``state_dict`` should be saved.
        optimizer: The optimizer whose ``state_dict`` should be saved.
        scheduler: The LR scheduler, if any; its state is saved when present.
        scaler: The AMP gradient scaler, if any; its state is saved when present.
        best_val_top1: Best validation top-1 accuracy observed so far.
        config: The fully resolved training configuration as a plain,
            JSON/pickle-safe dictionary, kept for experiment provenance.
        seed: The random seed used for this run.
        model_name: A human-readable model identifier.
        num_classes: Number of output classes the model was built for.

    Returns:
        A dictionary suitable for ``torch.save``.
    """
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_top1": best_val_top1,
        "config": config,
        "seed": seed,
        "model_name": model_name,
        "num_classes": num_classes,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }


def save_checkpoint(payload: dict[str, Any], path: Path) -> None:
    """Atomically save a checkpoint payload to ``path``.

    Args:
        payload: The checkpoint dictionary, typically from ``build_checkpoint``.
        path: Destination file path; parent directories are created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as tmp_file:
            torch.save(payload, tmp_file)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    logger.info("Saved checkpoint to %s", path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    """Load a checkpoint payload from disk.

    Args:
        path: Path to the checkpoint file.
        map_location: Device mapping passed to ``torch.load``.

    Returns:
        The checkpoint payload dictionary.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False: our checkpoints intentionally carry non-tensor
    # provenance (a plain config dict, seed, version strings) alongside
    # tensor state, and are always produced locally by this codebase, not
    # loaded from an untrusted source.
    return torch.load(path, map_location=map_location, weights_only=False)


def restore_training_state(
    checkpoint: dict[str, Any],
    model: nn.Module,
    optimizer: Optimizer | None = None,
    scheduler: LRScheduler | None = None,
    scaler: torch.amp.GradScaler | None = None,
) -> None:
    """Restore model/optimizer/scheduler/scaler state from a checkpoint payload.

    Optimizer, scheduler, and scaler state are restored only when both the
    checkpoint contains that state and the corresponding object was passed
    in, so this function can also be used for inference-only loads (model
    weights only).

    Args:
        checkpoint: A checkpoint payload, as returned by ``load_checkpoint``.
        model: Model to load ``model_state_dict`` into.
        optimizer: Optional optimizer to restore ``optimizer_state_dict`` into.
        scheduler: Optional LR scheduler to restore ``scheduler_state_dict`` into.
        scaler: Optional AMP gradient scaler to restore ``scaler_state_dict`` into.
    """
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
