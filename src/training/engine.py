"""Reusable training and evaluation engine.

These functions operate purely on a model, dataloader, optimizer, and
device: they hold no reference to a particular experiment's configuration
object, output directory, or checkpoint/logging setup. This keeps them
reusable as-is by future federated-learning client training loops, which
will call the same ``train_one_epoch``/``evaluate`` functions locally on
each client.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, LRScheduler, SequentialLR
from torch.utils.data import DataLoader

from src.evaluation.metrics import topk_accuracy

logger = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    """Aggregate metrics computed over one training or evaluation epoch."""

    loss: float
    top1: float
    top5: float
    elapsed_seconds: float
    learning_rate: float | None = None


def resolve_device(device_config: str) -> torch.device:
    """Resolve a configured device string to a concrete ``torch.device``.

    Args:
        device_config: One of "auto", "cpu", or "cuda". "auto" selects CUDA
            when available, otherwise falls back to CPU.

    Returns:
        The resolved device.

    Raises:
        RuntimeError: If "cuda" is requested but no CUDA device is available.
        ValueError: If ``device_config`` is not a recognized value.
    """
    name = device_config.lower()
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' was requested but CUDA is not available.")
        return torch.device("cuda")
    if name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device '{device_config}'. Expected 'auto', 'cpu', or 'cuda'.")


def build_optimizer(
    model: nn.Module,
    optimizer_name: str,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
) -> Optimizer:
    """Build the optimizer specified by ``optimizer_name``.

    Args:
        model: The model whose parameters will be optimized.
        optimizer_name: "sgd" (SGD with Nesterov momentum when
            ``momentum > 0``) or "adamw".
        learning_rate: Base learning rate.
        momentum: SGD momentum (ignored for AdamW).
        weight_decay: Weight decay applied by the optimizer.

    Returns:
        The constructed optimizer.

    Raises:
        ValueError: If ``optimizer_name`` is not supported.
    """
    name = optimizer_name.lower()
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=momentum > 0,
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Expected 'sgd' or 'adamw'.")


def build_scheduler(
    optimizer: Optimizer,
    scheduler_name: str,
    epochs: int,
    warmup_epochs: int = 0,
) -> LRScheduler | None:
    """Build the learning-rate scheduler specified by ``scheduler_name``.

    Behavior:
        - "none": returns None. The caller should not call ``.step()`` on
          anything, and the optimizer's configured learning rate stays
          constant for every epoch.
        - "cosine": cosine annealing from the base learning rate down to
          (near) zero over ``epochs`` epochs. If ``warmup_epochs > 0``, a
          linear warmup from 10% to 100% of the base learning rate runs
          for the first ``warmup_epochs`` epochs, and cosine annealing
          governs the remaining ``epochs - warmup_epochs`` epochs.

    The returned scheduler is intended to be stepped exactly once per
    completed epoch (not per batch).

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        scheduler_name: "cosine" or "none".
        epochs: Total number of training epochs.
        warmup_epochs: Number of linear-warmup epochs before cosine
            annealing begins; 0 disables warmup.

    Returns:
        The scheduler, or None if ``scheduler_name == "none"``.

    Raises:
        ValueError: If ``scheduler_name`` is unsupported, or if
            ``warmup_epochs >= epochs``.
    """
    name = scheduler_name.lower()
    if name == "none":
        return None
    if name != "cosine":
        raise ValueError(f"Unsupported scheduler '{scheduler_name}'. Expected 'cosine' or 'none'.")

    if warmup_epochs > 0:
        if warmup_epochs >= epochs:
            raise ValueError(
                f"warmup_epochs ({warmup_epochs}) must be less than epochs ({epochs})."
            )
        warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
        cosine = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
        return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    return CosineAnnealingLR(optimizer, T_max=epochs)


def build_grad_scaler(device: torch.device, use_amp: bool) -> torch.amp.GradScaler:
    """Build a gradient scaler for mixed-precision training.

    The scaler is enabled only when ``use_amp`` is True and ``device`` is a
    CUDA device. GradScaler-based mixed precision is a CUDA-only technique;
    on CPU the scaler is constructed but disabled (a safe no-op).

    Args:
        device: The device training will run on.
        use_amp: Whether mixed precision was requested by the configuration.

    Returns:
        A ``torch.amp.GradScaler``, enabled only on CUDA with ``use_amp=True``.
    """
    enabled = use_amp and device.type == "cuda"
    return torch.amp.GradScaler(device=device.type, enabled=enabled)


def compute_proximal_term(model: nn.Module, global_params: list[torch.Tensor]) -> torch.Tensor:
    """Compute the FedProx proximal penalty's unscaled distance term.

    Returns ``sum_i ||w_i - g_i||^2`` over ``model.parameters()``
    (trainable parameters only -- never buffers such as BatchNorm
    ``running_mean``/``running_var``/``num_batches_tracked``, since
    ``nn.Module.parameters()`` by construction only ever iterates
    ``nn.Parameter``s, not registered buffers) paired positionally with
    ``global_params``, in the exact order ``model.parameters()`` iterates.

    This function has no dependency on ``mu`` at all -- it is a pure
    squared-distance computation; the caller (``train_one_epoch``)
    multiplies by ``mu / 2`` to form the actual FedProx objective term
    ``(mu / 2) * ||w - w_global||^2``. Kept as a standalone, importable
    function (rather than inlined) so it is directly unit-testable in
    isolation from a full training step.

    Args:
        model: The model whose current parameters (``w``) are compared
            against ``global_params``.
        global_params: The global reference parameters (``w_global`` --
            the global model at the start of the current communication
            round), one tensor per trainable parameter, in
            ``model.parameters()`` order and matching shape.

    Returns:
        A scalar tensor: the sum of squared elementwise differences
        across all paired parameter tensors.
    """
    return sum((p - g).pow(2).sum() for p, g in zip(model.parameters(), global_params))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    use_amp: bool = False,
    grad_clip_norm: float | None = None,
    log_interval: int = 100,
    proximal_mu: float = 0.0,
    global_params: list[torch.Tensor] | None = None,
) -> EpochMetrics:
    """Run one training epoch.

    Args:
        model: The model to train (switched to train mode internally).
        loader: Training data loader.
        optimizer: Optimizer used to update ``model``'s parameters.
        criterion: Loss function.
        device: Device to run computation on.
        scaler: Gradient scaler for mixed precision. Required (non-None)
            for ``use_amp=True`` to have any effect; ignored otherwise.
        use_amp: Whether to use automatic mixed precision. Automatically
            has no effect when ``device`` is CPU.
        grad_clip_norm: Optional max gradient norm for clipping; disabled
            when None.
        log_interval: Log progress every this many batches; disabled when
            ``log_interval <= 0``.
        proximal_mu: FedProx proximal-term coefficient (Sprint 7.1).
            ``0.0`` (the default) is a strict no-op -- the added code path
            below is skipped entirely, so centralized training
            (``src/train_centralized.py``, which never passes this
            argument) and any existing caller are behaviorally unchanged.
            Only takes effect when both ``proximal_mu > 0`` and
            ``global_params is not None``.
        global_params: The global reference parameters (``w_global``) the
            proximal term measures distance from -- see
            ``compute_proximal_term``. Required (non-None) for
            ``proximal_mu > 0`` to have any effect; ignored otherwise.

    Returns:
        Aggregated loss/top-1/top-5/elapsed-time/learning-rate for the
        epoch. When ``proximal_mu > 0`` and ``global_params`` is provided,
        the reported/logged ``loss`` (both the returned
        ``EpochMetrics.loss`` and the per-batch log line) **includes** the
        proximal penalty term -- it is the exact quantity being minimized
        and backpropagated, not the cross-entropy term alone. This is a
        deliberate, documented choice: comparing ``train_loss`` between a
        FedAvg run (``proximal_mu=0``) and a FedProx run (``proximal_mu>0``)
        in ``round_history.csv`` is therefore not apples-to-apples: only
        the cross-entropy component is directly comparable, and it is not
        reported separately. Validation loss (``evaluate()``, untouched by
        this change) is unaffected and remains directly comparable between
        FedAvg and FedProx runs.
    """
    model.train()
    start = time.perf_counter()

    autocast_enabled = use_amp and device.type == "cuda"
    apply_proximal_term = proximal_mu > 0 and global_params is not None

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
            outputs = model(images)
            loss = criterion(outputs, targets)

        # FedProx (Sprint 7.1): added after the autocast block, before
        # backward(), exactly as the objective in
        # research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md Section 5
        # specifies. Computed directly on `model.parameters()`, which are
        # always full-precision regardless of `autocast_enabled` (autocast
        # only affects intermediate activations produced by ops it wraps,
        # not raw parameter tensors read outside that context) -- no
        # additional mixed-precision handling is needed here.
        if apply_proximal_term:
            loss = loss + (proximal_mu / 2) * compute_proximal_term(model, global_params)

        if autocast_enabled and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        batch_size = targets.size(0)
        top1, top5 = topk_accuracy(outputs.detach(), targets, ks=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_samples += batch_size

        if log_interval > 0 and (batch_idx + 1) % log_interval == 0:
            logger.info(
                "  batch %d/%d - loss=%.4f top1=%.2f%%",
                batch_idx + 1,
                len(loader),
                loss.item(),
                top1 * 100,
            )

    elapsed = time.perf_counter() - start
    return EpochMetrics(
        loss=total_loss / total_samples,
        top1=total_top1 / total_samples,
        top5=total_top5 / total_samples,
        elapsed_seconds=elapsed,
        learning_rate=optimizer.param_groups[0]["lr"],
    )


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_amp: bool = False,
    return_predictions: bool = False,
) -> tuple[EpochMetrics, tuple[np.ndarray, np.ndarray] | None]:
    """Run one evaluation epoch with gradients disabled.

    Args:
        model: The model to evaluate (switched to eval mode internally).
        loader: Evaluation data loader.
        criterion: Loss function.
        device: Device to run computation on.
        use_amp: Whether to use autocast for the forward pass. Automatically
            has no effect on CPU.
        return_predictions: If True, also returns the full concatenated
            ``(targets, predictions)`` arrays for downstream metrics such as
            a confusion matrix. Disabled by default so routine per-epoch
            evaluation (called every epoch during training) does not retain
            a full prediction history.

    Returns:
        A tuple ``(metrics, predictions_or_none)``.
    """
    model.eval()
    start = time.perf_counter()

    autocast_enabled = use_amp and device.type == "cuda"

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_samples = 0

    all_preds: list[np.ndarray] | None = [] if return_predictions else None
    all_targets: list[np.ndarray] | None = [] if return_predictions else None

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                outputs = model(images)
                loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            top1, top5 = topk_accuracy(outputs, targets, ks=(1, 5))
            total_loss += loss.item() * batch_size
            total_top1 += top1 * batch_size
            total_top5 += top5 * batch_size
            total_samples += batch_size

            if return_predictions:
                all_preds.append(outputs.argmax(dim=1).cpu().numpy())
                all_targets.append(targets.cpu().numpy())

    elapsed = time.perf_counter() - start
    metrics = EpochMetrics(
        loss=total_loss / total_samples,
        top1=total_top1 / total_samples,
        top5=total_top5 / total_samples,
        elapsed_seconds=elapsed,
    )

    predictions = None
    if return_predictions:
        predictions = (np.concatenate(all_targets), np.concatenate(all_preds))

    return metrics, predictions
