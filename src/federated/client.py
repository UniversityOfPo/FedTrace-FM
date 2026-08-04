"""Reusable federated learning client abstraction (Sprint 3.1).

Scope of this module: a single ``FederatedClient`` that manages purely
local computation — its own dataset, dataloaders, model, optimizer,
scheduler, and local checkpoint. It internally reuses the existing,
experiment-agnostic training engine (`src/training/engine.py`) and
checkpoint utilities (`src/utils/checkpoint.py`) rather than duplicating
training or I/O logic.

This module deliberately does **not** know about:
    - FedAvg or any other aggregation strategy.
    - A federated Server or orchestrator.
    - Communication rounds between multiple clients.
    - Attacks or attribution.
    - How training data is partitioned across clients.

Those concerns belong to a future Server/orchestration module (Sprint 3.2
and later) that will construct one ``FederatedClient`` per participant,
call ``load_global_weights`` at the start of a round, call
``train_one_round`` to run local training, and call ``get_model_weights``
to collect the result for aggregation. This client only ever acts on
itself; it has no reference to any other client or to a global model
beyond the state dict it is handed.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.training import engine
from src.utils import checkpoint as checkpoint_utils

logger = logging.getLogger(__name__)

_VALID_OPTIMIZERS: tuple[str, ...] = ("sgd", "adamw")
_VALID_SCHEDULERS: tuple[str, ...] = ("cosine", "none")


@dataclass
class FederatedClientConfig:
    """Local (client-side) training configuration.

    This is intentionally separate from the centralized
    ``src.utils.config.TrainingConfig``: it holds only what a single
    client needs to run local training, with no reference to a dataset
    YAML path, an experiment output directory, or any multi-experiment
    orchestration concerns.

    Attributes:
        client_id: Unique identifier for this client.
        local_epochs: Number of local training epochs run per call to
            ``train_one_round()``.
        batch_size: Batch size used for this client's DataLoaders.
        num_workers: DataLoader worker process count (use 0 on Windows to
            avoid multiprocessing/spawn complications, e.g. in tests).
        pin_memory: Whether to use pinned host memory in DataLoaders.
        optimizer: "sgd" or "adamw".
        learning_rate: Base learning rate.
        momentum: SGD momentum (ignored for AdamW).
        weight_decay: Weight decay applied by the optimizer.
        scheduler: "cosine" or "none" (see
            ``src.training.engine.build_scheduler``).
        label_smoothing: Label smoothing factor for the loss function.
        mixed_precision: Whether to use automatic mixed precision;
            automatically has no effect on CPU.
        gradient_clip_norm: Optional max gradient norm for clipping.
        device: "auto", "cpu", or "cuda".
        seed: Reserved for future use by a client-side seeding step; not
            applied automatically by this class (see class docstring TODO).
        log_interval: Batch logging interval passed to
            ``train_one_epoch``; 0 disables per-batch logging.
    """

    client_id: str
    local_epochs: int = 1
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    optimizer: str = "sgd"
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0
    scheduler: str = "none"
    label_smoothing: float = 0.0
    mixed_precision: bool = False
    gradient_clip_norm: float | None = None
    device: str = "cpu"
    seed: int = 42
    log_interval: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values, raising ``ValueError`` with a clear message."""
        if not self.client_id:
            raise ValueError("client_id must be a non-empty string.")
        if self.local_epochs <= 0:
            raise ValueError(f"local_epochs must be a positive integer, got {self.local_epochs}.")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {self.batch_size}.")
        if self.num_workers < 0:
            raise ValueError(f"num_workers must be zero or positive, got {self.num_workers}.")
        if self.optimizer.lower() not in _VALID_OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {_VALID_OPTIMIZERS}, got '{self.optimizer}'.")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}.")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError(f"momentum must be in [0, 1), got {self.momentum}.")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be zero or positive, got {self.weight_decay}.")
        if self.scheduler.lower() not in _VALID_SCHEDULERS:
            raise ValueError(f"scheduler must be one of {_VALID_SCHEDULERS}, got '{self.scheduler}'.")
        if not (0.0 <= self.label_smoothing < 1.0):
            raise ValueError(f"label_smoothing must be in [0, 1), got {self.label_smoothing}.")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be positive when set, got {self.gradient_clip_norm}."
            )
        if self.log_interval < 0:
            raise ValueError(f"log_interval must be zero or positive, got {self.log_interval}.")


class FederatedClient:
    """Manages one federated participant's purely local computation.

    A ``FederatedClient`` holds its own dataset, dataloaders, model,
    optimizer, scheduler, and local configuration. It reuses
    ``src.training.engine`` for the actual training/evaluation loops, so
    any future improvement to that engine benefits this client
    automatically, with no duplicated training logic.

    The model is not built at construction time — call
    ``initialize_model()`` explicitly first. This separates "a client that
    knows its data and config" from "a client with a live model in memory",
    which keeps client construction cheap even before a model is needed.
    """

    def __init__(
        self,
        client_id: str,
        model_fn: Callable[[], nn.Module],
        train_dataset: Dataset,
        config: FederatedClientConfig,
        val_dataset: Dataset | None = None,
    ) -> None:
        """Construct a federated client around a local dataset and config.

        Args:
            client_id: Unique identifier for this client. Must match
                ``config.client_id``.
            model_fn: A zero-argument callable that returns a fresh,
                uninitialized model instance (e.g.
                ``lambda: resnet18_cifar(num_classes=100)``). Kept as a
                factory (rather than a model instance) so
                ``initialize_model()`` can (re)build a fresh model on demand
                without this class needing to know how to clone a model.
            train_dataset: This client's local training dataset. This class
                does not partition, download, or otherwise inspect the
                dataset's origin — it is handed a ready-to-use dataset.
            config: Local training configuration for this client.
            val_dataset: Optional local validation/evaluation dataset. If
                omitted, ``evaluate()`` raises a clear error when called.

        Raises:
            ValueError: If ``client_id`` does not match ``config.client_id``.
        """
        if client_id != config.client_id:
            raise ValueError(
                f"client_id ('{client_id}') does not match config.client_id "
                f"('{config.client_id}')."
            )

        self.client_id = client_id
        self.model_fn = model_fn
        self.config = config

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        self.train_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        self.val_loader: DataLoader | None = (
            DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=config.pin_memory,
            )
            if val_dataset is not None
            else None
        )

        self.device: torch.device | None = None
        self.model: nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.criterion: nn.Module | None = None

        self.current_round: int = 0
        self.round_history: list[list[engine.EpochMetrics]] = []
        self.last_eval_metrics: engine.EpochMetrics | None = None

    def _ensure_model_initialized(self) -> None:
        """Raise a clear error if ``initialize_model()`` has not been called yet."""
        if self.model is None:
            raise RuntimeError(
                f"Client '{self.client_id}' has no model yet. Call "
                "initialize_model() before training, evaluating, or "
                "accessing model weights."
            )

    def initialize_model(self) -> None:
        """Build this client's local model, optimizer, scheduler, and AMP scaler.

        Calling this again rebuilds a fresh model (via ``model_fn``) and a
        fresh optimizer/scheduler/scaler; it does not reset
        ``current_round`` or ``round_history``, since those track this
        client's participation over time, independent of any particular
        model instance held in memory.

        # TODO (Sprint 3.2): the scheduler here is scoped to
        # `config.local_epochs` (one round's worth of epochs). Whether a
        # client's scheduler should persist state across communication
        # rounds, reset every round, or be scoped to the total number of
        # rounds is a Server/orchestration-level decision that has not been
        # made yet. Revisit once round orchestration is designed.
        """
        self.device = engine.resolve_device(self.config.device)
        self.model = self.model_fn().to(self.device)
        self.optimizer = engine.build_optimizer(
            self.model,
            self.config.optimizer,
            self.config.learning_rate,
            self.config.momentum,
            self.config.weight_decay,
        )
        self.scheduler = engine.build_scheduler(
            self.optimizer,
            self.config.scheduler,
            epochs=self.config.local_epochs,
        )
        self.scaler = engine.build_grad_scaler(self.device, self.config.mixed_precision)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=self.config.label_smoothing)
        logger.info(
            "Client '%s': model initialized on %s.", self.client_id, self.device
        )

    def load_global_weights(self, state_dict: dict[str, torch.Tensor]) -> None:
        """Load a global model state dict into this client's local model.

        This is the integration point a Server/aggregator calls at the
        start of each communication round to synchronize this client with
        the current global model. This client has no knowledge of how
        ``state_dict`` was produced (e.g., via FedAvg) — it only knows how
        to load a state dict into its own model.

        Client-state policy (resolved Sprint 4.1, was a Sprint 3.2 TODO):
        loading new global weights also resets this client's optimizer,
        scheduler, and AMP scaler via ``reset_training_state()`` — no
        momentum or scheduler state carries over from a previous round.
        This is the baseline FedAvg client-state policy: it keeps
        communication rounds independent and avoids stale momentum
        computed against a now-superseded global model.

        # TODO (Sprint 4.2): this reset is currently unconditional. A
        # future strategy that intentionally wants persistent client-side
        # optimizer state across rounds (e.g. a FedProx/SCAFFOLD-style
        # variant) would need this made configurable (e.g. a
        # ``reset_state: bool = True`` parameter) rather than always-on.
        # Deferred rather than guessed, since no such strategy exists yet.

        Args:
            state_dict: A model state dict, structurally compatible with
                this client's model.
        """
        self._ensure_model_initialized()
        self.model.load_state_dict(state_dict)
        self.reset_training_state()
        logger.info("Client '%s': loaded global weights.", self.client_id)

    def reset_training_state(self) -> None:
        """Rebuild this client's optimizer, scheduler, and AMP scaler from scratch.

        Discards any accumulated momentum/scheduler state while leaving
        the model instance and its current weights untouched. Uses the
        same ``config`` used by ``initialize_model()``, so the rebuilt
        optimizer/scheduler/scaler are identical in kind to the originals
        — only their internal state (e.g. SGD momentum buffers, scheduler
        step count) is cleared.

        This is the smallest safe mechanism for a "reset optimizer state
        every round" policy: it does not touch the dataset, dataloaders,
        or model architecture, and it does not change any existing public
        method's signature.
        """
        self._ensure_model_initialized()
        self.optimizer = engine.build_optimizer(
            self.model,
            self.config.optimizer,
            self.config.learning_rate,
            self.config.momentum,
            self.config.weight_decay,
        )
        self.scheduler = engine.build_scheduler(
            self.optimizer,
            self.config.scheduler,
            epochs=self.config.local_epochs,
        )
        self.scaler = engine.build_grad_scaler(self.device, self.config.mixed_precision)
        logger.info("Client '%s': training state (optimizer/scheduler/scaler) reset.", self.client_id)

    def get_model_weights(self) -> dict[str, torch.Tensor]:
        """Return a CPU copy of this client's current local model state dict.

        Returned on CPU (detached, cloned) so a caller collecting weights
        from multiple clients (e.g., a future Server, for aggregation)
        never holds a live reference into this client's GPU memory.

        Returns:
            A plain ``state_dict``-shaped mapping of parameter/buffer name
            to CPU tensor.
        """
        self._ensure_model_initialized()
        return {key: value.detach().cpu().clone() for key, value in self.model.state_dict().items()}

    def train_one_round(self) -> list[engine.EpochMetrics]:
        """Run ``config.local_epochs`` epochs of local training.

        Reuses ``src.training.engine.train_one_epoch`` directly, with no
        federated-specific training logic of its own. This method assumes
        it is called once per communication round by an external caller;
        it has no concept of "round" beyond incrementing its own counter.

        Returns:
            One ``EpochMetrics`` per local epoch run this round.
        """
        self._ensure_model_initialized()

        epoch_metrics: list[engine.EpochMetrics] = []
        for _ in range(self.config.local_epochs):
            metrics = engine.train_one_epoch(
                self.model,
                self.train_loader,
                self.optimizer,
                self.criterion,
                self.device,
                scaler=self.scaler,
                use_amp=self.config.mixed_precision,
                grad_clip_norm=self.config.gradient_clip_norm,
                log_interval=self.config.log_interval,
            )
            epoch_metrics.append(metrics)
            if self.scheduler is not None:
                self.scheduler.step()

        self.round_history.append(epoch_metrics)
        self.current_round += 1
        logger.info(
            "Client '%s': completed round %d (%d local epoch(s)).",
            self.client_id,
            self.current_round,
            len(epoch_metrics),
        )
        return epoch_metrics

    def evaluate(self) -> engine.EpochMetrics:
        """Evaluate the local model on this client's validation dataset.

        Reuses ``src.training.engine.evaluate`` directly.

        Returns:
            Aggregate loss/top-1/top-5 metrics for this client's
            validation data.

        Raises:
            RuntimeError: If this client was constructed without a
                ``val_dataset``.
        """
        self._ensure_model_initialized()
        if self.val_loader is None:
            raise RuntimeError(
                f"Client '{self.client_id}' has no validation dataset configured; "
                "pass val_dataset to the constructor to enable evaluate()."
            )

        metrics, _ = engine.evaluate(
            self.model,
            self.val_loader,
            self.criterion,
            self.device,
            use_amp=self.config.mixed_precision,
        )
        self.last_eval_metrics = metrics
        return metrics

    def _serialize_metrics(self) -> dict[str, Any]:
        """Convert accumulated round history and last eval metrics into a plain dict."""
        return {
            "round_history": [
                [asdict(epoch_metrics) for epoch_metrics in round_epochs]
                for round_epochs in self.round_history
            ],
            "last_eval_metrics": (
                asdict(self.last_eval_metrics) if self.last_eval_metrics is not None else None
            ),
        }

    def save_local_checkpoint(self, path: Path) -> None:
        """Atomically save this client's local state to ``path``.

        The checkpoint includes the client id, current round id, model
        state, optimizer state, scheduler state (if any), and this
        client's accumulated local metrics. Reuses
        ``src.utils.checkpoint.save_checkpoint`` for the actual atomic
        write, so this does not duplicate that write-then-replace logic.

        Args:
            path: Destination file path; parent directories are created
                if needed.
        """
        self._ensure_model_initialized()
        payload: dict[str, Any] = {
            "client_id": self.client_id,
            "round_id": self.current_round,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "local_metrics": self._serialize_metrics(),
        }
        checkpoint_utils.save_checkpoint(payload, path)

    def load_local_checkpoint(self, path: Path) -> None:
        """Load a previously saved local checkpoint from ``path``.

        Restores model/optimizer/scheduler state (reusing
        ``src.utils.checkpoint.restore_training_state``) and this client's
        round id. Local metric history is not reconstructed into
        ``round_history``/``last_eval_metrics`` (those remain in-memory
        bookkeeping for the current process); the raw serialized metrics
        are returned unused inside the checkpoint payload for external
        inspection if needed.

        Args:
            path: Path to a checkpoint previously written by
                ``save_local_checkpoint``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the checkpoint's stored ``client_id`` does not
                match this client's id.
        """
        self._ensure_model_initialized()
        payload = checkpoint_utils.load_checkpoint(path, map_location=self.device)

        stored_client_id = payload.get("client_id")
        if stored_client_id != self.client_id:
            raise ValueError(
                f"Checkpoint at {path} belongs to client '{stored_client_id}', "
                f"not '{self.client_id}'."
            )

        checkpoint_utils.restore_training_state(
            payload, self.model, self.optimizer, self.scheduler
        )
        self.current_round = payload.get("round_id", self.current_round)
        logger.info(
            "Client '%s': loaded local checkpoint from %s (round %d).",
            self.client_id,
            path,
            self.current_round,
        )

    def get_training_summary(self) -> dict[str, Any]:
        """Return a plain-dict summary of this client's current local state.

        Intended for lightweight logging/inspection by an external caller
        (e.g., a future Server) without needing to know this class's
        internal attributes.

        Returns:
            A dictionary with the client id, round progress, dataset
            sizes, and the most recent training/evaluation metrics
            (``None`` where not yet available).
        """
        latest_epoch_metrics = self.round_history[-1][-1] if self.round_history else None
        return {
            "client_id": self.client_id,
            "current_round": self.current_round,
            "rounds_completed": len(self.round_history),
            "num_train_samples": len(self.train_dataset),
            "num_val_samples": len(self.val_dataset) if self.val_dataset is not None else None,
            "latest_train_loss": latest_epoch_metrics.loss if latest_epoch_metrics else None,
            "latest_train_top1": latest_epoch_metrics.top1 if latest_epoch_metrics else None,
            "latest_eval_loss": self.last_eval_metrics.loss if self.last_eval_metrics else None,
            "latest_eval_top1": self.last_eval_metrics.top1 if self.last_eval_metrics else None,
        }
