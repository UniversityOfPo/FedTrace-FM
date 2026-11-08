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
import math
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
_VALID_ROUND_LR_SCHEDULES: tuple[str, ...] = ("constant", "cosine")
_VALID_LOCAL_TRAINING_METHODS: tuple[str, ...] = ("fedavg", "fedprox")


def compute_round_learning_rate(
    round_id: int,
    total_rounds: int,
    base_lr: float,
    min_lr: float,
    schedule: str,
) -> float:
    """Deterministically compute the learning rate to use for one communication round.

    This is a pure, closed-form function of the round index only — it
    holds no ``torch.optim.lr_scheduler`` state and needs none, unlike the
    existing local ``scheduler`` (see ``FederatedClientConfig.scheduler``,
    ``src.training.engine.build_scheduler``), which is scoped to
    ``local_epochs`` and rebuilt from scratch every round by
    ``reset_training_state()``. With ``local_epochs=1``, a round-scoped
    ``LRScheduler`` never gets to span more than one round before it is
    discarded, so it cannot meaningfully anneal across a whole multi-round
    run. This function sidesteps that entirely by computing the round's
    learning rate directly from the round index, independent of any
    scheduler object's internal step count.

    Behavior:
        - ``"constant"``: always returns ``base_lr``, regardless of
          ``round_id`` or ``total_rounds``.
        - ``"cosine"``: cosine-anneals from ``base_lr`` at round 0 down to
          exactly ``min_lr`` at round ``total_rounds - 1``, using the
          standard closed-form cosine-annealing formula. ``round_id``
          values at or beyond ``total_rounds - 1`` are clamped to
          ``total_rounds - 1`` (i.e. the schedule holds at ``min_lr``
          rather than extrapolating past it).

    Args:
        round_id: The communication round to compute the learning rate
            for (0-indexed).
        total_rounds: Total number of communication rounds the schedule
            spans. Only consulted when ``schedule == "cosine"``.
        base_lr: The learning rate at round 0.
        min_lr: The learning rate at the final round. Only consulted when
            ``schedule == "cosine"``.
        schedule: ``"constant"`` or ``"cosine"``.

    Returns:
        The learning rate to use for ``round_id``.

    Raises:
        ValueError: If ``round_id`` is negative, if ``schedule`` is not
            ``"constant"``/``"cosine"``, or if ``total_rounds`` is not
            positive when ``schedule == "cosine"``.
    """
    if round_id < 0:
        raise ValueError(f"round_id must be >= 0, got {round_id}.")

    name = schedule.lower()
    if name == "constant":
        return base_lr
    if name != "cosine":
        raise ValueError(
            f"Unsupported round_lr_schedule '{schedule}'. Expected 'constant' or 'cosine'."
        )
    if total_rounds <= 0:
        raise ValueError(
            f"total_rounds must be a positive integer for a cosine round schedule, got {total_rounds}."
        )

    if total_rounds == 1:
        return base_lr

    progress = min(round_id, total_rounds - 1) / (total_rounds - 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


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
            ``src.training.engine.build_scheduler``). This is the existing
            *local*, epoch-scoped scheduler — rebuilt from scratch every
            communication round and scoped to ``local_epochs``. It is
            independent of ``round_lr_schedule`` below.
        label_smoothing: Label smoothing factor for the loss function.
        mixed_precision: Whether to use automatic mixed precision;
            automatically has no effect on CPU.
        gradient_clip_norm: Optional max gradient norm for clipping.
        device: "auto", "cpu", or "cuda".
        seed: Reserved for future use by a client-side seeding step; not
            applied automatically by this class (see class docstring TODO).
        log_interval: Batch logging interval passed to
            ``train_one_epoch``; 0 disables per-batch logging.
        round_lr_schedule: "constant" (default) or "cosine" (Sprint 5.1).
            An explicit *round-level* learning-rate policy, independent of
            ``scheduler``: it is applied by ``reset_training_state()``/
            ``load_global_weights()`` whenever a ``round_id`` is provided,
            via a deterministic closed-form calculation
            (``compute_round_learning_rate``) rather than any
            ``torch.optim.lr_scheduler`` state. This is what allows a
            cosine anneal to meaningfully span many communication rounds
            even when ``local_epochs=1`` and optimizer state resets every
            round — conditions under which ``scheduler="cosine"`` alone
            cannot vary the learning rate across rounds. "constant" (the
            default) is a no-op relative to the pre-Sprint-5.1 behavior:
            the learning rate always stays at ``learning_rate``.
        min_learning_rate: The learning rate at the final communication
            round when ``round_lr_schedule="cosine"``. Ignored for
            "constant". Must be ``>= 0`` and ``<= learning_rate``.
        total_communication_rounds: Total number of communication rounds
            the round-level schedule spans. Required (must be a positive
            integer) when ``round_lr_schedule="cosine"``; unused and may
            be left ``None`` for "constant".
        local_training_method: "fedavg" (default) or "fedprox" (Sprint
            7.1; see ``research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md``).
            "fedavg" is a strict no-op relative to pre-Sprint-7.1
            behavior: no global-parameter snapshot is taken and no
            proximal term is ever applied, **regardless of
            ``fedprox_mu``** -- ``fedprox_mu`` only has any effect when
            this is ``"fedprox"``.
        fedprox_mu: FedProx proximal-term coefficient. Must be ``>= 0``;
            must be ``> 0`` when ``local_training_method="fedprox"``
            (a zero-strength proximal term is indistinguishable from
            plain FedAvg and is rejected as a likely configuration
            mistake rather than silently accepted). Ignored (inert) when
            ``local_training_method="fedavg"``.
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
    round_lr_schedule: str = "constant"
    min_learning_rate: float = 0.0
    total_communication_rounds: int | None = None
    local_training_method: str = "fedavg"
    fedprox_mu: float = 0.0

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
        if self.round_lr_schedule.lower() not in _VALID_ROUND_LR_SCHEDULES:
            raise ValueError(
                f"round_lr_schedule must be one of {_VALID_ROUND_LR_SCHEDULES}, "
                f"got '{self.round_lr_schedule}'."
            )
        if self.min_learning_rate < 0:
            raise ValueError(
                f"min_learning_rate must be zero or positive, got {self.min_learning_rate}."
            )
        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                f"min_learning_rate ({self.min_learning_rate}) must be <= learning_rate "
                f"({self.learning_rate})."
            )
        if self.round_lr_schedule.lower() == "cosine" and (
            self.total_communication_rounds is None or self.total_communication_rounds <= 0
        ):
            raise ValueError(
                "total_communication_rounds must be a positive integer when "
                f"round_lr_schedule='cosine', got {self.total_communication_rounds}."
            )
        if self.local_training_method.lower() not in _VALID_LOCAL_TRAINING_METHODS:
            raise ValueError(
                f"local_training_method must be one of {_VALID_LOCAL_TRAINING_METHODS}, "
                f"got '{self.local_training_method}'."
            )
        if self.fedprox_mu < 0:
            raise ValueError(f"fedprox_mu must be zero or positive, got {self.fedprox_mu}.")
        if self.local_training_method.lower() == "fedprox" and self.fedprox_mu <= 0:
            raise ValueError(
                "fedprox_mu must be > 0 when local_training_method='fedprox', "
                f"got {self.fedprox_mu}."
            )


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

        # FedProx (Sprint 7.1): the global model's trainable-parameter
        # values at the start of the current round ("w_global" in
        # research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md Section 5),
        # captured by load_global_weights() and refreshed every round.
        # None whenever config.local_training_method != "fedprox" (see
        # load_global_weights()), so no snapshot is ever taken -- and no
        # memory/compute overhead incurred -- for plain FedAvg clients.
        self._global_reference_params: list[torch.Tensor] | None = None

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

        # NOTE (Sprint 3.2 TODO, resolved by Sprint 5.1): the scheduler
        # built here (`self.scheduler`) is scoped to `config.local_epochs`
        # (one round's worth of epochs) and rebuilt from scratch every
        # round by `reset_training_state()` — with `local_epochs=1`, it
        # cannot vary the learning rate across rounds at all. Sprint 5.1
        # added an independent, round-level learning-rate mechanism
        # (`config.round_lr_schedule`, `compute_round_learning_rate`,
        # applied via `reset_training_state(round_id=...)`) specifically to
        # allow a schedule to span the full multi-round run without relying
        # on any scheduler object's state persisting across rounds.
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

    def load_global_weights(
        self, state_dict: dict[str, torch.Tensor], round_id: int | None = None
    ) -> None:
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
            round_id: The communication round about to run, forwarded to
                ``reset_training_state()`` to apply this round's
                round-level learning rate (Sprint 5.1). ``None`` (the
                default) preserves the pre-Sprint-5.1 behavior exactly —
                no round-level learning-rate override is applied.
        """
        self._ensure_model_initialized()
        self.model.load_state_dict(state_dict)

        # FedProx (Sprint 7.1): snapshot the just-loaded global weights as
        # this round's proximal reference point, before any local training
        # runs. Detached clones of `model.parameters()` only (trainable
        # parameters, never buffers like BatchNorm running stats/
        # `num_batches_tracked`) -- a true independent copy, not a live
        # reference, so it cannot change as local SGD subsequently updates
        # `self.model`. Retaken on every call (i.e. every communication
        # round), so it never goes stale. `None` for "fedavg" (the
        # default) -- no snapshot is taken, matching pre-Sprint-7.1
        # behavior exactly.
        if self.config.local_training_method.lower() == "fedprox":
            self._global_reference_params = [p.detach().clone() for p in self.model.parameters()]
        else:
            self._global_reference_params = None

        self.reset_training_state(round_id=round_id)
        logger.info("Client '%s': loaded global weights.", self.client_id)

    def reset_training_state(self, round_id: int | None = None) -> None:
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
        method's signature beyond the new optional ``round_id`` parameter.

        Args:
            round_id: If provided, this round's round-level learning rate
                (Sprint 5.1; see ``FederatedClientConfig.round_lr_schedule``
                and ``compute_round_learning_rate``) is computed and
                applied to the freshly-rebuilt optimizer immediately after
                this method finishes rebuilding it — i.e. before any local
                training for that round begins. ``None`` (the default)
                skips this entirely, leaving the optimizer at
                ``config.learning_rate`` exactly as before Sprint 5.1.
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
        if round_id is not None:
            self._apply_round_learning_rate(round_id)
        logger.info("Client '%s': training state (optimizer/scheduler/scaler) reset.", self.client_id)

    def _apply_round_learning_rate(self, round_id: int) -> float:
        """Compute and apply this round's learning rate, before local training begins.

        Overrides the optimizer's learning rate (every param group) with
        the value ``compute_round_learning_rate`` computes for
        ``round_id``, using this client's ``round_lr_schedule``,
        ``learning_rate`` (base), ``min_learning_rate``, and
        ``total_communication_rounds``. Independent of ``self.scheduler``
        (the local, epoch-scoped scheduler) — both may be configured
        simultaneously; this method only sets the *starting* learning rate
        for the round that ``self.scheduler`` (if any) then varies within.

        Args:
            round_id: The communication round about to run.

        Returns:
            The learning rate that was applied (for logging/testing).
        """
        self._ensure_model_initialized()
        round_lr = compute_round_learning_rate(
            round_id=round_id,
            total_rounds=self.config.total_communication_rounds or 1,
            base_lr=self.config.learning_rate,
            min_lr=self.config.min_learning_rate,
            schedule=self.config.round_lr_schedule,
        )
        for group in self.optimizer.param_groups:
            group["lr"] = round_lr
        logger.info(
            "Client '%s': round %d learning rate set to %.6g (schedule='%s').",
            self.client_id,
            round_id,
            round_lr,
            self.config.round_lr_schedule,
        )
        return round_lr

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

        When ``config.local_training_method == "fedprox"``, every local
        epoch this round is trained against the same frozen global
        reference snapshot captured by the most recent
        ``load_global_weights()`` call (Sprint 7.1) — see
        ``research_plans/EXP_010_FedProx_Strong_NonIID_Plan.md`` Section 7.

        Returns:
            One ``EpochMetrics`` per local epoch run this round.

        Raises:
            RuntimeError: If ``config.local_training_method == "fedprox"``
                but no global reference snapshot exists yet (i.e.
                ``load_global_weights()`` was never called) — training
                with FedProx before a snapshot exists would silently
                behave like plain FedAvg instead of raising, which is
                worse than a clear error.
        """
        self._ensure_model_initialized()

        use_fedprox = self.config.local_training_method.lower() == "fedprox"
        if use_fedprox and self._global_reference_params is None:
            raise RuntimeError(
                f"Client '{self.client_id}' is configured for local_training_method='fedprox' "
                "but has no global reference snapshot. Call load_global_weights() before "
                "train_one_round()."
            )
        proximal_mu = self.config.fedprox_mu if use_fedprox else 0.0
        global_params = self._global_reference_params if use_fedprox else None

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
                proximal_mu=proximal_mu,
                global_params=global_params,
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
