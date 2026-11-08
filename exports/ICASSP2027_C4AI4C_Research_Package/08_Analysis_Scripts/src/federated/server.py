"""Reusable federated server abstraction (Sprint 3.2).

Scope of this module: a ``FederatedServer`` that owns the global model,
registered-client metadata, and per-round bookkeeping, and defines the
client-server update contract (via ``src.federated.contracts``). It
deliberately does **not** implement any aggregation strategy (e.g.
FedAvg): ``receive_update`` only validates and stores a ``ClientUpdate``,
and never mutates the global model. A future Sprint 3.3+ aggregation
module is expected to read ``list_received_updates()`` and
``load_global_weights()`` to actually combine client updates.

This module also does not:
    - Run communication rounds or any round-orchestration loop.
    - Train any client (it never calls ``FederatedClient.train_one_round``).
    - Implement client dropout simulation, secure aggregation, or network
      communication.
    - Know about attacks, attribution, or provenance.

The server never holds ``FederatedClient`` instances — only plain
metadata (sample counts, arbitrary metadata) supplied at registration
time. It is the caller's responsibility to run clients and hand their
resulting ``ClientUpdate`` objects to ``receive_update``.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from src.federated.contracts import ClientUpdate, RoundRecord, utc_now_iso
from src.utils import checkpoint as checkpoint_utils

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FederatedServerConfig:
    """Server-side configuration.

    Attributes:
        server_id: Unique identifier for this server instance.
        seed: Base random seed for deterministic client selection.
        clients_per_round: Number of clients to select per round.
        minimum_updates: Minimum accepted updates required before a round
            can be finalized as "completed"; must be <= clients_per_round.
        checkpoint_dir: Default directory server checkpoints are written
            to/read from.
        strict_round_matching: If True, ``receive_update`` rejects updates
            whose ``round_id`` does not exactly match the server's active
            round id.
    """

    server_id: str
    seed: int
    clients_per_round: int
    minimum_updates: int
    checkpoint_dir: Path
    strict_round_matching: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values and coerce ``checkpoint_dir`` to a ``Path``."""
        if not self.server_id:
            raise ValueError("server_id must be a non-empty string.")
        if self.seed < 0:
            raise ValueError(f"seed must be >= 0, got {self.seed}.")
        if self.clients_per_round <= 0:
            raise ValueError(f"clients_per_round must be > 0, got {self.clients_per_round}.")
        if self.minimum_updates <= 0:
            raise ValueError(f"minimum_updates must be > 0, got {self.minimum_updates}.")
        if self.minimum_updates > self.clients_per_round:
            raise ValueError(
                f"minimum_updates ({self.minimum_updates}) must be <= "
                f"clients_per_round ({self.clients_per_round})."
            )
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))


def _serializable_config(config: FederatedServerConfig) -> dict[str, Any]:
    """Convert a ``FederatedServerConfig`` into a plain, pickle-safe dict (Path -> str)."""
    raw = asdict(config)
    return {key: (str(value) if isinstance(value, Path) else value) for key, value in raw.items()}


def _round_record_to_dict(record: RoundRecord) -> dict[str, Any]:
    """Convert a ``RoundRecord`` into a plain dict for checkpoint storage."""
    return {
        "round_id": record.round_id,
        "selected_client_ids": list(record.selected_client_ids),
        "received_client_ids": list(record.received_client_ids),
        "rejected_client_ids": list(record.rejected_client_ids),
        "status": record.status,
        "started_at_utc": record.started_at_utc,
        "completed_at_utc": record.completed_at_utc,
        "metadata": dict(record.metadata),
    }


def _round_record_from_dict(payload: dict[str, Any]) -> RoundRecord:
    """Reconstruct a ``RoundRecord`` from a plain dict previously produced by
    ``_round_record_to_dict``."""
    return RoundRecord(
        round_id=payload["round_id"],
        selected_client_ids=tuple(payload["selected_client_ids"]),
        received_client_ids=tuple(payload["received_client_ids"]),
        rejected_client_ids=tuple(payload["rejected_client_ids"]),
        status=payload["status"],
        started_at_utc=payload["started_at_utc"],
        metadata=payload["metadata"],
        completed_at_utc=payload.get("completed_at_utc"),
    )


class FederatedServer:
    """Owns the global model, registered-client metadata, and round bookkeeping.

    Deliberately independent of any aggregation strategy: this class never
    combines multiple ``ClientUpdate``s into new global weights. It only
    tracks who was selected, what was received, and whether a round can be
    considered complete, so a future aggregation module (Sprint 3.3+) can
    be built on top without this class needing to change.
    """

    def __init__(self, config: FederatedServerConfig, model_fn: Callable[[], nn.Module]) -> None:
        """Construct a server around a config and a global-model factory.

        Args:
            config: Validated server configuration.
            model_fn: A zero-argument callable that returns a fresh model
                instance. Called once here to build the initial global
                model (always held on CPU — the server never trains, so it
                has no need for GPU placement).
        """
        self.config = config
        self._model_fn = model_fn
        self._global_model: nn.Module = model_fn().cpu()

        self._registered_clients: dict[str, dict[str, Any]] = {}

        self._current_round_id: int | None = None
        self._selected_client_ids: tuple[str, ...] | None = None
        self._received_updates: dict[str, ClientUpdate] = {}
        self._rejected_client_ids: set[str] = set()
        self._round_started_at_utc: str | None = None

        self._round_history: list[RoundRecord] = []

        logger.info("Server '%s': initialized.", config.server_id)

    # ------------------------------------------------------------------
    # Client registration
    # ------------------------------------------------------------------

    def register_client(
        self, client_id: str, num_samples: int, metadata: Mapping[str, Any] | None = None
    ) -> None:
        """Register a client's metadata with this server.

        Only plain metadata is stored — never a ``FederatedClient``
        instance. This server has no reference to any client's dataset,
        model, or training state; it only knows a client id, a declared
        sample count, and arbitrary metadata.

        Args:
            client_id: Unique identifier for the client.
            num_samples: The client's local training sample count.
            metadata: Optional free-form metadata to store alongside the
                registration.

        Raises:
            ValueError: If ``client_id`` is empty, already registered, or
                ``num_samples`` is not positive.
        """
        if not client_id:
            raise ValueError("client_id must be a non-empty string.")
        if client_id in self._registered_clients:
            raise ValueError(f"Client '{client_id}' is already registered.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}.")

        self._registered_clients[client_id] = {
            "num_samples": num_samples,
            "metadata": dict(metadata) if metadata is not None else {},
            "registered_at_utc": utc_now_iso(),
        }
        logger.info(
            "Server '%s': registered client '%s' (%d samples).",
            self.config.server_id, client_id, num_samples,
        )

    def unregister_client(self, client_id: str) -> None:
        """Remove a client's registration.

        # TODO (Sprint 3.3): unregistering a client that is currently
        # selected for an active round is allowed today and has no special
        # effect on that round (the client remains in
        # `_selected_client_ids` for this round and could still submit a
        # valid update). Whether unregistration mid-round should instead
        # immediately reject/exclude that client is a round-orchestration
        # policy decision left for when the orchestration module is built.

        Args:
            client_id: The client id to remove.

        Raises:
            ValueError: If ``client_id`` is not currently registered.
        """
        if client_id not in self._registered_clients:
            raise ValueError(f"Client '{client_id}' is not registered.")
        del self._registered_clients[client_id]
        logger.info("Server '%s': unregistered client '%s'.", self.config.server_id, client_id)

    def list_registered_clients(self) -> tuple[str, ...]:
        """Return the sorted tuple of currently registered client ids."""
        return tuple(sorted(self._registered_clients.keys()))

    # ------------------------------------------------------------------
    # Global model weights
    # ------------------------------------------------------------------

    def get_global_weights(self) -> dict[str, torch.Tensor]:
        """Return a cloned, detached, CPU copy of the current global model state dict.

        Never returns a live reference into the server's internal model,
        so callers cannot accidentally mutate server state through the
        returned tensors.
        """
        return {
            key: value.detach().cpu().clone()
            for key, value in self._global_model.state_dict().items()
        }

    def load_global_weights(self, model_state: Mapping[str, torch.Tensor]) -> None:
        """Replace the global model's weights with ``model_state``.

        Args:
            model_state: A state dict structurally compatible (same keys
                and tensor shapes) with the server's global model.

        Raises:
            ValueError: If ``model_state``'s keys or tensor shapes are
                incompatible with the current global model.
        """
        current_state = self._global_model.state_dict()
        self._raise_if_incompatible(model_state, current_state, context="model_state")

        cloned = {key: value.detach().cpu().clone() for key, value in model_state.items()}
        self._global_model.load_state_dict(cloned)
        logger.info("Server '%s': loaded new global weights.", self.config.server_id)

    @staticmethod
    def _raise_if_incompatible(
        candidate: Mapping[str, torch.Tensor],
        reference: Mapping[str, torch.Tensor],
        context: str,
    ) -> None:
        """Raise ``ValueError`` if ``candidate`` doesn't structurally match ``reference``."""
        candidate_keys = set(candidate.keys())
        reference_keys = set(reference.keys())
        if candidate_keys != reference_keys:
            missing = sorted(reference_keys - candidate_keys)
            extra = sorted(candidate_keys - reference_keys)
            raise ValueError(
                f"{context} has incompatible keys (missing={missing}, extra={extra})."
            )
        for key, reference_tensor in reference.items():
            candidate_tensor = candidate[key]
            if tuple(candidate_tensor.shape) != tuple(reference_tensor.shape):
                raise ValueError(
                    f"{context}['{key}'] has incompatible shape: expected "
                    f"{tuple(reference_tensor.shape)}, got {tuple(candidate_tensor.shape)}."
                )

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def select_clients(
        self, round_id: int, available_client_ids: Sequence[str] | None = None
    ) -> tuple[str, ...]:
        """Deterministically select ``clients_per_round`` clients for ``round_id``.

        Selection depends only on ``config.seed``, ``round_id``, and the
        set of candidate client ids (sorted before sampling, so
        registration order never affects the result) — repeated calls with
        the same registered clients, seed, and round id always return the
        same selection.

        Args:
            round_id: The round to select clients for.
            available_client_ids: Optional explicit candidate pool
                (must all be registered); defaults to all registered
                clients.

        Returns:
            A sorted tuple of exactly ``config.clients_per_round`` client ids.

        Raises:
            ValueError: If ``round_id`` is negative, if
                ``available_client_ids`` contains unregistered clients, or
                if fewer than ``config.clients_per_round`` candidates are
                available.
        """
        if round_id < 0:
            raise ValueError(f"round_id must be >= 0, got {round_id}.")

        if available_client_ids is None:
            pool = sorted(self._registered_clients.keys())
        else:
            pool = sorted(set(available_client_ids))
            unregistered = [cid for cid in pool if cid not in self._registered_clients]
            if unregistered:
                raise ValueError(
                    f"available_client_ids contains unregistered client(s): {unregistered}."
                )

        if len(pool) < self.config.clients_per_round:
            raise ValueError(
                f"Not enough registered clients to select {self.config.clients_per_round} "
                f"for round {round_id}: only {len(pool)} available."
            )

        rng = random.Random(f"{self.config.seed}-{round_id}")
        selected = tuple(sorted(rng.sample(pool, self.config.clients_per_round)))
        return selected

    # ------------------------------------------------------------------
    # Round lifecycle
    # ------------------------------------------------------------------

    def start_round(
        self, round_id: int, selected_client_ids: Sequence[str] | None = None
    ) -> tuple[str, ...]:
        """Start a new active round.

        Only one round may be active at a time; call
        ``finalize_round_record`` to close the current round before
        starting another.

        Args:
            round_id: Identifier for the new round.
            selected_client_ids: Optional explicit client selection
                (all must be registered, non-empty, unique). If omitted,
                clients are chosen via ``select_clients(round_id)``.

        Returns:
            The tuple of selected client ids for this round.

        Raises:
            RuntimeError: If a round is already active.
            ValueError: If an explicit selection is empty, contains
                duplicates, or contains unregistered clients.
        """
        if self._current_round_id is not None:
            raise RuntimeError(
                f"Round {self._current_round_id} is already active; call "
                "finalize_round_record() before starting a new round."
            )

        if selected_client_ids is None:
            selected = self.select_clients(round_id)
        else:
            if not selected_client_ids:
                raise ValueError("selected_client_ids must not be empty.")
            if len(set(selected_client_ids)) != len(selected_client_ids):
                raise ValueError(
                    f"selected_client_ids must not contain duplicates: {list(selected_client_ids)}."
                )
            unregistered = [cid for cid in selected_client_ids if cid not in self._registered_clients]
            if unregistered:
                raise ValueError(
                    f"selected_client_ids contains unregistered client(s): {unregistered}."
                )
            selected = tuple(sorted(selected_client_ids))

        self._current_round_id = round_id
        self._selected_client_ids = selected
        self._received_updates = {}
        self._rejected_client_ids = set()
        self._round_started_at_utc = utc_now_iso()

        logger.info(
            "Server '%s': started round %d with %d selected client(s).",
            self.config.server_id, round_id, len(selected),
        )
        return selected

    def receive_update(self, update: ClientUpdate) -> None:
        """Validate and store one client's update for the active round.

        This method never aggregates or mutates the global model — it
        only validates ``update`` and, if valid, adds it to the set of
        accepted updates for the current round.

        # TODO (Sprint 3.3): whether `update.num_samples` should be
        # cross-checked against the client's registered sample count is
        # left undecided here. A future weighted-aggregation strategy will
        # need a defined answer for "which count is authoritative", so
        # this is deferred rather than guessed.

        Args:
            update: The client's reported update.

        Raises:
            RuntimeError: If no round is currently active.
            ValueError: If the client was not selected, already submitted
                an update this round, submitted a mismatched round id
                under ``strict_round_matching``, or submitted a
                model_state incompatible with the global model.
        """
        if self._current_round_id is None or self._selected_client_ids is None:
            raise RuntimeError("No active round. Call start_round() before receiving updates.")

        if update.client_id not in self._selected_client_ids:
            raise ValueError(
                f"Client '{update.client_id}' was not selected for round "
                f"{self._current_round_id}."
            )

        if update.client_id in self._received_updates:
            raise ValueError(
                f"Client '{update.client_id}' already submitted an update for round "
                f"{self._current_round_id}."
            )

        if self.config.strict_round_matching and update.round_id != self._current_round_id:
            self._rejected_client_ids.add(update.client_id)
            raise ValueError(
                f"Update round_id ({update.round_id}) does not match the active round "
                f"({self._current_round_id})."
            )

        try:
            self._raise_if_incompatible(
                update.model_state,
                self._global_model.state_dict(),
                context=f"Update from client '{update.client_id}': model_state",
            )
        except ValueError:
            self._rejected_client_ids.add(update.client_id)
            raise

        self._rejected_client_ids.discard(update.client_id)
        self._received_updates[update.client_id] = update
        logger.info(
            "Server '%s': accepted update from client '%s' for round %d (%d/%d accepted).",
            self.config.server_id, update.client_id, self._current_round_id,
            len(self._received_updates), self.config.minimum_updates,
        )

    def list_received_updates(self) -> tuple[ClientUpdate, ...]:
        """Return the accepted updates for the active round, in no particular order."""
        return tuple(self._received_updates.values())

    def has_minimum_updates(self) -> bool:
        """Return whether the active round has at least ``config.minimum_updates`` accepted updates.

        Returns False (not an error) when no round is active, since there
        are then trivially zero accepted updates.
        """
        return len(self._received_updates) >= self.config.minimum_updates

    def finalize_round_record(self, status: str) -> RoundRecord:
        """Close the active round and produce its ``RoundRecord``.

        This method never aggregates client updates into the global model
        — it only records what happened during the round. After this
        call, all active-round transient state (selection, accepted
        updates, rejected-client tracking) is cleared, and a new round can
        be started.

        Args:
            status: Either "completed" or "failed". ``RoundRecord`` itself
                supports a broader status vocabulary
                (``contracts.VALID_ROUND_STATUSES``) for future use, but
                this method — the only way to finalize a round today —
                only accepts these two terminal states.

        Returns:
            The finalized ``RoundRecord``, also appended to
            ``get_round_history()``.

        Raises:
            RuntimeError: If no round is currently active.
            ValueError: If ``status`` is not "completed"/"failed", or if
                ``status == "completed"`` without ``has_minimum_updates()``.
        """
        if self._current_round_id is None or self._selected_client_ids is None:
            raise RuntimeError("No active round to finalize. Call start_round() first.")
        if status not in ("completed", "failed"):
            raise ValueError(f"status must be 'completed' or 'failed', got '{status}'.")
        if status == "completed" and not self.has_minimum_updates():
            raise ValueError(
                f"Cannot finalize round {self._current_round_id} as 'completed': received "
                f"{len(self._received_updates)} update(s), need at least "
                f"{self.config.minimum_updates}."
            )

        record = RoundRecord(
            round_id=self._current_round_id,
            selected_client_ids=self._selected_client_ids,
            received_client_ids=tuple(self._received_updates.keys()),
            rejected_client_ids=tuple(self._rejected_client_ids),
            status=status,
            started_at_utc=self._round_started_at_utc or utc_now_iso(),
            metadata={},
            completed_at_utc=utc_now_iso() if status == "completed" else None,
        )
        self._round_history.append(record)

        logger.info(
            "Server '%s': finalized round %d as '%s' (%d received, %d rejected).",
            self.config.server_id, record.round_id, status,
            len(record.received_client_ids), len(record.rejected_client_ids),
        )

        self._current_round_id = None
        self._selected_client_ids = None
        self._received_updates = {}
        self._rejected_client_ids = set()
        self._round_started_at_utc = None

        return record

    def get_round_history(self) -> tuple[RoundRecord, ...]:
        """Return every finalized ``RoundRecord``, oldest first."""
        return tuple(self._round_history)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _default_checkpoint_path(self) -> Path:
        """Return the default checkpoint path for this server."""
        return self.config.checkpoint_dir / f"{self.config.server_id}_server.pt"

    def save_server_checkpoint(self, path: Path | None = None) -> Path:
        """Atomically save this server's state to ``path`` (or the default path).

        Excludes ``FederatedClient`` objects and raw datasets (the server
        never holds those in the first place) and excludes accepted
        ``ClientUpdate`` tensors for an unfinished round — only the
        *selected* client ids for an in-progress round are persisted, not
        their submitted model-state tensors, so resuming from a checkpoint
        taken mid-round starts that round over with zero accepted updates.

        Args:
            path: Destination path; defaults to
                ``checkpoint_dir / f"{server_id}_server.pt"``.

        Returns:
            The path the checkpoint was written to.
        """
        destination = path if path is not None else self._default_checkpoint_path()

        next_round_id = (
            self._current_round_id
            if self._current_round_id is not None
            else (self._round_history[-1].round_id + 1 if self._round_history else 0)
        )

        payload: dict[str, Any] = {
            "server_id": self.config.server_id,
            "current_round_id": self._current_round_id,
            "next_round_id": next_round_id,
            "selected_client_ids": (
                list(self._selected_client_ids) if self._selected_client_ids is not None else None
            ),
            "round_started_at_utc": self._round_started_at_utc,
            "global_model_state_dict": self._global_model.state_dict(),
            "registered_clients": self._registered_clients,
            "round_history": [_round_record_to_dict(record) for record in self._round_history],
            "config": _serializable_config(self.config),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        checkpoint_utils.save_checkpoint(payload, destination)
        return destination

    def load_server_checkpoint(self, path: Path) -> None:
        """Load a previously saved server checkpoint from ``path``.

        Validates that the checkpoint belongs to this server (matching
        ``server_id``) and that its global model is structurally
        compatible with this server's model before applying anything.

        Any in-progress (unfinished) round's accepted updates are not
        restored (they were never persisted); if a round was active when
        the checkpoint was saved, it is restored as active again but with
        zero accepted updates.

        Args:
            path: Path to a checkpoint previously written by
                ``save_server_checkpoint``.

        Raises:
            FileNotFoundError: If ``path`` does not exist.
            ValueError: If the checkpoint belongs to a different server id
                or has an incompatible global model.
        """
        payload = checkpoint_utils.load_checkpoint(path, map_location="cpu")

        stored_server_id = payload.get("server_id")
        if stored_server_id != self.config.server_id:
            raise ValueError(
                f"Checkpoint at {path} belongs to server '{stored_server_id}', "
                f"not '{self.config.server_id}'."
            )

        global_state_dict = payload["global_model_state_dict"]
        self._raise_if_incompatible(
            global_state_dict, self._global_model.state_dict(), context="Checkpoint global model"
        )

        self._global_model.load_state_dict(global_state_dict)
        self._registered_clients = payload.get("registered_clients", {})
        self._round_history = [
            _round_record_from_dict(record) for record in payload.get("round_history", [])
        ]
        self._current_round_id = payload.get("current_round_id")
        selected = payload.get("selected_client_ids")
        self._selected_client_ids = tuple(selected) if selected is not None else None
        self._round_started_at_utc = payload.get("round_started_at_utc")
        self._received_updates = {}
        self._rejected_client_ids = set()

        logger.info("Server '%s': loaded checkpoint from %s.", self.config.server_id, path)
