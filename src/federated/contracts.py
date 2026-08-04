"""Client-server data contracts for federated learning (Sprint 3.2).

Defines the two data shapes that flow between a ``FederatedClient`` and a
``FederatedServer``:

- ``ClientUpdate``: what a client reports back after local training.
- ``RoundRecord``: the server's bookkeeping for one round of client
  participation.

Both are immutable, validated dataclasses with no knowledge of any
specific aggregation algorithm (e.g. FedAvg), communication mechanism, or
model architecture. Neither type performs any training, aggregation, or
network I/O.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

import torch

VALID_ROUND_STATUSES: tuple[str, ...] = (
    "created",
    "collecting",
    "ready",
    "completed",
    "failed",
)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ClientUpdate:
    """One client's reported result for a single federated round.

    Immutable and validated at construction time. Tensors in
    ``model_state`` are always stored as detached, cloned CPU tensors,
    regardless of what device they were on when passed in, so a
    ``ClientUpdate`` never keeps a live reference into a client's (possibly
    GPU-resident) model. ``model_state``, ``train_metrics``, and
    ``metadata`` are additionally wrapped as read-only mappings so nested
    mutation (e.g. ``update.model_state["x"] = ...``) is blocked too, not
    just reassignment of the dataclass fields themselves.

    This class has no field for optimizer or scheduler state by design —
    there is nowhere to put it, so it structurally cannot be included. A
    future aggregation strategy (Sprint 3.3+) is expected to need only
    model state, sample counts, and metrics.

    Attributes:
        client_id: Identifier of the client that produced this update.
        round_id: The round this update was produced for.
        model_state: The client's local model ``state_dict`` (parameter
            and buffer names to tensors) after local training.
        num_samples: Number of local training samples used to produce this
            update.
        train_metrics: Local training metrics (e.g. final loss/top-1); all
            values must be finite.
        metadata: Free-form, simple serializable key/value pairs. Not
            required to be numeric and not validated for finiteness.
        created_at_utc: ISO-8601 UTC timestamp string of when this update
            was created.
    """

    client_id: str
    round_id: int
    model_state: Mapping[str, torch.Tensor]
    num_samples: int
    train_metrics: Mapping[str, float]
    metadata: Mapping[str, Any]
    created_at_utc: str

    def __post_init__(self) -> None:
        """Validate all fields, then normalize the mapping fields to safe, immutable copies."""
        if not self.client_id:
            raise ValueError("client_id must be a non-empty string.")
        if self.round_id < 0:
            raise ValueError(f"round_id must be >= 0, got {self.round_id}.")
        if self.num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {self.num_samples}.")
        if not self.model_state:
            raise ValueError("model_state must be non-empty.")

        cloned_state: dict[str, torch.Tensor] = {}
        for key, value in self.model_state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"model_state['{key}'] must be a torch.Tensor, got {type(value).__name__}."
                )
            cloned_state[key] = value.detach().cpu().clone()

        for key, value in self.train_metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"train_metrics['{key}'] must be finite, got {value}.")

        # Frozen dataclass: object.__setattr__ is required to normalize
        # fields in __post_init__. Wrapping in MappingProxyType makes the
        # mappings themselves read-only, not just the dataclass fields.
        object.__setattr__(self, "model_state", MappingProxyType(cloned_state))
        object.__setattr__(self, "train_metrics", MappingProxyType(dict(self.train_metrics)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @classmethod
    def create(
        cls,
        client_id: str,
        round_id: int,
        model_state: Mapping[str, torch.Tensor],
        num_samples: int,
        train_metrics: Mapping[str, float] | None = None,
        metadata: Mapping[str, Any] | None = None,
        created_at_utc: str | None = None,
    ) -> ClientUpdate:
        """Safe factory for ``ClientUpdate`` with convenient defaults.

        All validation and tensor cloning still happens in
        ``__post_init__`` via the normal constructor call this factory
        makes, so there is exactly one validation path to keep correct.

        Args:
            client_id: Identifier of the reporting client.
            round_id: The round this update was produced for.
            model_state: The client's local model state dict.
            num_samples: Number of local training samples used.
            train_metrics: Local training metrics; defaults to ``{}``.
            metadata: Free-form metadata; defaults to ``{}``.
            created_at_utc: ISO-8601 UTC timestamp; defaults to now.

        Returns:
            A validated, immutable ``ClientUpdate``.
        """
        return cls(
            client_id=client_id,
            round_id=round_id,
            model_state=model_state,
            num_samples=num_samples,
            train_metrics=dict(train_metrics) if train_metrics is not None else {},
            metadata=dict(metadata) if metadata is not None else {},
            created_at_utc=created_at_utc if created_at_utc is not None else utc_now_iso(),
        )


@dataclass(frozen=True)
class RoundRecord:
    """Server-side bookkeeping for one federated round.

    Tracks which clients were selected, which of them the server accepted
    updates from, and which were rejected — without holding any
    aggregation result. Aggregation results belong to a future Sprint
    3.3+ module, not to this record.

    Attributes:
        round_id: The round this record describes.
        selected_client_ids: Client IDs selected to participate this round.
        received_client_ids: Client IDs whose updates were accepted.
        rejected_client_ids: Client IDs that were selected but whose
            update was rejected (e.g. mismatched round id or incompatible
            model state).
        status: One of ``VALID_ROUND_STATUSES``.
        started_at_utc: ISO-8601 UTC timestamp of round start.
        metadata: Free-form, simple serializable key/value pairs.
        completed_at_utc: ISO-8601 UTC timestamp of round completion;
            required when ``status == "completed"``.
    """

    round_id: int
    selected_client_ids: tuple[str, ...]
    received_client_ids: tuple[str, ...]
    rejected_client_ids: tuple[str, ...]
    status: str
    started_at_utc: str
    metadata: Mapping[str, Any]
    completed_at_utc: str | None = None

    def __post_init__(self) -> None:
        """Validate round bookkeeping invariants and freeze mutable fields."""
        if self.round_id < 0:
            raise ValueError(f"round_id must be >= 0, got {self.round_id}.")
        if self.status not in VALID_ROUND_STATUSES:
            raise ValueError(
                f"status must be one of {VALID_ROUND_STATUSES}, got '{self.status}'."
            )

        selected = tuple(self.selected_client_ids)
        received = tuple(self.received_client_ids)
        rejected = tuple(self.rejected_client_ids)

        if len(set(selected)) != len(selected):
            raise ValueError(f"selected_client_ids must not contain duplicates: {selected}.")
        if len(set(received)) != len(received):
            raise ValueError(f"received_client_ids must not contain duplicates: {received}.")
        if len(set(rejected)) != len(rejected):
            raise ValueError(f"rejected_client_ids must not contain duplicates: {rejected}.")

        selected_set = set(selected)
        missing_received = set(received) - selected_set
        if missing_received:
            raise ValueError(
                "received_client_ids contains client(s) not in selected_client_ids: "
                f"{sorted(missing_received)}."
            )
        missing_rejected = set(rejected) - selected_set
        if missing_rejected:
            raise ValueError(
                "rejected_client_ids contains client(s) not in selected_client_ids: "
                f"{sorted(missing_rejected)}."
            )

        overlap = set(received) & set(rejected)
        if overlap:
            raise ValueError(
                f"A client cannot be both received and rejected: {sorted(overlap)}."
            )

        if self.status == "completed" and self.completed_at_utc is None:
            raise ValueError("completed_at_utc is required when status == 'completed'.")

        object.__setattr__(self, "selected_client_ids", selected)
        object.__setattr__(self, "received_client_ids", received)
        object.__setattr__(self, "rejected_client_ids", rejected)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
