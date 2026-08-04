"""Abstract aggregation interface for federated learning (Sprint 3.3).

Defines the plugin contract every aggregation strategy (starting with
``FedAvg``; future strategies such as FedProx/SCAFFOLD/FedNova would
implement the same contract) must satisfy, and the immutable, validated
result type they must return.

This module contains no concrete aggregation logic itself and no
dependency on ``FederatedServer`` or ``FederatedClient`` — it only defines
the interface, the result type, and structural validation shared by any
strategy that combines a sequence of ``ClientUpdate``s.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from src.federated.contracts import ClientUpdate


@dataclass(frozen=True)
class AggregationResult:
    """The validated, immutable output of one aggregation strategy call.

    Attributes:
        strategy_name: Name of the strategy that produced this result
            (e.g. ``"fedavg"``).
        round_id: The round all contributing updates belonged to.
        aggregated_state: The aggregated model state dict. Always cloned,
            detached, CPU tensors — never a live reference into any
            client's or the global model's tensors.
        participating_client_ids: The client ids whose updates
            contributed to this result. Unique and non-empty.
        total_weight: The sum of the weighting quantity used (e.g. total
            sample count for FedAvg). Must be positive and finite.
        metadata: Free-form, simple serializable key/value pairs
            describing how aggregation was performed.
    """

    strategy_name: str
    round_id: int
    aggregated_state: Mapping[str, torch.Tensor]
    participating_client_ids: tuple[str, ...]
    total_weight: float
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate all fields, then normalize mapping fields to safe, immutable copies."""
        if not self.strategy_name:
            raise ValueError("strategy_name must be a non-empty string.")
        if self.round_id < 0:
            raise ValueError(f"round_id must be >= 0, got {self.round_id}.")
        if not self.aggregated_state:
            raise ValueError("aggregated_state must be non-empty.")
        if not self.participating_client_ids:
            raise ValueError("participating_client_ids must be non-empty.")
        if len(set(self.participating_client_ids)) != len(self.participating_client_ids):
            raise ValueError(
                "participating_client_ids must not contain duplicates: "
                f"{self.participating_client_ids}."
            )
        if not math.isfinite(self.total_weight) or self.total_weight <= 0:
            raise ValueError(
                f"total_weight must be positive and finite, got {self.total_weight}."
            )

        cloned_state: dict[str, torch.Tensor] = {}
        for key, value in self.aggregated_state.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"aggregated_state['{key}'] must be a torch.Tensor, got {type(value).__name__}."
                )
            cloned_state[key] = value.detach().cpu().clone()

        # Frozen dataclass: object.__setattr__ is required to normalize
        # fields in __post_init__.
        object.__setattr__(self, "aggregated_state", MappingProxyType(cloned_state))
        object.__setattr__(
            self, "participating_client_ids", tuple(self.participating_client_ids)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class AggregationStrategy(ABC):
    """Abstract base class for pluggable federated aggregation strategies.

    Concrete strategies (e.g. ``FedAvg``) implement ``aggregate`` to
    combine a global model's state dict with a sequence of
    ``ClientUpdate``s into an ``AggregationResult``. Implementations must
    not mutate ``global_state`` or any ``ClientUpdate`` passed in, and must
    not depend on ``FederatedServer`` or ``FederatedClient`` — they operate
    purely on the data contracts in ``src.federated.contracts``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A short, stable identifier for this strategy (e.g. ``"fedavg"``)."""
        raise NotImplementedError

    @abstractmethod
    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        """Combine ``updates`` (and optionally ``global_state``) into one result.

        Args:
            global_state: The current global model's state dict, provided
                as a reference for keys/shapes (and, for strategies that
                need it, e.g. a future proximal-term strategy, as the
                current global weights themselves). Implementations must
                not mutate this mapping.
            updates: The accepted client updates to aggregate. Must not be
                mutated.

        Returns:
            A validated ``AggregationResult``.
        """
        raise NotImplementedError


def validate_common_update_structure(
    updates: Sequence[ClientUpdate],
    global_state: Mapping[str, torch.Tensor],
) -> int:
    """Validate structural invariants shared by any aggregation strategy.

    Checks that ``updates`` is non-empty, has unique client ids, all share
    a single ``round_id``, and every update's ``model_state`` keys exactly
    match ``global_state``'s keys. Does not check tensor shapes or dtypes
    — those checks can differ by strategy and are left to each concrete
    strategy's own ``aggregate`` implementation.

    Args:
        updates: The client updates to validate.
        global_state: The current global model state dict.

    Returns:
        The single shared ``round_id`` across all updates.

    Raises:
        ValueError: If ``updates`` is empty, contains duplicate client
            ids, disagrees on ``round_id``, or any update's ``model_state``
            keys don't exactly match ``global_state``'s keys.
    """
    if not updates:
        raise ValueError("aggregate() requires at least one update.")

    client_ids = [update.client_id for update in updates]
    if len(set(client_ids)) != len(client_ids):
        raise ValueError(f"Duplicate client ids in updates: {client_ids}.")

    round_ids = {update.round_id for update in updates}
    if len(round_ids) != 1:
        raise ValueError(f"All updates must share the same round_id, got {sorted(round_ids)}.")

    reference_keys = set(global_state.keys())
    for update in updates:
        update_keys = set(update.model_state.keys())
        if update_keys != reference_keys:
            missing = sorted(reference_keys - update_keys)
            extra = sorted(update_keys - reference_keys)
            raise ValueError(
                f"Update from client '{update.client_id}' has model_state keys "
                f"incompatible with global_state (missing={missing}, extra={extra})."
            )

    return next(iter(round_ids))
