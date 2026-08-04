"""FedAvg aggregation strategy (Sprint 3.3).

Sample-count-weighted averaging of client model updates, as in McMahan et
al., 2017 ("Communication-Efficient Learning of Deep Networks from
Decentralized Data"). This module performs pure aggregation only — it
does not select clients, run rounds, train clients, or apply its result
to any model. The caller decides whether and how to load the returned
``AggregationResult.aggregated_state`` (e.g. via
``FederatedServer.load_global_weights``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

from src.federated.aggregation.base import (
    AggregationResult,
    AggregationStrategy,
    validate_common_update_structure,
)
from src.federated.contracts import ClientUpdate

_STRATEGY_NAME = "fedavg"


class FedAvg(AggregationStrategy):
    """Sample-count-weighted federated averaging.

    Floating-point tensors are averaged with weight
    ``update.num_samples / total_samples`` per client, accumulated in
    float64 for numerical safety and cast back to the global model's
    original dtype for that parameter.

    Non-floating tensors (integer/boolean — e.g. BatchNorm's
    ``num_batches_tracked``) are never averaged numerically: every
    participating client must report an identical value, and that value
    is returned unchanged. See ``_require_identical`` for the rationale.
    """

    @property
    def name(self) -> str:
        """Return ``"fedavg"``."""
        return _STRATEGY_NAME

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        """Aggregate ``updates`` via sample-count-weighted averaging.

        Args:
            global_state: The current global model's state dict. Read
                only as a reference for keys/shapes/dtypes; never mutated
                and its values never used numerically.
            updates: The client updates to aggregate. Must be non-empty,
                have unique client ids, agree on ``round_id``, and each
                have ``model_state`` keys matching ``global_state``.

        Returns:
            A validated ``AggregationResult`` with the weighted-averaged
            state dict.

        Raises:
            ValueError: If ``updates`` is empty, has duplicate client ids,
                disagreeing round ids, mismatched model-state keys, a
                shape mismatch for any tensor, or differing values for a
                non-floating tensor.
            TypeError: If a client's tensor for some key has a dtype
                category (floating vs. non-floating) incompatible with the
                global model's tensor for that key.
        """
        round_id = validate_common_update_structure(updates, global_state)
        total_samples = sum(update.num_samples for update in updates)
        # ClientUpdate.__post_init__ guarantees num_samples > 0 for every
        # update, so total_samples > 0 whenever updates is non-empty.

        aggregated_state: dict[str, torch.Tensor] = {}
        for key in sorted(global_state.keys()):
            reference_tensor = global_state[key]
            per_client_tensors = [update.model_state[key] for update in updates]

            self._validate_shapes_and_dtypes(key, reference_tensor, per_client_tensors, updates)

            if reference_tensor.is_floating_point():
                aggregated_state[key] = self._weighted_average(
                    per_client_tensors, updates, total_samples, reference_tensor.dtype
                )
            else:
                aggregated_state[key] = self._require_identical(key, per_client_tensors, updates)

        metadata = {
            "num_clients": len(updates),
            "weighting": "num_samples",
            "total_samples": total_samples,
        }

        return AggregationResult(
            strategy_name=self.name,
            round_id=round_id,
            aggregated_state=aggregated_state,
            participating_client_ids=tuple(update.client_id for update in updates),
            total_weight=float(total_samples),
            metadata=metadata,
        )

    @staticmethod
    def _validate_shapes_and_dtypes(
        key: str,
        reference_tensor: torch.Tensor,
        per_client_tensors: list[torch.Tensor],
        updates: Sequence[ClientUpdate],
    ) -> None:
        """Raise if any client's tensor for ``key`` has an incompatible shape or dtype category."""
        for update, tensor in zip(updates, per_client_tensors):
            if tuple(tensor.shape) != tuple(reference_tensor.shape):
                raise ValueError(
                    f"Update from client '{update.client_id}' has incompatible shape for "
                    f"'{key}': expected {tuple(reference_tensor.shape)}, got {tuple(tensor.shape)}."
                )
            if tensor.is_floating_point() != reference_tensor.is_floating_point():
                expected_kind = "floating-point" if reference_tensor.is_floating_point() else "non-floating"
                raise TypeError(
                    f"Update from client '{update.client_id}' has incompatible dtype for "
                    f"'{key}': expected a {expected_kind} tensor (global dtype "
                    f"{reference_tensor.dtype}), got {tensor.dtype}."
                )

    @staticmethod
    def _weighted_average(
        per_client_tensors: list[torch.Tensor],
        updates: Sequence[ClientUpdate],
        total_samples: int,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sample-count-weighted average of a floating-point tensor across clients.

        Accumulation happens in float64 for numerical safety regardless of
        the tensors' original dtype, then the result is cast back to
        ``output_dtype`` (the global model's dtype for this parameter), so
        the returned aggregated tensor's dtype matches what the model
        actually expects. Inputs are never modified: all arithmetic here
        is out-of-place.
        """
        accumulator = torch.zeros_like(per_client_tensors[0], dtype=torch.float64)
        for update, tensor in zip(updates, per_client_tensors):
            weight = update.num_samples / total_samples
            accumulator = accumulator + tensor.detach().to(torch.float64) * weight
        return accumulator.to(output_dtype).detach().cpu().clone()

    @staticmethod
    def _require_identical(
        key: str, per_client_tensors: list[torch.Tensor], updates: Sequence[ClientUpdate]
    ) -> torch.Tensor:
        """Return the shared value of a non-floating tensor, or raise if clients disagree.

        Non-floating tensors (integer counters, boolean flags) have no
        well-defined "average" — e.g. averaging BatchNorm's
        ``num_batches_tracked`` across clients that ran a different number
        of local batches would produce a fabricated, meaningless count.
        FedAvg therefore requires exact agreement and passes the value
        through unchanged rather than guessing at a numeric policy.
        """
        reference = per_client_tensors[0]
        for update, tensor in zip(updates, per_client_tensors):
            if not torch.equal(tensor, reference):
                raise ValueError(
                    f"Non-floating tensor '{key}' differs across clients (client "
                    f"'{updates[0].client_id}'={reference.tolist()} vs client "
                    f"'{update.client_id}'={tensor.tolist()}); FedAvg requires identical "
                    "values for non-floating tensors."
                )
        return reference.detach().cpu().clone()
