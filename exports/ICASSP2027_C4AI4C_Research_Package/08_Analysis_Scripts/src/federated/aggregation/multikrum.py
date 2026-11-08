"""Multi-Krum aggregation strategy: external Byzantine-robust baseline.

Implements the frozen definition of Blanchard, El Mhamdi, Guerraoui,
Stainer, "Machine Learning with Adversaries: Byzantine Tolerant Gradient
Descent" (NeurIPS 2017), exactly as specified in
``research_plans/MULTIKRUM_EXTERNAL_BASELINE_IMPLEMENTATION_PLAN.md``. No
variant of the original algorithm is introduced here.

This module serves as an external, literature-faithful robust-aggregation
baseline for comparison against this project's own FedTrace v3 attribution
method -- it is **not** a native attacker-attribution or onset-localization
method. Any per-round selected/rejected-client pattern this strategy
produces (exposed via ``AggregationResult.metadata["multikrum_diagnostics"]``)
reflects Multi-Krum's own Byzantine-robust aggregation decision, not
attacker ground truth, and must never be interpreted as malicious-client
detection without independent, offline evaluation against ground truth --
see the implementation plan's Section 11 ("repurposed rejection proxy").

Algorithm, for ``n`` participating clients and an assumed Byzantine bound
``f``:

    d(i, j) = || u_i - u_j ||_2^2   (squared Euclidean distance between
                                      client i's and client j's flattened
                                      trainable-parameter delta)

    score(i) = sum of the (n - f - 2) smallest values of
               { d(i, j) : j != i }

Multi-Krum selects the ``m = n - f`` lowest-scoring clients and averages
their updates **unweighted** (equal 1/m weight each) -- the original
method's convention, deliberately not FedAvg's sample-count weighting; see
the implementation plan Section 4 for why this difference is preserved
rather than "fixed" to match FedAvg.

Feasibility requires ``n >= 2f + 3`` (Krum's resilience proof); this
strategy raises rather than silently falling back to any other aggregation
rule when that bound is violated.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import torch

from src.federated.aggregation.base import (
    AggregationResult,
    AggregationStrategy,
    validate_common_update_structure,
)
from src.federated.aggregation.fedavg import FedAvg
from src.federated.contracts import ClientUpdate

_STRATEGY_NAME = "multikrum"


class MultiKrum(AggregationStrategy):
    """Literature-faithful Multi-Krum: distance-based Byzantine-robust aggregation.

    Krum scoring and client selection operate **only** on the flattened
    delta (``client_model_state[k] - global_state[k]``) of the trainable
    parameter keys supplied at construction time -- exactly the vector
    every attack in this project's series already manipulates (see the
    implementation plan Section 3). Non-trainable floating buffers (e.g.
    BatchNorm ``running_mean``/``running_var``) and non-floating buffers
    (e.g. ``.num_batches_tracked``) are explicitly excluded from the
    distance computation; see ``aggregate`` for how each is still given a
    value in the returned aggregated state dict.
    """

    def __init__(self, trainable_keys: Iterable[str], f: int = 1) -> None:
        """Construct a Multi-Krum strategy around a fixed set of trainable keys.

        Args:
            trainable_keys: The ``global_state``/``model_state`` keys that
                identify trainable parameters (as opposed to buffers).
                Typically ``frozenset(name for name, _ in
                model_fn().named_parameters())`` -- the same set this
                project's attacks and FedTrace already use to define "the
                update". Must be non-empty.
            f: Assumed upper bound on the number of Byzantine clients this
                round. Fixed at ``1`` for this project's frozen baseline
                (single-attacker threat model; see the implementation plan
                Section 2) -- callers running the real experiment pipeline
                must not vary this per attack family or seed. Accepted as
                a constructor argument (rather than hardcoded) only so
                this class remains independently unit-testable against
                other ``n``/``f`` combinations.

        Raises:
            ValueError: If ``trainable_keys`` is empty or ``f`` is
                negative.
        """
        trainable_keys = frozenset(trainable_keys)
        if not trainable_keys:
            raise ValueError("trainable_keys must be non-empty.")
        if f < 0:
            raise ValueError(f"f must be >= 0, got {f}.")

        self._trainable_keys = trainable_keys
        self._f = f

    @property
    def name(self) -> str:
        """Return ``"multikrum"``."""
        return _STRATEGY_NAME

    def aggregate(
        self,
        global_state: Mapping[str, torch.Tensor],
        updates: Sequence[ClientUpdate],
    ) -> AggregationResult:
        """Aggregate ``updates`` via Multi-Krum selection and unweighted averaging.

        Args:
            global_state: The current global model's state dict. Read
                only as a reference for keys/shapes/dtypes and as the
                origin point for each client's delta; never mutated.
            updates: The client updates to aggregate. Must be non-empty,
                have unique client ids, agree on ``round_id``, each have
                ``model_state`` keys matching ``global_state``, and number
                at least ``2f + 3`` (this strategy's own ``f``) for
                Krum's resilience guarantee to hold.

        Returns:
            A validated ``AggregationResult`` whose ``aggregated_state``
            is the unweighted mean of the ``m = n - f`` lowest-Krum-score
            clients' trainable parameters (and, for non-trainable floating
            buffers, the same selected clients' unweighted mean; for
            non-floating buffers, FedAvg's existing max/identical
            conventions applied across all participating clients). Its
            ``metadata["multikrum_diagnostics"]`` carries, per client, the
            Krum score, rank, and selection outcome -- aggregation
            diagnostics only, never attacker ground truth (see the module
            docstring).

        Raises:
            ValueError: If ``updates`` is empty, has duplicate client ids,
                disagreeing round ids, mismatched model-state keys, a
                shape mismatch for any tensor, a non-floating buffer
                disagreement, or if ``len(updates) < 2f + 3`` (infeasible
                for this strategy's ``f``).
            TypeError: If a client's tensor for some key has a dtype
                category (floating vs. non-floating) incompatible with the
                global model's tensor for that key.
        """
        round_id = validate_common_update_structure(updates, global_state)

        unknown_trainable_keys = self._trainable_keys - set(global_state.keys())
        if unknown_trainable_keys:
            raise ValueError(
                "trainable_keys contains key(s) not present in global_state: "
                f"{sorted(unknown_trainable_keys)}."
            )

        n = len(updates)
        f = self._f
        if n < 2 * f + 3:
            raise ValueError(
                f"Infeasible Multi-Krum configuration: {n} participating update(s) but "
                f"f={f} requires n >= 2f+3 = {2 * f + 3} for Krum's resilience guarantee "
                "to hold. Multi-Krum does not silently fall back to any other aggregation "
                "rule; either more clients must participate or f must be lowered."
            )
        m = n - f
        neighbors_used = n - f - 2
        # Guaranteed by the n >= 2f+3 check above for any f >= 0; asserted
        # defensively rather than re-derived with separate raise paths.
        assert 1 <= m <= n
        assert neighbors_used >= 1

        for key in sorted(global_state.keys()):
            reference_tensor = global_state[key]
            per_client_tensors = [update.model_state[key] for update in updates]
            FedAvg._validate_shapes_and_dtypes(key, reference_tensor, per_client_tensors, updates)

        sorted_trainable_keys = sorted(self._trainable_keys)
        vectors = {
            update.client_id: self._flatten_trainable_delta(
                update.model_state, global_state, sorted_trainable_keys
            )
            for update in updates
        }

        distances = self._pairwise_squared_distances(vectors)
        scores = self._krum_scores(distances, neighbors_used)
        ranks, selected_ids, rejected_ids = self._rank_and_select(scores, m)

        selected_updates = [update for update in updates if update.client_id in set(selected_ids)]

        aggregated_state: dict[str, torch.Tensor] = {}
        for key in sorted(global_state.keys()):
            reference_tensor = global_state[key]
            if reference_tensor.is_floating_point():
                # Trainable parameters AND non-trainable floating buffers
                # (e.g. BatchNorm running_mean/running_var) both follow
                # Multi-Krum's own selection decision, unweighted-averaged
                # over the same m selected clients -- see the module
                # docstring and implementation plan Section 4.
                per_selected_tensors = [update.model_state[key] for update in selected_updates]
                aggregated_state[key] = self._unweighted_average(
                    per_selected_tensors, reference_tensor.dtype
                )
            elif key.endswith(".num_batches_tracked"):
                per_all_tensors = [update.model_state[key] for update in updates]
                aggregated_state[key] = FedAvg._max_across_clients(per_all_tensors, reference_tensor)
            else:
                per_all_tensors = [update.model_state[key] for update in updates]
                aggregated_state[key] = FedAvg._require_identical(key, per_all_tensors, updates)

        multikrum_diagnostics = {
            client_id: {
                "krum_score": scores[client_id],
                "krum_rank": ranks[client_id],
                "selected": client_id in set(selected_ids),
            }
            for client_id in sorted(scores.keys())
        }

        metadata = {
            "num_clients": n,
            "f": f,
            "m": m,
            "neighbors_used": neighbors_used,
            "weighting": "unweighted_selected_mean",
            "selected_client_ids": selected_ids,
            "rejected_client_ids": rejected_ids,
            "multikrum_diagnostics": multikrum_diagnostics,
        }

        return AggregationResult(
            strategy_name=self.name,
            round_id=round_id,
            aggregated_state=aggregated_state,
            participating_client_ids=selected_ids,
            total_weight=float(m),
            metadata=metadata,
        )

    @staticmethod
    def _flatten_trainable_delta(
        model_state: Mapping[str, torch.Tensor],
        global_state: Mapping[str, torch.Tensor],
        sorted_trainable_keys: list[str],
    ) -> torch.Tensor:
        """Flatten one client's trainable-parameter delta into a single float64 vector.

        Concatenates ``model_state[k] - global_state[k]`` for
        ``k in sorted_trainable_keys`` (a fixed, deterministic key order,
        so the same client always produces the same vector layout) into a
        single 1-D tensor. Non-trainable buffers are never included --
        this is the exact vector Krum's distance computation operates on.
        """
        pieces = [
            (
                model_state[key].detach().to(torch.float64)
                - global_state[key].detach().to(torch.float64)
            ).reshape(-1)
            for key in sorted_trainable_keys
        ]
        return torch.cat(pieces)

    @staticmethod
    def _pairwise_squared_distances(
        vectors: Mapping[str, torch.Tensor]
    ) -> dict[str, dict[str, float]]:
        """Compute ``d(i, j) = ||u_i - u_j||_2^2`` for every ordered pair of distinct clients.

        Computed directly as a sum of squared differences (never via a
        Euclidean-norm-then-square round trip), so results are exact for
        hand-computable test vectors, not subject to a sqrt/square
        precision loss.

        Returns:
            A mapping ``{client_id: {peer_client_id: squared_distance}}``
            containing every ``(i, j)`` with ``i != j`` -- no self-entry.
        """
        client_ids = sorted(vectors.keys())
        result: dict[str, dict[str, float]] = {client_id: {} for client_id in client_ids}
        for i, client_i in enumerate(client_ids):
            for j, client_j in enumerate(client_ids):
                if i == j:
                    continue
                diff = vectors[client_i] - vectors[client_j]
                result[client_i][client_j] = float((diff * diff).sum().item())
        return result

    @staticmethod
    def _krum_scores(
        distances: Mapping[str, Mapping[str, float]], neighbors_used: int
    ) -> dict[str, float]:
        """Sum each client's ``neighbors_used`` smallest peer distances into its Krum score."""
        scores: dict[str, float] = {}
        for client_id, peer_distances in distances.items():
            smallest = sorted(peer_distances.values())[:neighbors_used]
            scores[client_id] = float(sum(smallest))
        return scores

    @staticmethod
    def _rank_and_select(
        scores: Mapping[str, float], m: int
    ) -> tuple[dict[str, int], tuple[str, ...], tuple[str, ...]]:
        """Rank clients by ascending Krum score and select the ``m`` lowest-scoring.

        Ties are broken deterministically by ``client_id`` lexical
        ordering (the narrowest, stable rule available without depending
        on input iteration order), matching this project's own
        deterministic-selection convention elsewhere (e.g.
        ``FederatedServer.select_clients``'s seeded, sorted-pool
        approach).

        Returns:
            ``(ranks, selected_client_ids, rejected_client_ids)`` where
            ``ranks`` maps client id to its 1-indexed rank (1 = lowest
            score / most trusted) and the two id tuples are each sorted
            ascending by client id.
        """
        ordered = sorted(scores.keys(), key=lambda client_id: (scores[client_id], client_id))
        ranks = {client_id: rank for rank, client_id in enumerate(ordered, start=1)}
        selected = tuple(sorted(ordered[:m]))
        rejected = tuple(sorted(ordered[m:]))
        return ranks, selected, rejected

    @staticmethod
    def _unweighted_average(
        per_client_tensors: list[torch.Tensor], output_dtype: torch.dtype
    ) -> torch.Tensor:
        """Unweighted (equal-weight) mean of a floating-point tensor across selected clients.

        Deliberately not sample-count-weighted -- see the module and class
        docstrings. Accumulation happens in float64 for numerical safety
        regardless of the tensors' original dtype, then the result is cast
        back to ``output_dtype``.
        """
        accumulator = torch.zeros_like(per_client_tensors[0], dtype=torch.float64)
        for tensor in per_client_tensors:
            accumulator = accumulator + tensor.detach().to(torch.float64)
        accumulator = accumulator / len(per_client_tensors)
        return accumulator.to(output_dtype).detach().cpu().clone()
