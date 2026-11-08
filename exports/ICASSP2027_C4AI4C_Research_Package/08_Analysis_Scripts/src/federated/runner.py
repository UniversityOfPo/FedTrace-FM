"""Federated round orchestration (Sprint 3.4).

``FederatedRunner`` connects an already-constructed ``FederatedServer``, a
mapping of already-constructed ``FederatedClient``s, and a pluggable
``AggregationStrategy`` into communication rounds: select clients, train
them locally, collect ``ClientUpdate``s, aggregate, and explicitly apply
the result to the server's global model.

This module does not:
    - Create clients or a server (both are handed in via the constructor).
    - Implement any aggregation algorithm itself (it is strategy-agnostic
      by construction — see ``AggregationStrategy``).
    - Run more than one client "at a time" (clients are processed
      sequentially, in the order the server selected them).
    - Know anything about CIFAR-100, attacks, attribution, or a CLI. (Phase
      5: an optional, fully generic per-client ``update_transform_callback``
      hook may be supplied by the caller -- e.g. to implement an attack --
      but the runner itself has no built-in concept of what an attack is;
      see that parameter's docstring.)

Precondition: every ``FederatedClient`` passed to this runner must already
have had ``initialize_model()`` called on it before the runner is used
(for both ``run()``/``run_round()`` and ``load_runner_checkpoint()``) —
``FederatedClient.load_global_weights``/``load_local_checkpoint`` both
require this already, and the runner does not call
``initialize_model()`` on the caller's behalf (consistent with
``FederatedClient``'s own design: model construction is explicit, not an
implicit side effect of something else).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import torch
from torch import nn

from src.federated.aggregation.base import AggregationStrategy
from src.federated.client import FederatedClient
from src.federated.contracts import ClientUpdate, utc_now_iso
from src.federated.server import FederatedServer
from src.utils import checkpoint as checkpoint_utils

logger = logging.getLogger(__name__)

VALID_ROUND_RESULT_STATUSES: tuple[str, ...] = ("completed", "failed", "skipped")


@dataclass(frozen=True)
class FederatedRunnerConfig:
    """Runner-level configuration.

    Attributes:
        total_rounds: Total number of communication rounds ``run()`` will
            drive the runner through.
        evaluate_every: Evaluation cadence in rounds. A round is evaluated
            (when an evaluation callback is configured) when
            ``round_id % evaluate_every == 0``; round ids are 0-indexed,
            so the default of 1 evaluates every round, including round 0.
        checkpoint_every: Reserved cadence (in rounds) for future automatic
            checkpointing. Validated here but **not yet wired to any
            behavior** — see the ``FederatedRunner`` class docstring TODO.
        fail_round_on_client_error: If True, the first client training/
            update-submission error in a round aborts that round
            immediately (remaining selected clients are not processed) and
            the round is finalized as failed. If False (default), the
            runner continues with the remaining selected clients.
        continue_after_failed_round: If True (default), ``run()`` proceeds
            to the next round after a failed round. If False, ``run()``
            stops as soon as a round fails.
        deterministic: Recorded for provenance and forwarded into
            checkpoints. Has **no runtime effect yet** — see the
            ``FederatedRunner`` class docstring TODO.
    """

    total_rounds: int
    evaluate_every: int = 1
    checkpoint_every: int = 1
    fail_round_on_client_error: bool = False
    continue_after_failed_round: bool = True
    deterministic: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.total_rounds <= 0:
            raise ValueError(f"total_rounds must be > 0, got {self.total_rounds}.")
        if self.evaluate_every <= 0:
            raise ValueError(f"evaluate_every must be > 0, got {self.evaluate_every}.")
        if self.checkpoint_every <= 0:
            raise ValueError(f"checkpoint_every must be > 0, got {self.checkpoint_every}.")


@dataclass(frozen=True)
class FederatedRoundResult:
    """The outcome of one ``FederatedRunner.run_round()`` call.

    Attributes:
        round_id: The round this result describes.
        selected_client_ids: Client ids the server selected for this round.
        successful_client_ids: Client ids that trained locally and had
            their update accepted by the server.
        failed_client_ids: Client ids that were selected but did not
            produce an accepted update (training error, update-submission
            error, or — defensively — an id the runner has no
            ``FederatedClient`` for).
        aggregation_strategy: ``AggregationStrategy.name`` of the strategy
            used (or configured) for this round.
        total_samples: Sum of ``num_samples`` across successful clients'
            updates. 0 for a round that never reached aggregation.
        train_metrics: Sample-weighted average of local training metrics
            shared by every successful client this round (see
            ``FederatedRunner``'s metrics-aggregation policy). Empty for a
            round with no successful clients.
        evaluation_metrics: The evaluation callback's return value for
            this round, or empty if no callback is configured, this round
            was not due for evaluation, or the callback raised.
        round_status: One of ``VALID_ROUND_RESULT_STATUSES``.
        started_at_utc: ISO-8601 UTC timestamp when this round began.
        completed_at_utc: ISO-8601 UTC timestamp when the runner finished
            processing this round (regardless of ``round_status`` — this
            is "when the attempt finished", not "when it succeeded").
        metadata: Free-form, simple serializable key/value pairs (e.g.
            per-client error text, aggregation metadata, elapsed seconds).
    """

    round_id: int
    selected_client_ids: tuple[str, ...]
    successful_client_ids: tuple[str, ...]
    failed_client_ids: tuple[str, ...]
    aggregation_strategy: str
    total_samples: int
    train_metrics: Mapping[str, float]
    evaluation_metrics: Mapping[str, float]
    round_status: str
    started_at_utc: str
    completed_at_utc: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Validate round-result invariants and freeze mutable fields."""
        if self.round_id < 0:
            raise ValueError(f"round_id must be >= 0, got {self.round_id}.")
        if self.round_status not in VALID_ROUND_RESULT_STATUSES:
            raise ValueError(
                f"round_status must be one of {VALID_ROUND_RESULT_STATUSES}, "
                f"got '{self.round_status}'."
            )
        if self.total_samples < 0:
            raise ValueError(f"total_samples must be >= 0, got {self.total_samples}.")

        selected = tuple(self.selected_client_ids)
        successful = tuple(self.successful_client_ids)
        failed = tuple(self.failed_client_ids)

        for field_name, ids in (
            ("selected_client_ids", selected),
            ("successful_client_ids", successful),
            ("failed_client_ids", failed),
        ):
            if len(set(ids)) != len(ids):
                raise ValueError(f"{field_name} must not contain duplicates: {ids}.")

        selected_set = set(selected)
        missing_successful = set(successful) - selected_set
        if missing_successful:
            raise ValueError(
                "successful_client_ids contains client(s) not in selected_client_ids: "
                f"{sorted(missing_successful)}."
            )
        missing_failed = set(failed) - selected_set
        if missing_failed:
            raise ValueError(
                "failed_client_ids contains client(s) not in selected_client_ids: "
                f"{sorted(missing_failed)}."
            )
        overlap = set(successful) & set(failed)
        if overlap:
            raise ValueError(f"A client cannot be both successful and failed: {sorted(overlap)}.")

        for metrics_name, metrics in (
            ("train_metrics", self.train_metrics),
            ("evaluation_metrics", self.evaluation_metrics),
        ):
            for key, value in metrics.items():
                if not math.isfinite(float(value)):
                    raise ValueError(f"{metrics_name}['{key}'] must be finite, got {value}.")

        object.__setattr__(self, "selected_client_ids", selected)
        object.__setattr__(self, "successful_client_ids", successful)
        object.__setattr__(self, "failed_client_ids", failed)
        object.__setattr__(self, "train_metrics", MappingProxyType(dict(self.train_metrics)))
        object.__setattr__(
            self, "evaluation_metrics", MappingProxyType(dict(self.evaluation_metrics))
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


EvaluationCallback = Callable[[nn.Module, int], Mapping[str, float]]
"""Evaluates the server's current global model for one round.

Receives a safe, independent model reference (loaded with the current
global weights, but never a live reference into server internals — the
callback cannot corrupt server state through it) and the round_id being
evaluated. Must return a mapping of metric name to a finite float value.
Must not implement any dataset-specific logic in this module; the caller
supplies a callback closed over whatever evaluation dataset/logic it needs.
"""

RoundCallback = Callable[[FederatedRoundResult], None]
"""Invoked once per round, after that round's FederatedRoundResult is built."""

UpdateTransformCallback = Callable[
    [str, int, Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]],
    Mapping[str, torch.Tensor],
]
"""Optional per-client hook (Phase 5) applied to a client's local update
after honest local training has completed and ``get_model_weights()`` has
been called, but strictly before that update is wrapped into a
``ClientUpdate`` and submitted to the server for aggregation -- the
narrowest point where an honest update exists but has not yet been
accepted (``research_plans/PHASE5_Malicious_Client_Traceability_Plan.md``
Section 8).

Receives ``(client_id, round_id, honest_model_state, pre_round_global_state)``
and must return a state dict structurally compatible with
``honest_model_state`` (same keys, shapes, dtypes) -- the value actually
submitted to the server. Called for *every* selected, successfully-trained
client, honest and malicious alike, so any per-client conditional logic
(e.g. "only client_002, only round >= 20") is entirely the caller's
responsibility; the runner itself has no concept of an attack, a
malicious client, or a threat model. The default (``None``) is a strict
no-op: the honest ``model_state`` is submitted completely unchanged,
exactly as before this hook existed.

``client.train_one_round()`` and everything inside ``FederatedClient`` are
never touched by this hook -- it only ever sees the already-computed,
already-detached, already-CPU-cloned ``model_state`` dict
``get_model_weights()`` returns, so it can never contaminate any other
client's training, data, or model."""


def _serializable_runner_config(config: FederatedRunnerConfig) -> dict[str, Any]:
    """Convert a ``FederatedRunnerConfig`` into a plain, pickle-safe dict."""
    return asdict(config)


def _round_result_to_dict(result: FederatedRoundResult) -> dict[str, Any]:
    """Convert a ``FederatedRoundResult`` into a plain dict for checkpoint storage."""
    return {
        "round_id": result.round_id,
        "selected_client_ids": list(result.selected_client_ids),
        "successful_client_ids": list(result.successful_client_ids),
        "failed_client_ids": list(result.failed_client_ids),
        "aggregation_strategy": result.aggregation_strategy,
        "total_samples": result.total_samples,
        "train_metrics": dict(result.train_metrics),
        "evaluation_metrics": dict(result.evaluation_metrics),
        "round_status": result.round_status,
        "started_at_utc": result.started_at_utc,
        "completed_at_utc": result.completed_at_utc,
        "metadata": dict(result.metadata),
    }


def _round_result_from_dict(payload: dict[str, Any]) -> FederatedRoundResult:
    """Reconstruct a ``FederatedRoundResult`` from a plain dict previously
    produced by ``_round_result_to_dict``."""
    return FederatedRoundResult(
        round_id=payload["round_id"],
        selected_client_ids=tuple(payload["selected_client_ids"]),
        successful_client_ids=tuple(payload["successful_client_ids"]),
        failed_client_ids=tuple(payload["failed_client_ids"]),
        aggregation_strategy=payload["aggregation_strategy"],
        total_samples=payload["total_samples"],
        train_metrics=payload["train_metrics"],
        evaluation_metrics=payload["evaluation_metrics"],
        round_status=payload["round_status"],
        started_at_utc=payload["started_at_utc"],
        completed_at_utc=payload["completed_at_utc"],
        metadata=payload["metadata"],
    )


class FederatedRunner:
    """Orchestrates federated communication rounds over an existing client/server setup.

    # TODO (Sprint 3.5): ``config.deterministic`` is accepted and validated
    # but has no runtime effect yet (e.g. it does not seed anything before
    # client training). Whether determinism should be applied once per
    # run(), once per round, or delegated entirely to each
    # FederatedClient's own config is an open design question, deferred
    # rather than guessed.
    #
    # TODO (Sprint 3.5): ``config.checkpoint_every`` is accepted and
    # validated but ``run()`` does not automatically call
    # ``save_runner_checkpoint`` at that cadence — ``FederatedRunnerConfig``
    # has no configured checkpoint path (unlike
    # ``FederatedServerConfig.checkpoint_dir``), so it is unclear where an
    # automatic checkpoint would be written. A caller wanting per-round
    # checkpoints today can call ``save_runner_checkpoint(path)`` from a
    # ``round_callback``.
    """

    def __init__(
        self,
        server: FederatedServer,
        clients: Mapping[str, FederatedClient],
        aggregation_strategy: AggregationStrategy,
        config: FederatedRunnerConfig,
        evaluation_callback: EvaluationCallback | None = None,
        round_callback: RoundCallback | None = None,
        update_transform_callback: UpdateTransformCallback | None = None,
    ) -> None:
        """Construct a runner around already-built collaborators.

        Args:
            server: The federated server (already constructed; the runner
                never constructs one).
            clients: Every client this runner may select and train,
                keyed by client id. Must exactly match the server's
                registered client set. Must not be empty. The runner
                never constructs a ``FederatedClient``.
            aggregation_strategy: The aggregation plugin to use every
                round. The runner has no special-cased knowledge of any
                specific strategy (including ``FedAvg``).
            config: Runner-level configuration.
            evaluation_callback: Optional callback to evaluate the global
                model each round due for evaluation.
            round_callback: Optional callback invoked once per round with
                that round's ``FederatedRoundResult``.
            update_transform_callback: Optional per-client update-transform
                hook (Phase 5), applied after honest local training but
                before submission to the server. ``None`` (the default) is
                a strict no-op -- see ``UpdateTransformCallback``'s
                docstring.

        Raises:
            ValueError: If ``clients`` is empty, if any client id doesn't
                match its own ``FederatedClient.client_id``, or if the
                client id set doesn't exactly match the server's
                registered clients.
        """
        if not clients:
            raise ValueError("clients must be a non-empty mapping.")
        for client_id, client in clients.items():
            if client_id != client.client_id:
                raise ValueError(
                    f"clients key '{client_id}' does not match "
                    f"FederatedClient.client_id '{client.client_id}'."
                )

        registered = set(server.list_registered_clients())
        provided = set(clients.keys())
        if provided != registered:
            missing = sorted(registered - provided)
            extra = sorted(provided - registered)
            raise ValueError(
                "clients must exactly match the server's registered clients "
                f"(missing={missing}, extra={extra})."
            )

        self.server = server
        self.clients: dict[str, FederatedClient] = dict(clients)
        self.aggregation_strategy = aggregation_strategy
        self.config = config
        self.evaluation_callback = evaluation_callback
        self.round_callback = round_callback
        self.update_transform_callback = update_transform_callback

        self._history: list[FederatedRoundResult] = []
        self._next_round_id: int = 0

        # A scratch model, synced with the global weights before each
        # evaluation call, handed to the evaluation callback instead of
        # any live server-internal model reference. Built from an
        # arbitrary client's model_fn (all clients must share one
        # architecture for aggregation to work at all, so any one will
        # do). Only built if an evaluation callback is actually configured.
        self._evaluation_model: nn.Module | None = (
            next(iter(self.clients.values())).model_fn() if evaluation_callback is not None else None
        )

        logger.info(
            "Runner: initialized with %d client(s), strategy '%s'.",
            len(self.clients), aggregation_strategy.name,
        )

    # ------------------------------------------------------------------
    # Round execution
    # ------------------------------------------------------------------

    def run_round(self, round_id: int) -> FederatedRoundResult:
        """Run exactly one communication round end to end.

        See the module/class docstrings for the precondition that every
        client already has ``initialize_model()`` called on it.

        Args:
            round_id: The round to run. Passed through to
                ``server.start_round``, every ``ClientUpdate`` this round
                produces, and (Sprint 5.1) each selected client's
                ``load_global_weights(..., round_id=round_id)``, so a
                client configured with ``round_lr_schedule="cosine"`` can
                apply this round's learning rate before training.

        Returns:
            The ``FederatedRoundResult`` for this round (also appended to
            ``get_history()``).
        """
        started_at_utc = utc_now_iso()
        start_time = time.perf_counter()
        client_errors: dict[str, str] = {}

        def _finalize(result: FederatedRoundResult) -> FederatedRoundResult:
            self._history.append(result)
            self._next_round_id = round_id + 1
            if self.round_callback is not None:
                self.round_callback(result)
            return result

        def _elapsed() -> float:
            return time.perf_counter() - start_time

        # Step 2-3: server selects clients for this round; export pre-round weights.
        selected_client_ids = self.server.start_round(round_id)
        pre_round_global_state = self.server.get_global_weights()

        # Step 4-5: sequentially train each selected client and submit its update.
        successful_client_ids: list[str] = []
        failed_client_ids: list[str] = []
        successful_records: list[tuple[str, int, dict[str, float]]] = []
        aborted_on_client_error = False

        for client_id in selected_client_ids:
            client = self.clients.get(client_id)
            if client is None:
                # Defensive: constructor validation guarantees this cannot
                # happen unless a client was registered on the server after
                # this runner was constructed.
                error_text = f"No FederatedClient available for selected client '{client_id}'."
                failed_client_ids.append(client_id)
                client_errors[client_id] = error_text
                logger.error("Round %d: %s", round_id, error_text)
                if self.config.fail_round_on_client_error:
                    aborted_on_client_error = True
                    break
                continue

            try:
                client.load_global_weights(pre_round_global_state, round_id=round_id)
                epoch_metrics = client.train_one_round()
                last_epoch = epoch_metrics[-1]
                client_train_metrics = {
                    "loss": float(last_epoch.loss),
                    "top1": float(last_epoch.top1),
                    "top5": float(last_epoch.top5),
                }
                num_samples = len(client.train_dataset)

                submitted_model_state = client.get_model_weights()
                if self.update_transform_callback is not None:
                    submitted_model_state = self.update_transform_callback(
                        client_id, round_id, submitted_model_state, pre_round_global_state
                    )

                update = ClientUpdate.create(
                    client_id=client_id,
                    round_id=round_id,
                    model_state=submitted_model_state,
                    num_samples=num_samples,
                    train_metrics=client_train_metrics,
                    metadata={"local_epochs": len(epoch_metrics)},
                )
                self.server.receive_update(update)

                successful_client_ids.append(client_id)
                successful_records.append((client_id, num_samples, client_train_metrics))
            except Exception as exc:  # noqa: BLE001 - one client's failure must not crash the round
                failed_client_ids.append(client_id)
                client_errors[client_id] = f"{type(exc).__name__}: {exc}"
                logger.error("Round %d: client '%s' failed: %s", round_id, client_id, exc)
                if self.config.fail_round_on_client_error:
                    aborted_on_client_error = True
                    break

        if aborted_on_client_error:
            self.server.finalize_round_record(status="failed")
            return _finalize(
                FederatedRoundResult(
                    round_id=round_id,
                    selected_client_ids=selected_client_ids,
                    successful_client_ids=tuple(successful_client_ids),
                    failed_client_ids=tuple(failed_client_ids),
                    aggregation_strategy=self.aggregation_strategy.name,
                    total_samples=0,
                    train_metrics={},
                    evaluation_metrics={},
                    round_status="failed",
                    started_at_utc=started_at_utc,
                    completed_at_utc=utc_now_iso(),
                    metadata={
                        "client_errors": client_errors,
                        "reason": "client_error_with_fail_round_on_client_error",
                        "elapsed_seconds": _elapsed(),
                    },
                )
            )

        # Step 6: minimum-update check. No aggregation, no global model change.
        if not self.server.has_minimum_updates():
            self.server.finalize_round_record(status="failed")
            return _finalize(
                FederatedRoundResult(
                    round_id=round_id,
                    selected_client_ids=selected_client_ids,
                    successful_client_ids=tuple(successful_client_ids),
                    failed_client_ids=tuple(failed_client_ids),
                    aggregation_strategy=self.aggregation_strategy.name,
                    total_samples=0,
                    train_metrics={},
                    evaluation_metrics={},
                    round_status="failed",
                    started_at_utc=started_at_utc,
                    completed_at_utc=utc_now_iso(),
                    metadata={
                        "client_errors": client_errors,
                        "reason": "insufficient_updates",
                        "elapsed_seconds": _elapsed(),
                    },
                )
            )

        # Step 7: aggregation. Never assumes a specific strategy.
        received_updates = self.server.list_received_updates()
        try:
            aggregation_result = self.aggregation_strategy.aggregate(
                pre_round_global_state, received_updates
            )
        except Exception as exc:  # noqa: BLE001 - preserve pre-round global state on failure
            self.server.finalize_round_record(status="failed")
            logger.error("Round %d: aggregation failed: %s", round_id, exc)
            return _finalize(
                FederatedRoundResult(
                    round_id=round_id,
                    selected_client_ids=selected_client_ids,
                    successful_client_ids=tuple(successful_client_ids),
                    failed_client_ids=tuple(failed_client_ids),
                    aggregation_strategy=self.aggregation_strategy.name,
                    total_samples=0,
                    train_metrics={},
                    evaluation_metrics={},
                    round_status="failed",
                    started_at_utc=started_at_utc,
                    completed_at_utc=utc_now_iso(),
                    metadata={
                        "client_errors": client_errors,
                        "reason": "aggregation_error",
                        "aggregation_error": f"{type(exc).__name__}: {exc}",
                        "elapsed_seconds": _elapsed(),
                    },
                )
            )

        # Step 8: explicit global update -- the only place the global model changes.
        self.server.load_global_weights(aggregation_result.aggregated_state)

        # Step 9: finalize as completed (safe: has_minimum_updates() already confirmed True).
        self.server.finalize_round_record(status="completed")

        # Step 10: evaluation (best-effort; failure here does not undo the aggregation above).
        evaluation_metrics: dict[str, float] = {}
        round_metadata: dict[str, Any] = {
            "client_errors": client_errors,
            "aggregation_metadata": dict(aggregation_result.metadata),
        }
        if self.evaluation_callback is not None and round_id % self.config.evaluate_every == 0:
            try:
                self._evaluation_model.load_state_dict(self.server.get_global_weights())
                raw_metrics = self.evaluation_callback(self._evaluation_model, round_id)
                for key, value in raw_metrics.items():
                    numeric_value = float(value)
                    if not math.isfinite(numeric_value):
                        raise ValueError(f"evaluation metric '{key}' is not finite: {value}.")
                    evaluation_metrics[key] = numeric_value
            except Exception as exc:  # noqa: BLE001 - evaluation failure must not roll back aggregation
                evaluation_metrics = {}
                round_metadata["evaluation_error"] = f"{type(exc).__name__}: {exc}"
                logger.error("Round %d: evaluation callback failed: %s", round_id, exc)

        # Step 11: sample-weighted train-metric aggregation across successful clients.
        train_metrics = self._aggregate_train_metrics(successful_records)
        total_samples = sum(num_samples for _, num_samples, _ in successful_records)
        round_metadata["elapsed_seconds"] = _elapsed()

        result = FederatedRoundResult(
            round_id=round_id,
            selected_client_ids=selected_client_ids,
            successful_client_ids=tuple(successful_client_ids),
            failed_client_ids=tuple(failed_client_ids),
            aggregation_strategy=self.aggregation_strategy.name,
            total_samples=total_samples,
            train_metrics=train_metrics,
            evaluation_metrics=evaluation_metrics,
            round_status="completed",
            started_at_utc=started_at_utc,
            completed_at_utc=utc_now_iso(),
            metadata=round_metadata,
        )
        logger.info(
            "Round %d: completed (%d/%d successful, %.1fs).",
            round_id, len(successful_client_ids), len(selected_client_ids), _elapsed(),
        )
        # Step 12: round callback invoked via _finalize, after the result exists.
        return _finalize(result)

    @staticmethod
    def _aggregate_train_metrics(
        successful_records: list[tuple[str, int, dict[str, float]]]
    ) -> dict[str, float]:
        """Sample-weighted average of local training metrics shared by all successful clients.

        Policy: only metric keys present in *every* successful client's
        per-round metrics are included. A metric missing from even one
        successful client is dropped entirely for this round -- it is
        never imputed, defaulted, or estimated. This keeps
        ``FederatedRoundResult.train_metrics`` honest: every value in it
        reflects a metric every contributing client actually reported.

        Args:
            successful_records: ``(client_id, num_samples, train_metrics)``
                for every client whose update was accepted this round.

        Returns:
            A dict of metric name to sample-weighted average, or an empty
            dict if there were no successful clients or no shared metric
            keys.
        """
        if not successful_records:
            return {}

        common_keys = set(successful_records[0][2].keys())
        for _, _, metrics in successful_records[1:]:
            common_keys &= set(metrics.keys())

        total_samples = sum(num_samples for _, num_samples, _ in successful_records)
        if total_samples <= 0 or not common_keys:
            return {}

        return {
            key: sum(num_samples * metrics[key] for _, num_samples, metrics in successful_records)
            / total_samples
            for key in sorted(common_keys)
        }

    def run(self) -> tuple[FederatedRoundResult, ...]:
        """Run rounds from the next unfinished round through ``config.total_rounds``.

        Never reruns a round already reflected in ``get_history()`` (e.g.
        after ``load_runner_checkpoint``): it always starts from
        ``self._next_round_id``, which resume restores to one past the
        last recorded round.

        Returns:
            An immutable tuple of the ``FederatedRoundResult``s produced
            by *this* call (not the full history — use ``get_history()``
            for that, which also includes any rounds run before this call
            or restored from a checkpoint).
        """
        produced: list[FederatedRoundResult] = []
        while self._next_round_id < self.config.total_rounds:
            result = self.run_round(self._next_round_id)
            produced.append(result)
            if result.round_status != "completed" and not self.config.continue_after_failed_round:
                break
        return tuple(produced)

    def get_history(self) -> tuple[FederatedRoundResult, ...]:
        """Return every recorded ``FederatedRoundResult``, oldest first.

        Returns a fresh tuple snapshot each call; mutating the returned
        tuple's contents is impossible (tuples are immutable) and does not
        affect the runner's internal history.
        """
        return tuple(self._history)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _server_checkpoint_path(self, base_path: Path) -> Path:
        return base_path.with_name(f"{base_path.stem}_server.pt")

    def _client_checkpoint_path(self, base_path: Path, client_id: str) -> Path:
        safe_id = client_id.replace("/", "_").replace("\\", "_")
        return base_path.with_name(f"{base_path.stem}_client_{safe_id}.pt")

    def save_runner_checkpoint(self, path: Path) -> Path:
        """Atomically save this runner's state to ``path``, plus companion files.

        Writes three kinds of files:
            - ``path`` itself: runner config, round history, next round id,
              aggregation strategy name, server id, and pointers to the
              companion files below (by filename, not absolute path, so
              the whole set stays valid if the directory is moved).
            - ``<path>_server.pt``: via ``server.save_server_checkpoint``.
            - ``<path>_client_<id>.pt`` per client: via
              ``client.save_local_checkpoint``.

        No raw dataset, no full ``FederatedClient``/``FederatedServer``
        object, and no client optimizer/scheduler state beyond what each
        component's own checkpoint method already persists is serialized
        here directly -- this method only orchestrates the existing,
        component-owned checkpoint methods.

        Args:
            path: Destination path for the runner's own checkpoint file.

        Returns:
            ``path`` (for convenience/consistency with
            ``FederatedServer.save_server_checkpoint``).
        """
        path = Path(path)
        server_checkpoint_path = self._server_checkpoint_path(path)
        self.server.save_server_checkpoint(server_checkpoint_path)

        client_checkpoint_names: dict[str, str] = {}
        for client_id, client in self.clients.items():
            client_checkpoint_path = self._client_checkpoint_path(path, client_id)
            client.save_local_checkpoint(client_checkpoint_path)
            client_checkpoint_names[client_id] = client_checkpoint_path.name

        payload: dict[str, Any] = {
            "runner_config": _serializable_runner_config(self.config),
            "history": [_round_result_to_dict(result) for result in self._history],
            "next_round_id": self._next_round_id,
            "aggregation_strategy_name": self.aggregation_strategy.name,
            "server_id": self.server.config.server_id,
            "server_checkpoint_name": server_checkpoint_path.name,
            "client_ids": sorted(self.clients.keys()),
            "client_checkpoint_names": client_checkpoint_names,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }
        checkpoint_utils.save_checkpoint(payload, path)
        logger.info("Runner: saved checkpoint to %s (next_round_id=%d).", path, self._next_round_id)
        return path

    def load_runner_checkpoint(self, path: Path) -> None:
        """Load a previously saved runner checkpoint from ``path``.

        Validates aggregation-strategy name, server identity, and client
        id set compatibility before restoring anything. Delegates the
        actual server/client state restoration to their own existing
        checkpoint-loading methods.

        # TODO (Sprint 3.5): FederatedClient.load_local_checkpoint requires
        # initialize_model() to already have been called (see the module
        # docstring precondition). If a caller constructs fresh clients and
        # calls load_runner_checkpoint before initializing them, this will
        # raise a RuntimeError from within FederatedClient rather than a
        # runner-specific error. Whether the runner should take
        # responsibility for calling initialize_model() automatically is
        # deferred rather than guessed (see class docstring TODO on client
        # lifecycle ownership).

        Args:
            path: Path to a checkpoint previously written by
                ``save_runner_checkpoint``.

        Raises:
            FileNotFoundError: If ``path`` (or a required companion file)
                does not exist.
            ValueError: If the checkpoint's aggregation strategy name,
                server id, or client id set don't match this runner.
        """
        path = Path(path)
        payload = checkpoint_utils.load_checkpoint(path, map_location="cpu")

        stored_strategy_name = payload.get("aggregation_strategy_name")
        if stored_strategy_name != self.aggregation_strategy.name:
            raise ValueError(
                f"Checkpoint at {path} was produced with aggregation strategy "
                f"'{stored_strategy_name}', but this runner uses "
                f"'{self.aggregation_strategy.name}'."
            )

        stored_server_id = payload.get("server_id")
        if stored_server_id != self.server.config.server_id:
            raise ValueError(
                f"Checkpoint at {path} belongs to server '{stored_server_id}', "
                f"not '{self.server.config.server_id}'."
            )

        stored_client_ids = set(payload.get("client_ids", []))
        if stored_client_ids != set(self.clients.keys()):
            missing = sorted(stored_client_ids - set(self.clients.keys()))
            extra = sorted(set(self.clients.keys()) - stored_client_ids)
            raise ValueError(
                f"Checkpoint at {path} has an incompatible client set "
                f"(missing_here={missing}, unexpected_here={extra})."
            )

        server_checkpoint_path = path.parent / payload["server_checkpoint_name"]
        self.server.load_server_checkpoint(server_checkpoint_path)

        for client_id, checkpoint_name in payload.get("client_checkpoint_names", {}).items():
            client_checkpoint_path = path.parent / checkpoint_name
            self.clients[client_id].load_local_checkpoint(client_checkpoint_path)

        self._history = [_round_result_from_dict(item) for item in payload.get("history", [])]
        self._next_round_id = payload.get("next_round_id", 0)

        logger.info(
            "Runner: loaded checkpoint from %s (next_round_id=%d).", path, self._next_round_id
        )
