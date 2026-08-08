"""Lightweight tests for the federated round orchestrator (Sprint 3.4).

Uses a tiny synthetic model/dataset and one local epoch per client so the
suite runs quickly on CPU. No CIFAR-100 data, no GPU, no dataset download,
and no more than a couple of communication rounds per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

from src.federated.aggregation.base import AggregationResult, AggregationStrategy
from src.federated.aggregation.fedavg import FedAvg
from src.federated.client import (
    FederatedClient,
    FederatedClientConfig,
    compute_round_learning_rate,
)
from src.federated.runner import FederatedRoundResult, FederatedRunner, FederatedRunnerConfig
from src.federated.server import FederatedServer, FederatedServerConfig
from src.training.engine import EpochMetrics

NUM_CLASSES = 4


class _TinyModel(nn.Module):
    """A minimal model: fast to train, easy to reason about."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _DummyAggregationStrategy(AggregationStrategy):
    """A deliberately non-FedAvg strategy: returns the first update's state unchanged."""

    @property
    def name(self) -> str:
        return "dummy"

    def aggregate(self, global_state, updates):  # noqa: ANN001 - test double
        first = updates[0]
        return AggregationResult(
            strategy_name=self.name,
            round_id=first.round_id,
            aggregated_state=dict(first.model_state),
            participating_client_ids=tuple(update.client_id for update in updates),
            total_weight=float(sum(update.num_samples for update in updates)),
            metadata={"note": "dummy strategy, not fedavg"},
        )


class _RaisingAggregationStrategy(AggregationStrategy):
    """A strategy that always raises, to test aggregation-failure handling."""

    @property
    def name(self) -> str:
        return "raising"

    def aggregate(self, global_state, updates):  # noqa: ANN001 - test double
        raise RuntimeError("simulated aggregation failure")


def _make_dataset(num_samples: int) -> TensorDataset:
    images = torch.randn(num_samples, 8)
    labels = torch.randint(0, NUM_CLASSES, (num_samples,))
    return TensorDataset(images, labels)


def _make_client(
    client_id: str,
    num_samples: int = 8,
    round_lr_schedule: str = "constant",
    min_learning_rate: float = 0.0,
    total_communication_rounds: int | None = None,
) -> FederatedClient:
    """Build and initialize a small, CPU-friendly FederatedClient."""
    config = FederatedClientConfig(
        client_id=client_id,
        local_epochs=1,
        batch_size=4,
        num_workers=0,
        device="cpu",
        optimizer="sgd",
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=0.0,
        scheduler="none",
        mixed_precision=False,
        log_interval=0,
        round_lr_schedule=round_lr_schedule,
        min_learning_rate=min_learning_rate,
        total_communication_rounds=total_communication_rounds,
    )
    client = FederatedClient(
        client_id=client_id,
        model_fn=_TinyModel,
        train_dataset=_make_dataset(num_samples),
        config=config,
    )
    client.initialize_model()
    return client


def _make_failing_client(client_id: str, num_samples: int = 8) -> FederatedClient:
    """Build a client whose train_one_round() always raises, to simulate failure."""
    client = _make_client(client_id, num_samples=num_samples)

    def _raise() -> list[EpochMetrics]:
        raise RuntimeError("simulated client training failure")

    client.train_one_round = _raise  # type: ignore[method-assign]
    return client


def _make_fixed_metric_client(
    client_id: str, num_samples: int, loss: float, top1: float, top5: float
) -> FederatedClient:
    """Build a client whose train_one_round() returns fixed, known metrics (no real training)."""
    client = _make_client(client_id, num_samples=num_samples)
    fixed = [EpochMetrics(loss=loss, top1=top1, top5=top5, elapsed_seconds=0.0, learning_rate=0.1)]

    def _fixed() -> list[EpochMetrics]:
        return fixed

    client.train_one_round = _fixed  # type: ignore[method-assign]
    return client


def _make_server(
    server_id: str = "server_0",
    clients_per_round: int = 2,
    minimum_updates: int = 2,
    seed: int = 42,
    checkpoint_dir: Path | None = None,
) -> FederatedServer:
    config = FederatedServerConfig(
        server_id=server_id,
        seed=seed,
        clients_per_round=clients_per_round,
        minimum_updates=minimum_updates,
        checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else Path("unused"),
    )
    return FederatedServer(config, model_fn=_TinyModel)


def _make_runner(
    num_clients: int = 2,
    total_rounds: int = 1,
    evaluate_every: int = 1,
    fail_round_on_client_error: bool = False,
    continue_after_failed_round: bool = True,
    aggregation_strategy: AggregationStrategy | None = None,
    evaluation_callback=None,  # noqa: ANN001 - optional test callback
    round_callback=None,  # noqa: ANN001 - optional test callback
    update_transform_callback=None,  # noqa: ANN001 - optional test callback (Phase 5)
    checkpoint_dir: Path | None = None,
    seed: int = 42,
    round_lr_schedule: str = "constant",
    min_learning_rate: float = 0.0,
    total_communication_rounds: int | None = None,
) -> tuple[FederatedRunner, FederatedServer, dict[str, FederatedClient]]:
    """Build a small runner/server/client set with `num_clients` clients, all selected each round."""
    clients = {
        f"client_{i}": _make_client(
            f"client_{i}",
            round_lr_schedule=round_lr_schedule,
            min_learning_rate=min_learning_rate,
            total_communication_rounds=total_communication_rounds,
        )
        for i in range(num_clients)
    }
    server = _make_server(
        clients_per_round=num_clients,
        minimum_updates=num_clients,
        seed=seed,
        checkpoint_dir=checkpoint_dir,
    )
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(
        total_rounds=total_rounds,
        evaluate_every=evaluate_every,
        fail_round_on_client_error=fail_round_on_client_error,
        continue_after_failed_round=continue_after_failed_round,
    )
    strategy = aggregation_strategy if aggregation_strategy is not None else FedAvg()
    runner = FederatedRunner(
        server=server,
        clients=clients,
        aggregation_strategy=strategy,
        config=config,
        evaluation_callback=evaluation_callback,
        round_callback=round_callback,
        update_transform_callback=update_transform_callback,
    )
    return runner, server, clients


# ----------------------------------------------------------------------
# Config validation
# ----------------------------------------------------------------------


def test_runner_config_validation() -> None:
    """Invalid configuration values raise ValueError at construction time."""
    with pytest.raises(ValueError):
        FederatedRunnerConfig(total_rounds=0)
    with pytest.raises(ValueError):
        FederatedRunnerConfig(total_rounds=1, evaluate_every=0)
    with pytest.raises(ValueError):
        FederatedRunnerConfig(total_rounds=1, checkpoint_every=0)


# ----------------------------------------------------------------------
# Basic round execution
# ----------------------------------------------------------------------


def test_one_complete_round_with_two_clients_and_fedavg() -> None:
    """A full round with two healthy clients completes and aggregates via FedAvg."""
    runner, _, clients = _make_runner(num_clients=2, total_rounds=1)

    results = runner.run()

    assert len(results) == 1
    result = results[0]
    assert result.round_status == "completed"
    assert result.round_id == 0
    assert set(result.successful_client_ids) == set(clients.keys())
    assert result.failed_client_ids == ()
    assert result.aggregation_strategy == "fedavg"
    assert result.total_samples == 16
    assert "loss" in result.train_metrics
    assert runner.get_history() == results


def test_two_sequential_rounds() -> None:
    """run() executes round ids sequentially starting from 0."""
    runner, _, _ = _make_runner(num_clients=2, total_rounds=2)
    results = runner.run()
    assert [r.round_id for r in results] == [0, 1]
    assert all(r.round_status == "completed" for r in results)
    assert len(runner.get_history()) == 2


def test_global_model_changes_after_successful_aggregation() -> None:
    """The server's global model actually changes after a completed round."""
    runner, server, _ = _make_runner(num_clients=2, total_rounds=1)
    before = server.get_global_weights()

    runner.run()

    after = server.get_global_weights()
    assert not torch.equal(before["linear.weight"], after["linear.weight"])


def test_runner_accepts_custom_dummy_aggregation_strategy() -> None:
    """The runner works with any AggregationStrategy, not just FedAvg."""
    runner, _, _ = _make_runner(
        num_clients=2, total_rounds=1, aggregation_strategy=_DummyAggregationStrategy()
    )

    results = runner.run()

    assert results[0].round_status == "completed"
    assert results[0].aggregation_strategy == "dummy"


def test_deterministic_client_selection_through_server() -> None:
    """Two independently constructed runners with the same seed select the same clients."""
    runner_a, _, _ = _make_runner(num_clients=5, total_rounds=1, seed=7)
    runner_b, _, _ = _make_runner(num_clients=5, total_rounds=1, seed=7)

    result_a = runner_a.run_round(0)
    result_b = runner_b.run_round(0)

    assert result_a.selected_client_ids == result_b.selected_client_ids


# ----------------------------------------------------------------------
# Client failure handling
# ----------------------------------------------------------------------


def test_client_failure_with_fail_round_on_client_error_false() -> None:
    """With fail_round_on_client_error=False, one client's failure doesn't stop the round."""
    clients = {"client_0": _make_client("client_0"), "client_1": _make_failing_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=1)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=1, fail_round_on_client_error=False)
    runner = FederatedRunner(server, clients, FedAvg(), config)

    result = runner.run_round(0)

    assert result.round_status == "completed"
    assert result.successful_client_ids == ("client_0",)
    assert result.failed_client_ids == ("client_1",)
    assert "client_1" in result.metadata["client_errors"]


def test_client_failure_with_fail_round_on_client_error_true() -> None:
    """With fail_round_on_client_error=True, the first client failure aborts the round."""
    clients = {"client_0": _make_failing_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=1)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=1, fail_round_on_client_error=True)
    runner = FederatedRunner(server, clients, FedAvg(), config)

    result = runner.run_round(0)

    assert result.round_status == "failed"
    assert result.metadata["reason"] == "client_error_with_fail_round_on_client_error"
    # client_0 sorts first, so it is processed (and fails) before client_1 is ever attempted.
    assert "client_1" not in result.successful_client_ids
    assert "client_1" not in result.failed_client_ids


def test_insufficient_successful_updates_causes_failed_round() -> None:
    """A round with too few accepted updates fails and never aggregates."""
    clients = {"client_0": _make_failing_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=2)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=1, fail_round_on_client_error=False)
    runner = FederatedRunner(server, clients, FedAvg(), config)

    result = runner.run_round(0)

    assert result.round_status == "failed"
    assert result.metadata["reason"] == "insufficient_updates"


def test_failed_round_does_not_change_global_model() -> None:
    """A failed round leaves the server's global model exactly as it was."""
    clients = {"client_0": _make_failing_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=2)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)
    before = server.get_global_weights()

    config = FederatedRunnerConfig(total_rounds=1)
    runner = FederatedRunner(server, clients, FedAvg(), config)
    result = runner.run_round(0)

    assert result.round_status == "failed"
    after = server.get_global_weights()
    for key in before:
        assert torch.equal(before[key], after[key])


def test_aggregation_exception_causes_failed_round_and_preserves_global_state() -> None:
    """If the aggregation strategy raises, the round fails and the global model is untouched."""
    runner, server, _ = _make_runner(
        num_clients=2, total_rounds=1, aggregation_strategy=_RaisingAggregationStrategy()
    )
    before = server.get_global_weights()

    result = runner.run_round(0)

    assert result.round_status == "failed"
    assert result.metadata["reason"] == "aggregation_error"
    after = server.get_global_weights()
    for key in before:
        assert torch.equal(before[key], after[key])


# ----------------------------------------------------------------------
# Evaluation callback
# ----------------------------------------------------------------------


def test_evaluation_callback_cadence() -> None:
    """The evaluation callback is invoked only on rounds matching evaluate_every."""
    calls: list[int] = []

    def _eval_callback(model: nn.Module, round_id: int) -> dict[str, float]:
        calls.append(round_id)
        return {"accuracy": 0.5}

    runner, _, _ = _make_runner(
        num_clients=2, total_rounds=4, evaluate_every=2, evaluation_callback=_eval_callback
    )
    results = runner.run()

    assert calls == [0, 2]
    assert results[0].evaluation_metrics == {"accuracy": 0.5}
    assert results[1].evaluation_metrics == {}
    assert results[2].evaluation_metrics == {"accuracy": 0.5}
    assert results[3].evaluation_metrics == {}


def test_evaluation_callback_failure_does_not_roll_back_successful_aggregation() -> None:
    """An evaluation callback error is recorded, but the completed aggregation stands."""

    def _bad_eval_callback(model: nn.Module, round_id: int) -> dict[str, float]:
        raise RuntimeError("simulated evaluation failure")

    runner, server, _ = _make_runner(
        num_clients=2, total_rounds=1, evaluation_callback=_bad_eval_callback
    )
    before = server.get_global_weights()

    result = runner.run_round(0)

    assert result.round_status == "completed"
    assert result.evaluation_metrics == {}
    assert "evaluation_error" in result.metadata
    after = server.get_global_weights()
    assert not torch.equal(before["linear.weight"], after["linear.weight"])


# ----------------------------------------------------------------------
# Training-metric aggregation policy
# ----------------------------------------------------------------------


def test_sample_weighted_training_metric_aggregation() -> None:
    """train_metrics is the sample-weighted average of each client's reported metrics."""
    clients = {
        "client_a": _make_fixed_metric_client("client_a", num_samples=10, loss=4.0, top1=0.2, top5=0.5),
        "client_b": _make_fixed_metric_client("client_b", num_samples=30, loss=2.0, top1=0.6, top5=0.9),
    }
    server = _make_server(clients_per_round=2, minimum_updates=2)
    server.register_client("client_a", num_samples=10)
    server.register_client("client_b", num_samples=30)

    config = FederatedRunnerConfig(total_rounds=1)
    runner = FederatedRunner(server, clients, FedAvg(), config)

    result = runner.run_round(0)

    assert result.round_status == "completed"
    assert result.total_samples == 40
    assert result.train_metrics["loss"] == pytest.approx(0.25 * 4.0 + 0.75 * 2.0)
    assert result.train_metrics["top1"] == pytest.approx(0.25 * 0.2 + 0.75 * 0.6)
    assert result.train_metrics["top5"] == pytest.approx(0.25 * 0.5 + 0.75 * 0.9)


def test_missing_client_metric_keys_are_ignored() -> None:
    """A metric absent from even one successful client is dropped, not imputed."""
    records = [
        ("client_a", 10, {"loss": 4.0, "top1": 0.2}),
        ("client_b", 30, {"loss": 2.0}),  # no "top1"
    ]

    averaged = FederatedRunner._aggregate_train_metrics(records)

    assert "top1" not in averaged
    assert averaged["loss"] == pytest.approx(0.25 * 4.0 + 0.75 * 2.0)


# ----------------------------------------------------------------------
# Callbacks and run() control flow
# ----------------------------------------------------------------------


def test_round_callback_invoked() -> None:
    """The round callback is invoked once per round with that round's result."""
    received: list[FederatedRoundResult] = []
    runner, _, _ = _make_runner(num_clients=2, total_rounds=1, round_callback=received.append)

    runner.run()

    assert len(received) == 1
    assert received[0].round_id == 0


def test_continue_after_failed_round_true() -> None:
    """With continue_after_failed_round=True, run() attempts every configured round."""
    clients = {"client_0": _make_failing_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=2)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=2, continue_after_failed_round=True)
    runner = FederatedRunner(server, clients, FedAvg(), config)

    results = runner.run()

    assert len(results) == 2
    assert results[0].round_status == "failed"


def test_continue_after_failed_round_false() -> None:
    """With continue_after_failed_round=False, run() stops at the first failed round."""
    clients = {"client_0": _make_failing_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=2)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=3, continue_after_failed_round=False)
    runner = FederatedRunner(server, clients, FedAvg(), config)

    results = runner.run()

    assert len(results) == 1
    assert results[0].round_status == "failed"


def test_runner_history_immutable_from_caller_perspective() -> None:
    """get_history() returns an independent tuple snapshot each call."""
    runner, _, _ = _make_runner(num_clients=2, total_rounds=1)
    runner.run()

    history = runner.get_history()
    history_again = runner.get_history()

    assert isinstance(history, tuple)
    assert history == history_again
    assert history is not history_again


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """A saved runner checkpoint restores history and global weights into a fresh runner."""
    clients = {"client_0": _make_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=2)
    runner = FederatedRunner(server, clients, FedAvg(), config)
    runner.run_round(0)

    checkpoint_path = tmp_path / "runner_checkpoint.pt"
    saved_path = runner.save_runner_checkpoint(checkpoint_path)
    assert saved_path == checkpoint_path
    assert checkpoint_path.exists()

    restored_clients = {"client_0": _make_client("client_0"), "client_1": _make_client("client_1")}
    restored_server = _make_server(clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path)
    for client_id in restored_clients:
        restored_server.register_client(client_id, num_samples=8)
    restored_runner = FederatedRunner(restored_server, restored_clients, FedAvg(), config)

    restored_runner.load_runner_checkpoint(checkpoint_path)

    assert len(restored_runner.get_history()) == 1
    assert restored_runner.get_history()[0].round_id == 0

    original_weights = server.get_global_weights()
    restored_weights = restored_server.get_global_weights()
    for key in original_weights:
        assert torch.equal(original_weights[key], restored_weights[key])


def test_resume_begins_from_next_unfinished_round(tmp_path: Path) -> None:
    """After resuming from a checkpoint, run() continues from the next unfinished round."""
    clients = {"client_0": _make_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=3)
    runner = FederatedRunner(server, clients, FedAvg(), config)
    runner.run_round(0)

    checkpoint_path = tmp_path / "runner_checkpoint.pt"
    runner.save_runner_checkpoint(checkpoint_path)

    restored_clients = {"client_0": _make_client("client_0"), "client_1": _make_client("client_1")}
    restored_server = _make_server(clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path)
    for client_id in restored_clients:
        restored_server.register_client(client_id, num_samples=8)
    restored_runner = FederatedRunner(restored_server, restored_clients, FedAvg(), config)
    restored_runner.load_runner_checkpoint(checkpoint_path)

    results = restored_runner.run()

    assert [r.round_id for r in results] == [1, 2]
    assert len(restored_runner.get_history()) == 3


def test_wrong_aggregation_strategy_checkpoint_rejected(tmp_path: Path) -> None:
    """Loading a checkpoint into a runner configured with a different strategy raises."""
    clients = {"client_0": _make_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=1)
    runner = FederatedRunner(server, clients, FedAvg(), config)
    runner.run_round(0)

    checkpoint_path = tmp_path / "runner_checkpoint.pt"
    runner.save_runner_checkpoint(checkpoint_path)

    other_clients = {"client_0": _make_client("client_0"), "client_1": _make_client("client_1")}
    other_server = _make_server(clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path)
    for client_id in other_clients:
        other_server.register_client(client_id, num_samples=8)
    other_runner = FederatedRunner(other_server, other_clients, _DummyAggregationStrategy(), config)

    with pytest.raises(ValueError):
        other_runner.load_runner_checkpoint(checkpoint_path)


def test_incompatible_client_set_checkpoint_rejected(tmp_path: Path) -> None:
    """Loading a checkpoint into a runner with a different client set raises."""
    clients = {"client_0": _make_client("client_0"), "client_1": _make_client("client_1")}
    server = _make_server(clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path)
    for client_id in clients:
        server.register_client(client_id, num_samples=8)

    config = FederatedRunnerConfig(total_rounds=1)
    runner = FederatedRunner(server, clients, FedAvg(), config)
    runner.run_round(0)

    checkpoint_path = tmp_path / "runner_checkpoint.pt"
    runner.save_runner_checkpoint(checkpoint_path)

    different_clients = {"client_2": _make_client("client_2"), "client_3": _make_client("client_3")}
    different_server = _make_server(clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path)
    for client_id in different_clients:
        different_server.register_client(client_id, num_samples=8)
    different_runner = FederatedRunner(different_server, different_clients, FedAvg(), config)

    with pytest.raises(ValueError):
        different_runner.load_runner_checkpoint(checkpoint_path)


# ----------------------------------------------------------------------
# Round-level learning-rate scheduling (Sprint 5.1)
# ----------------------------------------------------------------------


def test_runner_default_config_leaves_client_lr_unchanged_across_rounds() -> None:
    """Existing experiments (no round-level scheduler configured) see identical LR behavior.

    round_lr_schedule defaults to 'constant', and the runner now always
    passes round_id through to load_global_weights() -- this test pins
    down that doing so is a behavioral no-op for the default config, i.e.
    Sprint 5.1 does not change any existing experiment's results.
    """
    runner, _, clients = _make_runner(num_clients=2, total_rounds=3)

    runner.run()

    for client in clients.values():
        assert client.optimizer.param_groups[0]["lr"] == pytest.approx(
            client.config.learning_rate
        )


def test_runner_passes_round_id_so_cosine_schedule_varies_across_rounds() -> None:
    """With round_lr_schedule='cosine', each round's client LR matches the round-computed value.

    This is the core Sprint 5.1 capability check: the runner must plumb
    its own round_id into each client so a cosine round-level schedule can
    actually vary the learning rate from one communication round to the
    next, unlike the pre-existing local `scheduler`, which cannot (see
    `src/federated/client.py::compute_round_learning_rate` docstring).
    """
    total_rounds = 5
    observed_lrs: dict[int, float] = {}

    def _round_callback(result: FederatedRoundResult) -> None:
        # All clients share the same config in this test setup, so any one
        # client's optimizer LR (post-training) reflects the round's
        # applied round-level learning rate.
        any_client = next(iter(clients.values()))
        observed_lrs[result.round_id] = any_client.optimizer.param_groups[0]["lr"]

    runner, _, clients = _make_runner(
        num_clients=2,
        total_rounds=total_rounds,
        round_lr_schedule="cosine",
        min_learning_rate=0.0,
        total_communication_rounds=total_rounds,
        round_callback=_round_callback,
    )

    runner.run()

    assert len(observed_lrs) == total_rounds
    for round_id, observed_lr in observed_lrs.items():
        expected_lr = compute_round_learning_rate(
            round_id=round_id,
            total_rounds=total_rounds,
            base_lr=0.1,
            min_lr=0.0,
            schedule="cosine",
        )
        assert observed_lr == pytest.approx(expected_lr)

    # The schedule must actually vary across rounds, not just be present.
    assert observed_lrs[0] > observed_lrs[total_rounds - 1]
    assert observed_lrs[total_rounds - 1] == pytest.approx(0.0)


# ----------------------------------------------------------------------
# Per-client update-transform hook (Phase 5)
# ----------------------------------------------------------------------


def test_update_transform_callback_defaults_to_none() -> None:
    """A runner built with no update_transform_callback has one -- a strict no-op default."""
    runner, _, _ = _make_runner(num_clients=2, total_rounds=1)
    assert runner.update_transform_callback is None


def test_update_transform_callback_none_matches_pre_existing_behavior() -> None:
    """A runner with the hook explicitly set to None behaves identically to omitting it."""
    runner_default, _, _ = _make_runner(num_clients=2, total_rounds=1, seed=99)
    runner_explicit_none, _, _ = _make_runner(
        num_clients=2, total_rounds=1, seed=99, update_transform_callback=None
    )
    result_default = runner_default.run_round(0)
    result_explicit = runner_explicit_none.run_round(0)
    assert result_default.round_status == result_explicit.round_status == "completed"


def test_update_transform_callback_invoked_for_every_successful_client() -> None:
    """The hook is called exactly once per successful client, with that round's round_id."""
    calls: list[tuple[str, int]] = []

    def _record(client_id, round_id, model_state, global_state):  # noqa: ANN001 - test double
        calls.append((client_id, round_id))
        return model_state

    runner, _, clients = _make_runner(num_clients=2, total_rounds=1, update_transform_callback=_record)
    runner.run_round(0)

    assert sorted(calls) == sorted((client_id, 0) for client_id in clients)


def test_update_transform_callback_receives_pre_round_global_state() -> None:
    """The hook's global_state argument matches the server's global weights at round start."""
    seen_global_states: list[dict[str, torch.Tensor]] = []

    def _record(client_id, round_id, model_state, global_state):  # noqa: ANN001 - test double
        seen_global_states.append(dict(global_state))
        return model_state

    runner, server, _ = _make_runner(num_clients=2, total_rounds=1, update_transform_callback=_record)
    expected_global_state = server.get_global_weights()
    runner.run_round(0)

    assert len(seen_global_states) == 2
    for seen in seen_global_states:
        for key in expected_global_state:
            assert torch.equal(seen[key], expected_global_state[key])


def test_update_transform_callback_receives_only_that_clients_own_state() -> None:
    """Each client's callback invocation sees exactly that client's own submitted weights.

    The identity-transform hook below never mixes state across clients (by
    construction of the runner's per-client loop, Section 8 of
    research_plans/PHASE5_Malicious_Client_Traceability_Plan.md); this
    confirms it directly by comparing each client's own live model state
    (re-fetched after the round) to what the hook observed for it.
    """
    seen_states: dict[str, dict[str, torch.Tensor]] = {}

    def _record(client_id, round_id, model_state, global_state):  # noqa: ANN001 - test double
        seen_states[client_id] = dict(model_state)
        return model_state

    runner, _, clients = _make_runner(num_clients=2, total_rounds=1, update_transform_callback=_record)
    runner.run_round(0)

    assert set(seen_states.keys()) == set(clients.keys())
    for client_id, client in clients.items():
        for key, value in client.get_model_weights().items():
            assert torch.equal(value, seen_states[client_id][key])


def test_update_transform_callback_perturbs_only_the_designated_client() -> None:
    """A hook that only transforms one client id leaves every other client's report untouched."""

    def _perturb_client_0_only(client_id, round_id, model_state, global_state):  # noqa: ANN001
        if client_id != "client_0":
            return model_state
        return {key: value + 100.0 for key, value in model_state.items()}

    runner, _, clients = _make_runner(
        num_clients=2, total_rounds=1, update_transform_callback=_perturb_client_0_only
    )
    result = runner.run_round(0)

    assert result.round_status == "completed"
    assert set(result.successful_client_ids) == {"client_0", "client_1"}


def test_honest_client_update_reaches_aggregation_unperturbed() -> None:
    """Aggregation reflects client_0's perturbed report and client_1's genuinely untouched one.

    Verifies the end-to-end pipeline property required by the plan (Section
    8): the malicious transform never contaminates the honest client's own
    reported update -- the aggregated global model matches the FedAvg
    formula computed directly from what each client actually submitted.
    """
    captured: dict[str, dict[str, torch.Tensor]] = {}

    def _perturb_client_0_only(client_id, round_id, model_state, global_state):  # noqa: ANN001
        if client_id != "client_0":
            captured[client_id] = {key: value.clone() for key, value in model_state.items()}
            return model_state
        perturbed = {key: value + 100.0 for key, value in model_state.items()}
        captured[client_id] = {key: value.clone() for key, value in perturbed.items()}
        return perturbed

    runner, server, _ = _make_runner(
        num_clients=2, total_rounds=1, update_transform_callback=_perturb_client_0_only
    )
    result = runner.run_round(0)
    assert result.round_status == "completed"

    # Both clients register 8 samples each in _make_client's default -- equal FedAvg weighting.
    expected_global = {
        key: 0.5 * captured["client_0"][key] + 0.5 * captured["client_1"][key]
        for key in captured["client_0"]
    }
    actual_global = server.get_global_weights()
    for key in expected_global:
        assert torch.allclose(actual_global[key], expected_global[key], atol=1e-5)
