"""Lightweight tests for the federated server abstraction and contracts (Sprint 3.2).

These use a tiny linear model and synthetic client updates so the suite
runs quickly on CPU. No CIFAR-100 data is loaded, no training is run, and
no aggregation (FedAvg or otherwise) is exercised, since none exists yet.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from src.federated.contracts import ClientUpdate, RoundRecord
from src.federated.server import FederatedServer, FederatedServerConfig


class _TinyModel(nn.Module):
    """A minimal model, just for exercising state-dict key/shape checks."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_server(
    server_id: str = "server_0",
    clients_per_round: int = 2,
    minimum_updates: int = 2,
    seed: int = 42,
    strict_round_matching: bool = True,
    checkpoint_dir: Path | None = None,
) -> FederatedServer:
    """Build a small FederatedServer around _TinyModel for tests."""
    config = FederatedServerConfig(
        server_id=server_id,
        seed=seed,
        clients_per_round=clients_per_round,
        minimum_updates=minimum_updates,
        checkpoint_dir=checkpoint_dir if checkpoint_dir is not None else Path("unused"),
        strict_round_matching=strict_round_matching,
    )
    return FederatedServer(config, model_fn=_TinyModel)


def _register_clients(server: FederatedServer, count: int, prefix: str = "client") -> list[str]:
    """Register `count` clients with sequential ids and return their ids."""
    ids = [f"{prefix}_{i}" for i in range(count)]
    for client_id in ids:
        server.register_client(client_id, num_samples=10)
    return ids


def _make_update(client_id: str, round_id: int, model: nn.Module, num_samples: int = 10) -> ClientUpdate:
    """Build a valid ClientUpdate from a model's current state dict."""
    return ClientUpdate.create(
        client_id=client_id,
        round_id=round_id,
        model_state=model.state_dict(),
        num_samples=num_samples,
        train_metrics={"loss": 0.5},
    )


# ----------------------------------------------------------------------
# Contracts: ClientUpdate / RoundRecord
# ----------------------------------------------------------------------


def test_client_update_clones_and_detaches_tensors() -> None:
    """ClientUpdate stores detached, cloned CPU tensors, immune to later mutation."""
    original = torch.ones(3, requires_grad=True)
    update = ClientUpdate.create(client_id="c", round_id=0, model_state={"w": original}, num_samples=1)

    with torch.no_grad():
        original.add_(5.0)

    stored = update.model_state["w"]
    assert stored.requires_grad is False
    assert stored.device.type == "cpu"
    assert torch.equal(stored, torch.ones(3))


def test_client_update_rejects_empty_model_state() -> None:
    """An empty model_state is rejected."""
    with pytest.raises(ValueError):
        ClientUpdate.create(client_id="c", round_id=0, model_state={}, num_samples=1)


def test_client_update_rejects_non_finite_metric() -> None:
    """A non-finite training metric is rejected."""
    with pytest.raises(ValueError):
        ClientUpdate.create(
            client_id="c",
            round_id=0,
            model_state={"w": torch.zeros(1)},
            num_samples=1,
            train_metrics={"loss": float("nan")},
        )


def test_round_record_rejects_invalid_status() -> None:
    """An unrecognized status value is rejected."""
    with pytest.raises(ValueError):
        RoundRecord(
            round_id=0,
            selected_client_ids=("a",),
            received_client_ids=(),
            rejected_client_ids=(),
            status="bogus",
            started_at_utc="now",
            metadata={},
        )


def test_round_record_rejects_received_not_subset_of_selected() -> None:
    """received_client_ids must be a subset of selected_client_ids."""
    with pytest.raises(ValueError):
        RoundRecord(
            round_id=0,
            selected_client_ids=("a",),
            received_client_ids=("b",),
            rejected_client_ids=(),
            status="completed",
            started_at_utc="now",
            metadata={},
            completed_at_utc="now",
        )


# ----------------------------------------------------------------------
# FederatedServerConfig validation
# ----------------------------------------------------------------------


def test_config_validation() -> None:
    """Invalid configuration values raise ValueError at construction time."""
    with pytest.raises(ValueError):
        FederatedServerConfig(server_id="", seed=0, clients_per_round=2, minimum_updates=1, checkpoint_dir=Path("x"))
    with pytest.raises(ValueError):
        FederatedServerConfig(server_id="s", seed=-1, clients_per_round=2, minimum_updates=1, checkpoint_dir=Path("x"))
    with pytest.raises(ValueError):
        FederatedServerConfig(server_id="s", seed=0, clients_per_round=0, minimum_updates=1, checkpoint_dir=Path("x"))
    with pytest.raises(ValueError):
        FederatedServerConfig(server_id="s", seed=0, clients_per_round=2, minimum_updates=0, checkpoint_dir=Path("x"))
    with pytest.raises(ValueError):
        FederatedServerConfig(server_id="s", seed=0, clients_per_round=2, minimum_updates=3, checkpoint_dir=Path("x"))


# ----------------------------------------------------------------------
# Client registration
# ----------------------------------------------------------------------


def test_client_registration_and_duplicate_rejection() -> None:
    """Registration stores metadata and rejects duplicate ids."""
    server = _make_server()
    server.register_client("client_a", num_samples=10)
    assert server.list_registered_clients() == ("client_a",)

    with pytest.raises(ValueError):
        server.register_client("client_a", num_samples=5)
    with pytest.raises(ValueError):
        server.register_client("client_b", num_samples=0)


def test_client_unregistration() -> None:
    """Unregistration removes a client; unregistering twice raises."""
    server = _make_server()
    server.register_client("client_a", num_samples=10)
    server.unregister_client("client_a")
    assert server.list_registered_clients() == ()

    with pytest.raises(ValueError):
        server.unregister_client("client_a")


# ----------------------------------------------------------------------
# Client selection
# ----------------------------------------------------------------------


def test_deterministic_client_selection() -> None:
    """Selection depends only on seed, round id, and the registered client set."""
    server_a = _make_server(clients_per_round=2, seed=42)
    server_b = _make_server(clients_per_round=2, seed=42)
    for server in (server_a, server_b):
        _register_clients(server, 5)

    selection_a = server_a.select_clients(round_id=1)
    selection_b = server_b.select_clients(round_id=1)
    assert selection_a == selection_b
    assert len(selection_a) == 2

    # Repeating the call on the same server yields the same result.
    assert server_a.select_clients(round_id=1) == selection_a


def test_insufficient_client_selection_failure() -> None:
    """Selecting more clients than are registered fails clearly."""
    server = _make_server(clients_per_round=3)
    _register_clients(server, 2)
    with pytest.raises(ValueError):
        server.select_clients(round_id=1)


# ----------------------------------------------------------------------
# Global weights
# ----------------------------------------------------------------------


def test_global_weight_export_and_load() -> None:
    """Exported weights are safe CPU copies; loading validates compatibility."""
    server = _make_server()
    weights = server.get_global_weights()
    assert all(tensor.device.type == "cpu" for tensor in weights.values())

    original = next(iter(weights.values())).clone()
    for tensor in weights.values():
        tensor.add_(1.0)  # mutate the exported copy
    weights_again = server.get_global_weights()
    assert torch.equal(next(iter(weights_again.values())), original)  # server unaffected

    server.load_global_weights(weights)
    reloaded = server.get_global_weights()
    for key in weights:
        assert torch.equal(weights[key], reloaded[key])

    bad_weights = dict(weights)
    bad_weights["extra_key"] = torch.zeros(1)
    with pytest.raises(ValueError):
        server.load_global_weights(bad_weights)


# ----------------------------------------------------------------------
# Round lifecycle: start_round / receive_update
# ----------------------------------------------------------------------


def test_start_round_state() -> None:
    """start_round selects clients and resets round-local state; only one round at a time."""
    server = _make_server(clients_per_round=2)
    _register_clients(server, 3)

    selected = server.start_round(round_id=1)
    assert len(selected) == 2
    assert server.has_minimum_updates() is False
    assert server.list_received_updates() == ()

    with pytest.raises(RuntimeError):
        server.start_round(round_id=2)


def test_valid_update_acceptance() -> None:
    """A well-formed update from a selected client is accepted."""
    server = _make_server(clients_per_round=2, minimum_updates=1)
    ids = _register_clients(server, 2)
    server.start_round(round_id=1, selected_client_ids=ids)

    server.receive_update(_make_update(ids[0], round_id=1, model=_TinyModel()))

    received = server.list_received_updates()
    assert len(received) == 1
    assert received[0].client_id == ids[0]


def test_unselected_client_rejection() -> None:
    """An update from a client not selected this round is rejected."""
    server = _make_server(clients_per_round=1, minimum_updates=1)
    ids = _register_clients(server, 2)
    server.start_round(round_id=1, selected_client_ids=[ids[0]])

    with pytest.raises(ValueError):
        server.receive_update(_make_update(ids[1], round_id=1, model=_TinyModel()))


def test_duplicate_update_rejection() -> None:
    """A second update from the same client this round is rejected."""
    server = _make_server(clients_per_round=1, minimum_updates=1)
    ids = _register_clients(server, 1)
    server.start_round(round_id=1, selected_client_ids=ids)

    server.receive_update(_make_update(ids[0], round_id=1, model=_TinyModel()))
    with pytest.raises(ValueError):
        server.receive_update(_make_update(ids[0], round_id=1, model=_TinyModel()))


def test_mismatched_round_rejection() -> None:
    """Under strict_round_matching, an update tagged with the wrong round id is rejected."""
    server = _make_server(clients_per_round=1, minimum_updates=1, strict_round_matching=True)
    ids = _register_clients(server, 1)
    server.start_round(round_id=5, selected_client_ids=ids)

    with pytest.raises(ValueError):
        server.receive_update(_make_update(ids[0], round_id=4, model=_TinyModel()))


def test_incompatible_model_state_rejection() -> None:
    """An update whose model_state doesn't match the global model's keys/shapes is rejected."""
    server = _make_server(clients_per_round=1, minimum_updates=1)
    ids = _register_clients(server, 1)
    server.start_round(round_id=1, selected_client_ids=ids)

    wrong_model = nn.Linear(8, 999)
    update = ClientUpdate.create(
        client_id=ids[0], round_id=1, model_state=wrong_model.state_dict(), num_samples=10
    )
    with pytest.raises(ValueError):
        server.receive_update(update)


def test_minimum_update_logic() -> None:
    """has_minimum_updates() reflects only accepted updates for the active round."""
    server = _make_server(clients_per_round=2, minimum_updates=2)
    ids = _register_clients(server, 2)
    server.start_round(round_id=1, selected_client_ids=ids)

    assert server.has_minimum_updates() is False
    server.receive_update(_make_update(ids[0], round_id=1, model=_TinyModel()))
    assert server.has_minimum_updates() is False
    server.receive_update(_make_update(ids[1], round_id=1, model=_TinyModel()))
    assert server.has_minimum_updates() is True


def test_receive_update_does_not_change_global_weights() -> None:
    """Accepting an update never mutates the server's global model."""
    server = _make_server(clients_per_round=1, minimum_updates=1)
    ids = _register_clients(server, 1)
    before = server.get_global_weights()

    server.start_round(round_id=1, selected_client_ids=ids)
    server.receive_update(_make_update(ids[0], round_id=1, model=_TinyModel()))

    after = server.get_global_weights()
    for key in before:
        assert torch.equal(before[key], after[key])


# ----------------------------------------------------------------------
# Round finalization
# ----------------------------------------------------------------------


def test_successful_completed_round_record() -> None:
    """Finalizing with enough updates produces a completed RoundRecord and clears round state."""
    server = _make_server(clients_per_round=2, minimum_updates=2)
    ids = _register_clients(server, 2)
    server.start_round(round_id=1, selected_client_ids=ids)
    for client_id in ids:
        server.receive_update(_make_update(client_id, round_id=1, model=_TinyModel()))

    record = server.finalize_round_record(status="completed")
    assert record.status == "completed"
    assert record.round_id == 1
    assert set(record.received_client_ids) == set(ids)
    assert record.rejected_client_ids == ()
    assert record.completed_at_utc is not None

    assert server.get_round_history() == (record,)
    assert server.list_received_updates() == ()
    with pytest.raises(RuntimeError):
        server.finalize_round_record(status="completed")


def test_completed_finalization_fails_without_minimum_updates() -> None:
    """Finalizing as 'completed' without enough accepted updates raises."""
    server = _make_server(clients_per_round=2, minimum_updates=2)
    ids = _register_clients(server, 2)
    server.start_round(round_id=1, selected_client_ids=ids)
    server.receive_update(_make_update(ids[0], round_id=1, model=_TinyModel()))

    with pytest.raises(ValueError):
        server.finalize_round_record(status="completed")


def test_failed_round_finalization_works() -> None:
    """Finalizing as 'failed' succeeds even without minimum updates."""
    server = _make_server(clients_per_round=2, minimum_updates=2)
    ids = _register_clients(server, 2)
    server.start_round(round_id=1, selected_client_ids=ids)
    server.receive_update(_make_update(ids[0], round_id=1, model=_TinyModel()))

    record = server.finalize_round_record(status="failed")
    assert record.status == "failed"
    assert record.completed_at_utc is None
    assert server.list_received_updates() == ()


# ----------------------------------------------------------------------
# Checkpointing
# ----------------------------------------------------------------------


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """A saved server checkpoint restores registered clients, history, and global weights."""
    server = _make_server(checkpoint_dir=tmp_path)
    ids = _register_clients(server, 2)
    server.start_round(round_id=1, selected_client_ids=ids)
    server.receive_update(_make_update(ids[0], round_id=1, model=_TinyModel()))
    server.finalize_round_record(status="failed")

    checkpoint_path = tmp_path / "checkpoint.pt"
    saved_path = server.save_server_checkpoint(checkpoint_path)
    assert saved_path == checkpoint_path
    assert checkpoint_path.exists()

    restored = _make_server(checkpoint_dir=tmp_path)
    restored.load_server_checkpoint(checkpoint_path)

    assert restored.list_registered_clients() == server.list_registered_clients()
    assert len(restored.get_round_history()) == 1
    assert restored.get_round_history()[0].status == "failed"

    original_weights = server.get_global_weights()
    restored_weights = restored.get_global_weights()
    for key in original_weights:
        assert torch.equal(original_weights[key], restored_weights[key])


def test_wrong_server_id_checkpoint_rejection(tmp_path: Path) -> None:
    """Loading a checkpoint into a server with a different server_id raises."""
    owner = _make_server(server_id="server_a", checkpoint_dir=tmp_path)
    checkpoint_path = tmp_path / "checkpoint.pt"
    owner.save_server_checkpoint(checkpoint_path)

    other = _make_server(server_id="server_b", checkpoint_dir=tmp_path)
    with pytest.raises(ValueError):
        other.load_server_checkpoint(checkpoint_path)
