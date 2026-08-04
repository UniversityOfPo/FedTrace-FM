"""Lightweight tests for the federated client abstraction (Sprint 3.1).

These use tiny synthetic data and a small model so the suite runs quickly
on CPU. No real dataset is loaded, no FedAvg/Server/aggregation logic is
exercised (none exists yet), and no communication rounds between multiple
clients are simulated here.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import TensorDataset

from src.federated.client import FederatedClient, FederatedClientConfig
from src.models.resnet18_cifar import resnet18_cifar

NUM_CLASSES = 10


def _make_tiny_dataset(num_samples: int = 8, num_classes: int = NUM_CLASSES) -> TensorDataset:
    """Build a tiny synthetic image/label dataset for fast CPU tests."""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


def _make_client(client_id: str = "client_0", with_val: bool = True) -> FederatedClient:
    """Build a small, CPU-friendly FederatedClient around synthetic data."""
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
    )
    return FederatedClient(
        client_id=client_id,
        model_fn=lambda: resnet18_cifar(num_classes=NUM_CLASSES),
        train_dataset=_make_tiny_dataset(),
        config=config,
        val_dataset=_make_tiny_dataset() if with_val else None,
    )


def test_config_rejects_mismatched_client_id() -> None:
    """Constructing a client with a client_id that disagrees with its config raises."""
    config = FederatedClientConfig(client_id="client_a")
    with pytest.raises(ValueError):
        FederatedClient(
            client_id="client_b",
            model_fn=lambda: resnet18_cifar(num_classes=NUM_CLASSES),
            train_dataset=_make_tiny_dataset(),
            config=config,
        )


def test_config_validation_rejects_invalid_values() -> None:
    """Invalid configuration values raise ValueError at construction time."""
    with pytest.raises(ValueError):
        FederatedClientConfig(client_id="client_0", local_epochs=0)
    with pytest.raises(ValueError):
        FederatedClientConfig(client_id="client_0", optimizer="not_an_optimizer")


def test_methods_before_initialize_raise() -> None:
    """Calling model-dependent methods before initialize_model() raises RuntimeError."""
    client = _make_client()
    with pytest.raises(RuntimeError):
        client.train_one_round()
    with pytest.raises(RuntimeError):
        client.evaluate()
    with pytest.raises(RuntimeError):
        client.get_model_weights()


def test_client_initialization() -> None:
    """initialize_model() builds a model, optimizer, scheduler-less run, and criterion."""
    client = _make_client()
    client.initialize_model()

    assert client.device is not None
    assert client.device.type == "cpu"
    assert client.model is not None
    assert client.optimizer is not None
    assert client.scheduler is None  # scheduler="none" in _make_client's config
    assert client.criterion is not None
    assert client.current_round == 0
    assert client.round_history == []


def test_weight_export_and_load() -> None:
    """Weights exported from one client load correctly into another client's model."""
    source = _make_client(client_id="client_a")
    target = _make_client(client_id="client_b")
    source.initialize_model()
    target.initialize_model()

    weights = source.get_model_weights()
    # Exported weights must be plain CPU tensors, not live references.
    assert all(tensor.device.type == "cpu" for tensor in weights.values())

    target.load_global_weights(weights)

    for source_param, target_param in zip(source.model.parameters(), target.model.parameters()):
        assert torch.equal(source_param, target_param)


def test_reset_training_state_clears_optimizer_momentum() -> None:
    """reset_training_state() discards accumulated SGD momentum without touching weights."""
    client = _make_client()
    client.initialize_model()
    client.train_one_round()  # accumulates SGD momentum buffers

    momentum_buffer_present = any(
        client.optimizer.state.get(param, {}).get("momentum_buffer") is not None
        for group in client.optimizer.param_groups
        for param in group["params"]
    )
    assert momentum_buffer_present

    weights_before_reset = [p.detach().clone() for p in client.model.parameters()]
    client.reset_training_state()
    weights_after_reset = [p.detach().clone() for p in client.model.parameters()]

    assert all(torch.equal(b, a) for b, a in zip(weights_before_reset, weights_after_reset))
    momentum_buffer_after_reset = any(
        client.optimizer.state.get(param, {}).get("momentum_buffer") is not None
        for group in client.optimizer.param_groups
        for param in group["params"]
    )
    assert not momentum_buffer_after_reset


def test_load_global_weights_resets_training_state() -> None:
    """load_global_weights() also resets optimizer/scheduler state (the FedAvg client-state policy)."""
    client = _make_client()
    client.initialize_model()
    client.train_one_round()

    momentum_buffer_present = any(
        client.optimizer.state.get(param, {}).get("momentum_buffer") is not None
        for group in client.optimizer.param_groups
        for param in group["params"]
    )
    assert momentum_buffer_present

    client.load_global_weights(client.get_model_weights())

    momentum_buffer_after_load = any(
        client.optimizer.state.get(param, {}).get("momentum_buffer") is not None
        for group in client.optimizer.param_groups
        for param in group["params"]
    )
    assert not momentum_buffer_after_load


def test_train_one_round_updates_parameters_and_round_counter() -> None:
    """One local training round changes model parameters and advances the round counter."""
    client = _make_client()
    client.initialize_model()

    before = [p.detach().clone() for p in client.model.parameters()]
    epoch_metrics = client.train_one_round()
    after = [p.detach().clone() for p in client.model.parameters()]

    assert client.current_round == 1
    assert len(client.round_history) == 1
    assert len(epoch_metrics) == client.config.local_epochs
    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_without_val_dataset_raises() -> None:
    """evaluate() raises a clear error when no validation dataset was configured."""
    client = _make_client(with_val=False)
    client.initialize_model()
    with pytest.raises(RuntimeError):
        client.evaluate()


def test_evaluate_with_val_dataset() -> None:
    """evaluate() runs successfully and does not change model parameters."""
    client = _make_client(with_val=True)
    client.initialize_model()

    before = [p.detach().clone() for p in client.model.parameters()]
    metrics = client.evaluate()
    after = [p.detach().clone() for p in client.model.parameters()]

    assert metrics is client.last_eval_metrics
    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """A saved local checkpoint restores model/optimizer state and round id."""
    client = _make_client(client_id="client_checkpoint")
    client.initialize_model()
    client.train_one_round()

    checkpoint_path = tmp_path / "client_checkpoint.pt"
    client.save_local_checkpoint(checkpoint_path)
    assert checkpoint_path.exists()

    restored = _make_client(client_id="client_checkpoint")
    restored.initialize_model()
    restored.load_local_checkpoint(checkpoint_path)

    assert restored.current_round == client.current_round
    for original_param, restored_param in zip(client.model.parameters(), restored.model.parameters()):
        assert torch.equal(original_param, restored_param)


def test_load_checkpoint_rejects_wrong_client_id(tmp_path: Path) -> None:
    """Loading another client's checkpoint raises a clear error."""
    owner = _make_client(client_id="client_owner")
    owner.initialize_model()
    checkpoint_path = tmp_path / "owner_checkpoint.pt"
    owner.save_local_checkpoint(checkpoint_path)

    other = _make_client(client_id="client_other")
    other.initialize_model()
    with pytest.raises(ValueError):
        other.load_local_checkpoint(checkpoint_path)


def test_get_training_summary() -> None:
    """get_training_summary() reports round progress and dataset sizes."""
    client = _make_client()
    client.initialize_model()

    summary_before = client.get_training_summary()
    assert summary_before["client_id"] == client.client_id
    assert summary_before["rounds_completed"] == 0
    assert summary_before["num_train_samples"] == 8
    assert summary_before["num_val_samples"] == 8

    client.train_one_round()
    summary_after = client.get_training_summary()
    assert summary_after["rounds_completed"] == 1
    assert summary_after["latest_train_loss"] is not None
