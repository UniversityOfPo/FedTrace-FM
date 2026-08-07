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
from torch import nn
from torch.utils.data import TensorDataset

from src.federated.client import (
    FederatedClient,
    FederatedClientConfig,
    compute_round_learning_rate,
)
from src.models.resnet18_cifar import resnet18_cifar

NUM_CLASSES = 10


def _make_tiny_dataset(num_samples: int = 8, num_classes: int = NUM_CLASSES) -> TensorDataset:
    """Build a tiny synthetic image/label dataset for fast CPU tests."""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return TensorDataset(images, labels)


def _make_client(
    client_id: str = "client_0",
    with_val: bool = True,
    round_lr_schedule: str = "constant",
    min_learning_rate: float = 0.0,
    total_communication_rounds: int | None = None,
    local_training_method: str = "fedavg",
    fedprox_mu: float = 0.0,
    model_fn=None,  # noqa: ANN001 - defaults to a fresh resnet18_cifar below
) -> FederatedClient:
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
        round_lr_schedule=round_lr_schedule,
        min_learning_rate=min_learning_rate,
        total_communication_rounds=total_communication_rounds,
        local_training_method=local_training_method,
        fedprox_mu=fedprox_mu,
    )
    return FederatedClient(
        client_id=client_id,
        model_fn=model_fn if model_fn is not None else lambda: resnet18_cifar(num_classes=NUM_CLASSES),
        train_dataset=_make_tiny_dataset(),
        config=config,
        val_dataset=_make_tiny_dataset() if with_val else None,
    )


class _TinyModelWithBatchNorm(nn.Module):
    """A tiny CIFAR-shaped model with a real BatchNorm layer (Sprint 7.1 tests)."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(self.conv(x))
        x = self.pool(x).flatten(1)
        return self.fc(x)


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


# ----------------------------------------------------------------------
# Round-level learning-rate scheduling (Sprint 5.1)
# ----------------------------------------------------------------------


def test_compute_round_learning_rate_constant_schedule_preserves_base_lr() -> None:
    """The 'constant' schedule always returns base_lr, for any round_id."""
    for round_id in (0, 1, 17, 49):
        lr = compute_round_learning_rate(
            round_id=round_id, total_rounds=50, base_lr=0.01, min_lr=0.0, schedule="constant"
        )
        assert lr == pytest.approx(0.01)


def test_compute_round_learning_rate_cosine_round_zero_equals_base_lr() -> None:
    """Round 0 of a cosine round schedule starts exactly at base_lr."""
    lr = compute_round_learning_rate(
        round_id=0, total_rounds=50, base_lr=0.01, min_lr=0.0001, schedule="cosine"
    )
    assert lr == pytest.approx(0.01)


def test_compute_round_learning_rate_cosine_final_round_equals_min_lr() -> None:
    """The last round (total_rounds - 1) of a cosine schedule reaches exactly min_lr."""
    lr = compute_round_learning_rate(
        round_id=49, total_rounds=50, base_lr=0.01, min_lr=0.0001, schedule="cosine"
    )
    assert lr == pytest.approx(0.0001)


def test_compute_round_learning_rate_cosine_round_beyond_final_holds_at_min_lr() -> None:
    """A round_id at or beyond total_rounds - 1 is clamped, not extrapolated past min_lr."""
    at_final = compute_round_learning_rate(
        round_id=49, total_rounds=50, base_lr=0.01, min_lr=0.0001, schedule="cosine"
    )
    beyond_final = compute_round_learning_rate(
        round_id=99, total_rounds=50, base_lr=0.01, min_lr=0.0001, schedule="cosine"
    )
    assert beyond_final == pytest.approx(at_final)


def test_compute_round_learning_rate_cosine_midpoint_between_base_and_min() -> None:
    """The midpoint round of a cosine schedule sits strictly between min_lr and base_lr."""
    base_lr, min_lr = 0.01, 0.0
    # total_rounds=51 makes (total_rounds - 1) = 50 evenly divisible by 2,
    # so round_id=25 lands exactly on the schedule's midpoint (progress=0.5).
    midpoint_lr = compute_round_learning_rate(
        round_id=25, total_rounds=51, base_lr=base_lr, min_lr=min_lr, schedule="cosine"
    )
    assert min_lr < midpoint_lr < base_lr
    # At progress=0.5, cos(pi/2)=0, so the closed form reduces exactly to
    # the arithmetic mean of base_lr and min_lr.
    assert midpoint_lr == pytest.approx(0.5 * (base_lr + min_lr), abs=1e-9)


def test_compute_round_learning_rate_cosine_monotonically_decreasing() -> None:
    """Every later round of a cosine schedule has a learning rate no higher than an earlier one."""
    values = [
        compute_round_learning_rate(
            round_id=r, total_rounds=50, base_lr=0.01, min_lr=0.0, schedule="cosine"
        )
        for r in range(50)
    ]
    assert all(earlier >= later for earlier, later in zip(values, values[1:]))


def test_compute_round_learning_rate_is_deterministic() -> None:
    """Calling compute_round_learning_rate twice with identical inputs returns identical output."""
    kwargs = dict(round_id=13, total_rounds=50, base_lr=0.01, min_lr=0.0002, schedule="cosine")
    first = compute_round_learning_rate(**kwargs)
    second = compute_round_learning_rate(**kwargs)
    assert first == second


def test_compute_round_learning_rate_rejects_invalid_inputs() -> None:
    """Invalid round_id, schedule, or total_rounds raise ValueError."""
    with pytest.raises(ValueError):
        compute_round_learning_rate(
            round_id=-1, total_rounds=50, base_lr=0.01, min_lr=0.0, schedule="constant"
        )
    with pytest.raises(ValueError):
        compute_round_learning_rate(
            round_id=0, total_rounds=50, base_lr=0.01, min_lr=0.0, schedule="not_a_schedule"
        )
    with pytest.raises(ValueError):
        compute_round_learning_rate(
            round_id=0, total_rounds=0, base_lr=0.01, min_lr=0.0, schedule="cosine"
        )


def test_config_rejects_invalid_round_lr_schedule() -> None:
    """An unrecognized round_lr_schedule raises ValueError at construction time."""
    with pytest.raises(ValueError):
        FederatedClientConfig(client_id="client_0", round_lr_schedule="not_a_schedule")


def test_config_rejects_negative_min_learning_rate() -> None:
    """A negative min_learning_rate raises ValueError at construction time."""
    with pytest.raises(ValueError):
        FederatedClientConfig(client_id="client_0", min_learning_rate=-0.001)


def test_config_rejects_min_learning_rate_above_base_lr() -> None:
    """min_learning_rate greater than learning_rate raises ValueError."""
    with pytest.raises(ValueError):
        FederatedClientConfig(
            client_id="client_0", learning_rate=0.01, min_learning_rate=0.1
        )


def test_config_cosine_round_schedule_requires_total_communication_rounds() -> None:
    """round_lr_schedule='cosine' without a valid total_communication_rounds raises ValueError."""
    with pytest.raises(ValueError):
        FederatedClientConfig(
            client_id="client_0", round_lr_schedule="cosine", total_communication_rounds=None
        )
    with pytest.raises(ValueError):
        FederatedClientConfig(
            client_id="client_0", round_lr_schedule="cosine", total_communication_rounds=0
        )


def test_config_cosine_round_schedule_accepts_valid_configuration() -> None:
    """A well-formed cosine round-schedule configuration constructs without error."""
    config = FederatedClientConfig(
        client_id="client_0",
        round_lr_schedule="cosine",
        min_learning_rate=0.0,
        total_communication_rounds=50,
    )
    assert config.round_lr_schedule == "cosine"


def test_client_default_config_has_constant_round_lr_schedule() -> None:
    """Existing clients built with no round-LR fields default to 'constant' (no behavior change)."""
    client = _make_client()
    assert client.config.round_lr_schedule == "constant"
    assert client.config.total_communication_rounds is None


def test_load_global_weights_without_round_id_leaves_lr_unchanged() -> None:
    """Existing callers that don't pass round_id see the pre-Sprint-5.1 behavior exactly."""
    client = _make_client(
        round_lr_schedule="cosine", min_learning_rate=0.0, total_communication_rounds=50
    )
    client.initialize_model()

    client.load_global_weights(client.get_model_weights())  # no round_id

    lr = client.optimizer.param_groups[0]["lr"]
    assert lr == pytest.approx(client.config.learning_rate)


def test_load_global_weights_applies_correct_round_lr_for_cosine_schedule() -> None:
    """load_global_weights(round_id=...) applies exactly the round's computed learning rate."""
    client = _make_client(
        round_lr_schedule="cosine", min_learning_rate=0.0, total_communication_rounds=50
    )
    client.initialize_model()

    client.load_global_weights(client.get_model_weights(), round_id=0)
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(client.config.learning_rate)

    client.load_global_weights(client.get_model_weights(), round_id=49)
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(client.config.min_learning_rate)

    expected_mid = compute_round_learning_rate(
        round_id=24,
        total_rounds=50,
        base_lr=client.config.learning_rate,
        min_lr=client.config.min_learning_rate,
        schedule="cosine",
    )
    client.load_global_weights(client.get_model_weights(), round_id=24)
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(expected_mid)


def test_load_global_weights_with_constant_round_schedule_stays_at_base_lr() -> None:
    """round_lr_schedule='constant' (the default) applies the same LR at every round."""
    client = _make_client()  # round_lr_schedule="constant" by default
    client.initialize_model()

    for round_id in (0, 5, 9):
        client.load_global_weights(client.get_model_weights(), round_id=round_id)
        assert client.optimizer.param_groups[0]["lr"] == pytest.approx(client.config.learning_rate)


def test_round_lr_override_does_not_disable_optimizer_reset() -> None:
    """Applying a round-level LR still fully resets optimizer momentum (Sprint 4.1 policy intact)."""
    client = _make_client(
        round_lr_schedule="cosine", min_learning_rate=0.0, total_communication_rounds=50
    )
    client.initialize_model()
    client.train_one_round()  # accumulates SGD momentum buffers

    momentum_buffer_present = any(
        client.optimizer.state.get(param, {}).get("momentum_buffer") is not None
        for group in client.optimizer.param_groups
        for param in group["params"]
    )
    assert momentum_buffer_present

    client.reset_training_state(round_id=17)

    momentum_buffer_after_reset = any(
        client.optimizer.state.get(param, {}).get("momentum_buffer") is not None
        for group in client.optimizer.param_groups
        for param in group["params"]
    )
    assert not momentum_buffer_after_reset
    # The round-17 learning rate was still applied on top of the reset optimizer.
    expected_lr = compute_round_learning_rate(
        round_id=17,
        total_rounds=50,
        base_lr=client.config.learning_rate,
        min_lr=client.config.min_learning_rate,
        schedule="cosine",
    )
    assert client.optimizer.param_groups[0]["lr"] == pytest.approx(expected_lr)


def test_train_one_round_runs_at_the_applied_round_lr() -> None:
    """A local training round trains under whatever LR load_global_weights applied for it."""
    client = _make_client(
        round_lr_schedule="cosine", min_learning_rate=0.0, total_communication_rounds=50
    )
    client.initialize_model()

    client.load_global_weights(client.get_model_weights(), round_id=49)
    epoch_metrics = client.train_one_round()

    assert epoch_metrics[0].learning_rate == pytest.approx(0.0)


# ----------------------------------------------------------------------
# FedProx local training method (Sprint 7.1)
# ----------------------------------------------------------------------


def test_config_default_local_training_method_is_fedavg() -> None:
    """local_training_method defaults to 'fedavg' (no behavior change for existing configs)."""
    config = FederatedClientConfig(client_id="client_0")
    assert config.local_training_method == "fedavg"
    assert config.fedprox_mu == pytest.approx(0.0)


def test_config_rejects_invalid_local_training_method() -> None:
    """An unrecognized local_training_method raises ValueError at construction time."""
    with pytest.raises(ValueError):
        FederatedClientConfig(client_id="client_0", local_training_method="not_a_method")


def test_config_rejects_negative_fedprox_mu() -> None:
    """A negative fedprox_mu raises ValueError at construction time."""
    with pytest.raises(ValueError):
        FederatedClientConfig(client_id="client_0", fedprox_mu=-0.01)


def test_config_fedprox_requires_positive_mu() -> None:
    """local_training_method='fedprox' with fedprox_mu=0.0 (the default) raises ValueError."""
    with pytest.raises(ValueError):
        FederatedClientConfig(client_id="client_0", local_training_method="fedprox")
    with pytest.raises(ValueError):
        FederatedClientConfig(client_id="client_0", local_training_method="fedprox", fedprox_mu=0.0)


def test_config_fedprox_accepts_positive_mu() -> None:
    """A well-formed FedProx configuration constructs without error."""
    config = FederatedClientConfig(
        client_id="client_0", local_training_method="fedprox", fedprox_mu=0.01
    )
    assert config.local_training_method == "fedprox"
    assert config.fedprox_mu == pytest.approx(0.01)


def test_fedavg_default_ignores_fedprox_mu_and_never_snapshots() -> None:
    """fedprox_mu is inert under local_training_method='fedavg', even if set to a nonzero value."""
    client = _make_client(local_training_method="fedavg", fedprox_mu=5.0)
    client.initialize_model()

    client.load_global_weights(client.get_model_weights())

    assert client._global_reference_params is None
    # Training still completes normally -- no proximal term is ever applied.
    epoch_metrics = client.train_one_round()
    assert len(epoch_metrics) == 1


def test_global_reference_snapshot_captured_on_load_global_weights_for_fedprox() -> None:
    """load_global_weights() captures a global-parameter snapshot when using FedProx."""
    client = _make_client(local_training_method="fedprox", fedprox_mu=0.01)
    client.initialize_model()

    client.load_global_weights(client.get_model_weights())

    assert client._global_reference_params is not None
    assert len(client._global_reference_params) == sum(1 for _ in client.model.parameters())
    for snapshot_param, model_param in zip(client._global_reference_params, client.model.parameters()):
        assert torch.equal(snapshot_param, model_param)


def test_global_reference_snapshot_is_detached_not_live() -> None:
    """The captured snapshot does not change as local training subsequently updates the model."""
    client = _make_client(local_training_method="fedprox", fedprox_mu=0.01)
    client.initialize_model()
    client.load_global_weights(client.get_model_weights())

    snapshot_before = [p.clone() for p in client._global_reference_params]
    client.train_one_round()  # updates client.model's parameters via SGD

    assert all(
        torch.equal(before, after)
        for before, after in zip(snapshot_before, client._global_reference_params)
    )
    # And the snapshot must differ from the now-trained live model -- proving
    # it is an independent copy, not a reference into the same tensors.
    assert any(
        not torch.equal(snapshot_param, model_param)
        for snapshot_param, model_param in zip(client._global_reference_params, client.model.parameters())
    )


def test_global_reference_snapshot_refreshed_each_round() -> None:
    """The snapshot is retaken (not stale) on every load_global_weights() call."""
    client = _make_client(local_training_method="fedprox", fedprox_mu=0.01)
    client.initialize_model()

    client.load_global_weights(client.get_model_weights())
    first_snapshot = [p.clone() for p in client._global_reference_params]

    client.train_one_round()  # changes client.model's parameters
    new_global_weights = client.get_model_weights()
    client.load_global_weights(new_global_weights)
    second_snapshot = client._global_reference_params

    # The second snapshot must match the newly-loaded (post-training)
    # weights, not the stale first-round snapshot.
    assert any(not torch.equal(a, b) for a, b in zip(first_snapshot, second_snapshot))
    for snapshot_param, model_param in zip(second_snapshot, client.model.parameters()):
        assert torch.equal(snapshot_param, model_param)


def test_global_reference_snapshot_excludes_non_trainable_buffers() -> None:
    """The snapshot covers only trainable parameters, never BatchNorm buffers."""
    client = _make_client(
        local_training_method="fedprox",
        fedprox_mu=0.01,
        model_fn=lambda: _TinyModelWithBatchNorm(num_classes=NUM_CLASSES),
    )
    client.initialize_model()

    client.load_global_weights(client.get_model_weights())

    num_trainable_params = sum(1 for _ in client.model.parameters())
    assert len(client._global_reference_params) == num_trainable_params
    # get_model_weights() (a full state_dict) has strictly more entries
    # than the parameter-only snapshot, since it also includes
    # bn.running_mean/running_var/num_batches_tracked.
    assert len(client.get_model_weights()) > num_trainable_params


def test_fedprox_train_one_round_requires_prior_global_weights() -> None:
    """Training with FedProx before any load_global_weights() call raises a clear error."""
    client = _make_client(local_training_method="fedprox", fedprox_mu=0.01)
    client.initialize_model()

    with pytest.raises(RuntimeError):
        client.train_one_round()


def test_fedprox_train_one_round_completes_end_to_end() -> None:
    """A full FedProx local round (snapshot + proximal-regularized training) completes normally."""
    client = _make_client(local_training_method="fedprox", fedprox_mu=0.1)
    client.initialize_model()
    client.load_global_weights(client.get_model_weights())

    before = [p.detach().clone() for p in client.model.parameters()]
    epoch_metrics = client.train_one_round()
    after = [p.detach().clone() for p in client.model.parameters()]

    assert len(epoch_metrics) == 1
    assert any(not torch.equal(b, a) for b, a in zip(before, after))
