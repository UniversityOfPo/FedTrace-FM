"""Lightweight tests for the federated CIFAR-100 experiment (Sprints 4.1, 6.1).

Uses tiny synthetic datasets and a tiny model throughout, exercising the
same reusable components (`FederatedClient`, `FederatedServer`, `FedAvg`,
`FederatedRunner`, `partition_dataset_iid`/`partition_dataset_dirichlet`)
and the experiment script's config loader, `select_partitions()` dispatch,
and output-writing helpers -- without ever calling `src.data.cifar100`, so
no CIFAR-100 download can occur here.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.federated_partition import partition_dataset_iid
from src.federated.aggregation.fedavg import FedAvg
from src.federated.client import FederatedClient, FederatedClientConfig
from src.federated.runner import FederatedRoundResult, FederatedRunner, FederatedRunnerConfig
from src.federated.server import FederatedServer, FederatedServerConfig
from src.training.engine import evaluate
from src.experiments.run_federated_cifar100 import (
    ConfigError,
    load_federated_experiment_config,
    save_client_partition_outputs,
    save_confusion_matrix_plot,
    save_per_class_accuracy_csv,
    save_round_duration_plot,
    save_round_history_csv,
    select_partitions,
    _line_plot,
    _round_result_row,
    _save_json,
)

NUM_CLASSES = 4
SMOKE_CONFIG_PATH = Path("configs/cifar100_fedavg_iid_smoke.yaml")
FULL_CONFIG_PATH = Path("configs/cifar100_fedavg_iid.yaml")
EXP_004_CONFIG_PATH = Path("configs/cifar100_fedavg_iid_50rounds.yaml")
EXP_005_COSINE_CONFIG_PATH = Path("configs/cifar100_fedavg_iid_50rounds_cosine.yaml")
EXP_006_DIRICHLET_CONFIG_PATH = Path("configs/cifar100_fedavg_dirichlet_a05_50rounds.yaml")


class _TinyModel(nn.Module):
    """A minimal model: fast to train, easy to reason about."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(8, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _make_dataset(num_samples: int) -> TensorDataset:
    images = torch.randn(num_samples, 8)
    labels = torch.randint(0, NUM_CLASSES, (num_samples,))
    return TensorDataset(images, labels)


def _make_client(
    client_id: str,
    dataset,
    local_training_method: str = "fedavg",
    fedprox_mu: float = 0.0,
) -> FederatedClient:
    config = FederatedClientConfig(
        client_id=client_id, local_epochs=1, batch_size=4, num_workers=0, device="cpu",
        optimizer="sgd", learning_rate=0.05, momentum=0.9, weight_decay=0.0,
        scheduler="none", mixed_precision=False, log_interval=0,
        local_training_method=local_training_method, fedprox_mu=fedprox_mu,
    )
    client = FederatedClient(client_id=client_id, model_fn=_TinyModel, train_dataset=dataset, config=config)
    client.initialize_model()
    return client


# ----------------------------------------------------------------------
# Experiment config validation (real repo configs; no dataset is touched)
# ----------------------------------------------------------------------


def test_smoke_and_full_configs_load_successfully() -> None:
    """The two committed experiment configs parse and validate cleanly."""
    smoke = load_federated_experiment_config(SMOKE_CONFIG_PATH)
    full = load_federated_experiment_config(FULL_CONFIG_PATH)

    assert smoke.experiment_id == "EXP_002_SMOKE"
    assert smoke.num_clients == 5
    assert smoke.clients_per_round == 2
    assert full.experiment_id == "EXP_002"
    assert full.clients_per_round == 5
    assert full.partition == "iid"
    assert full.aggregation == "fedavg"


def test_experiment_config_rejects_missing_field(tmp_path: Path) -> None:
    """A config missing a required field raises ConfigError."""
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text("experiment_id: X\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_minimum_updates_above_clients_per_round(tmp_path: Path) -> None:
    """minimum_updates > clients_per_round is rejected with a clear error."""
    text = SMOKE_CONFIG_PATH.read_text(encoding="utf-8").replace("minimum_updates: 2", "minimum_updates: 99")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_unsupported_partition(tmp_path: Path) -> None:
    """A partition value other than 'iid'/'dirichlet' is rejected (shard is not implemented yet)."""
    text = SMOKE_CONFIG_PATH.read_text(encoding="utf-8").replace("partition: iid", "partition: shard")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


# ----------------------------------------------------------------------
# Dirichlet Non-IID partitioning plumbing (Sprint 6.1)
# ----------------------------------------------------------------------


def test_dirichlet_config_loads_successfully() -> None:
    """EXP_006's Dirichlet(alpha=0.5) config parses and validates cleanly."""
    config = load_federated_experiment_config(EXP_006_DIRICHLET_CONFIG_PATH)
    assert config.partition == "dirichlet"
    assert config.dirichlet_alpha == pytest.approx(0.5)
    assert config.experiment_id == "EXP_006"
    assert config.communication_rounds == 50


def test_experiment_config_rejects_dirichlet_without_alpha(tmp_path: Path) -> None:
    """partition='dirichlet' with no dirichlet_alpha field raises ConfigError before training starts."""
    text = SMOKE_CONFIG_PATH.read_text(encoding="utf-8").replace("partition: iid", "partition: dirichlet")
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_non_positive_dirichlet_alpha(tmp_path: Path) -> None:
    """dirichlet_alpha <= 0 raises ConfigError before training starts."""
    text = SMOKE_CONFIG_PATH.read_text(encoding="utf-8").replace("partition: iid", "partition: dirichlet")
    text += "\ndirichlet_alpha: 0\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_select_partitions_uses_iid_for_iid_config() -> None:
    """select_partitions() dispatches to partition_dataset_iid when partition='iid'."""
    config = load_federated_experiment_config(EXP_004_CONFIG_PATH)  # partition: iid
    dataset = _make_dataset(40)

    partitions = select_partitions(config, dataset)

    assert set(partitions.keys()) == {f"client_{i:03d}" for i in range(config.num_clients)}
    assert sum(len(subset) for subset in partitions.values()) == 40


def test_select_partitions_uses_dirichlet_for_dirichlet_config() -> None:
    """select_partitions() dispatches to partition_dataset_dirichlet when partition='dirichlet'."""
    base_config = load_federated_experiment_config(EXP_004_CONFIG_PATH)
    # A tiny batch_size keeps the internal min_samples_per_client
    # (10 * batch_size) small enough for this test's tiny synthetic dataset.
    config = replace(base_config, partition="dirichlet", dirichlet_alpha=0.5, batch_size=2, num_clients=4)
    dataset = _make_dataset(200)
    labels = [int(label) for label in dataset.tensors[1].tolist()]

    partitions = select_partitions(config, dataset, labels=labels)

    assert set(partitions.keys()) == {"client_000", "client_001", "client_002", "client_003"}
    assert sum(len(subset) for subset in partitions.values()) == 200
    # min_samples_per_client = 10 * batch_size = 20; every client must meet it.
    assert all(len(subset) >= 20 for subset in partitions.values())


def test_select_partitions_requires_labels_for_dirichlet() -> None:
    """select_partitions() raises a clear error if 'dirichlet' is selected without labels."""
    base_config = load_federated_experiment_config(EXP_004_CONFIG_PATH)
    config = replace(base_config, partition="dirichlet", dirichlet_alpha=0.5)
    dataset = _make_dataset(200)

    with pytest.raises(ValueError):
        select_partitions(config, dataset, labels=None)


# ----------------------------------------------------------------------
# Round-level learning-rate schedule plumbing (Sprint 5.2)
# ----------------------------------------------------------------------


def test_exp004_config_loads_with_backward_compatible_round_lr_defaults() -> None:
    """EXP_004's config predates round_lr_schedule/min_learning_rate and still loads cleanly.

    Both fields must fall back to their documented backward-compatible
    defaults ("constant" / 0.0) rather than raising a missing-field error.
    """
    config = load_federated_experiment_config(EXP_004_CONFIG_PATH)
    assert config.round_lr_schedule == "constant"
    assert config.min_learning_rate == pytest.approx(0.0)


def test_exp005_cosine_config_loads_with_cosine_schedule_and_min_lr() -> None:
    """EXP_005's config is read with round_lr_schedule='cosine' and min_learning_rate=0.0001."""
    config = load_federated_experiment_config(EXP_005_COSINE_CONFIG_PATH)
    assert config.round_lr_schedule == "cosine"
    assert config.min_learning_rate == pytest.approx(0.0001)
    assert config.communication_rounds == 50


def _client_config_from_experiment_config(config, client_id: str = "client_000") -> FederatedClientConfig:
    """Build a FederatedClientConfig the same way main() does, for direct field inspection.

    Mirrors src.experiments.run_federated_cifar100.main()'s
    FederatedClientConfig construction exactly for the fields relevant to
    round-level learning-rate scheduling, without touching the CIFAR-100
    dataset pipeline (device is forced to "cpu" here only because this
    helper never calls initialize_model()/trains anything -- it exists
    purely to inspect the constructed config's fields).
    """
    return FederatedClientConfig(
        client_id=client_id,
        local_epochs=config.local_epochs,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        scheduler=config.scheduler,
        mixed_precision=config.mixed_precision,
        device="cpu",
        seed=config.seed,
        log_interval=0,
        round_lr_schedule=config.round_lr_schedule,
        min_learning_rate=config.min_learning_rate,
        total_communication_rounds=config.communication_rounds,
        local_training_method=config.local_training_method,
        fedprox_mu=config.fedprox_mu,
    )


def test_total_communication_rounds_is_forwarded_from_communication_rounds() -> None:
    """total_communication_rounds always equals communication_rounds -- no separate YAML key exists."""
    config = load_federated_experiment_config(EXP_005_COSINE_CONFIG_PATH)
    client_config = _client_config_from_experiment_config(config)
    assert client_config.total_communication_rounds == config.communication_rounds == 50


def test_client_construction_receives_expected_round_lr_fields_for_cosine_config() -> None:
    """A FederatedClientConfig built the way main() builds it carries EXP_005's cosine round-LR fields."""
    config = load_federated_experiment_config(EXP_005_COSINE_CONFIG_PATH)
    client_config = _client_config_from_experiment_config(config)

    assert client_config.round_lr_schedule == "cosine"
    assert client_config.min_learning_rate == pytest.approx(0.0001)
    assert client_config.total_communication_rounds == 50


def test_client_construction_receives_constant_round_lr_fields_for_exp004_config() -> None:
    """A FederatedClientConfig built from EXP_004's config keeps the backward-compatible default."""
    config = load_federated_experiment_config(EXP_004_CONFIG_PATH)
    client_config = _client_config_from_experiment_config(config)

    assert client_config.round_lr_schedule == "constant"
    assert client_config.min_learning_rate == pytest.approx(0.0)
    assert client_config.total_communication_rounds == 50  # EXP_004 is also a 50-round config


def test_experiment_config_rejects_invalid_round_lr_schedule(tmp_path: Path) -> None:
    """An unrecognized round_lr_schedule value raises ConfigError before training starts."""
    text = EXP_004_CONFIG_PATH.read_text(encoding="utf-8") + "\nround_lr_schedule: not_a_schedule\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_negative_min_learning_rate(tmp_path: Path) -> None:
    """A negative min_learning_rate raises ConfigError before training starts."""
    text = EXP_004_CONFIG_PATH.read_text(encoding="utf-8") + "\nmin_learning_rate: -0.001\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_min_learning_rate_above_base_learning_rate(tmp_path: Path) -> None:
    """min_learning_rate greater than learning_rate raises ConfigError before training starts."""
    text = EXP_004_CONFIG_PATH.read_text(encoding="utf-8") + "\nmin_learning_rate: 1.0\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


# ----------------------------------------------------------------------
# FedProx local training method plumbing (Sprint 7.1)
# ----------------------------------------------------------------------


def test_old_configs_load_with_backward_compatible_fedprox_defaults() -> None:
    """EXP_004's config predates local_training_method/fedprox_mu and still loads cleanly.

    Both fields must fall back to their documented backward-compatible
    defaults ("fedavg" / 0.0) rather than raising a missing-field error.
    """
    config = load_federated_experiment_config(EXP_004_CONFIG_PATH)
    assert config.local_training_method == "fedavg"
    assert config.fedprox_mu == pytest.approx(0.0)


def test_experiment_config_rejects_invalid_local_training_method(tmp_path: Path) -> None:
    """An unrecognized local_training_method value raises ConfigError before training starts."""
    text = EXP_004_CONFIG_PATH.read_text(encoding="utf-8") + "\nlocal_training_method: not_a_method\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_negative_fedprox_mu(tmp_path: Path) -> None:
    """A negative fedprox_mu raises ConfigError before training starts."""
    text = EXP_004_CONFIG_PATH.read_text(encoding="utf-8") + "\nfedprox_mu: -0.01\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_fedprox_without_positive_mu(tmp_path: Path) -> None:
    """local_training_method='fedprox' with the default fedprox_mu=0.0 raises ConfigError."""
    text = EXP_004_CONFIG_PATH.read_text(encoding="utf-8") + "\nlocal_training_method: fedprox\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_client_construction_receives_fedavg_defaults_for_old_config() -> None:
    """A FederatedClientConfig built from EXP_004's config keeps the backward-compatible FedProx defaults."""
    config = load_federated_experiment_config(EXP_004_CONFIG_PATH)
    client_config = _client_config_from_experiment_config(config)

    assert client_config.local_training_method == "fedavg"
    assert client_config.fedprox_mu == pytest.approx(0.0)


def test_client_construction_receives_fedprox_fields() -> None:
    """FederatedClientConfig construction carries local_training_method/fedprox_mu through correctly."""
    base_config = load_federated_experiment_config(EXP_004_CONFIG_PATH)
    config = replace(base_config, local_training_method="fedprox", fedprox_mu=0.01)
    client_config = _client_config_from_experiment_config(config)

    assert client_config.local_training_method == "fedprox"
    assert client_config.fedprox_mu == pytest.approx(0.01)


# ----------------------------------------------------------------------
# Malicious-client attack pilot config plumbing (Phase 5)
# ----------------------------------------------------------------------


def test_old_configs_load_with_backward_compatible_attack_defaults() -> None:
    """EXP_004's config predates the attack fields and still loads cleanly.

    All four fields must fall back to their documented backward-compatible
    defaults ("none"/None/None/0.0) rather than raising a missing-field
    error.
    """
    config = load_federated_experiment_config(EXP_004_CONFIG_PATH)
    assert config.attack_type == "none"
    assert config.malicious_client_id is None
    assert config.attack_start_round is None
    assert config.attack_strength_rho == pytest.approx(0.0)


def test_experiment_config_rejects_invalid_attack_type(tmp_path: Path) -> None:
    """An unrecognized attack_type value raises ConfigError before training starts."""
    text = EXP_004_CONFIG_PATH.read_text(encoding="utf-8") + "\nattack_type: not_a_type\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_negative_attack_strength_rho(tmp_path: Path) -> None:
    """A negative attack_strength_rho raises ConfigError before training starts."""
    text = EXP_004_CONFIG_PATH.read_text(encoding="utf-8") + "\nattack_strength_rho: -1.0\n"
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_attack_without_malicious_client_id(tmp_path: Path) -> None:
    """attack_type != 'none' without malicious_client_id raises ConfigError."""
    text = (
        EXP_004_CONFIG_PATH.read_text(encoding="utf-8")
        + "\nattack_type: gaussian_update_noise\nattack_start_round: 20\nattack_strength_rho: 1.0\n"
    )
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_attack_without_attack_start_round(tmp_path: Path) -> None:
    """attack_type != 'none' without attack_start_round raises ConfigError."""
    text = (
        EXP_004_CONFIG_PATH.read_text(encoding="utf-8")
        + "\nattack_type: gaussian_update_noise\nmalicious_client_id: client_002\n"
        "attack_strength_rho: 1.0\n"
    )
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_attack_start_round_out_of_range(tmp_path: Path) -> None:
    """attack_start_round outside [0, communication_rounds) raises ConfigError."""
    text = (
        EXP_004_CONFIG_PATH.read_text(encoding="utf-8")
        + "\nattack_type: gaussian_update_noise\nmalicious_client_id: client_002\n"
        "attack_start_round: 999\nattack_strength_rho: 1.0\n"
    )
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_rejects_gaussian_attack_without_positive_rho(tmp_path: Path) -> None:
    """attack_type='gaussian_update_noise' with the default attack_strength_rho=0.0 raises ConfigError."""
    text = (
        EXP_004_CONFIG_PATH.read_text(encoding="utf-8")
        + "\nattack_type: gaussian_update_noise\nmalicious_client_id: client_002\n"
        "attack_start_round: 20\n"
    )
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(text, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_federated_experiment_config(bad_path)


def test_experiment_config_accepts_well_formed_gaussian_attack_config(tmp_path: Path) -> None:
    """A well-formed gaussian_update_noise attack configuration loads and validates cleanly."""
    text = (
        EXP_004_CONFIG_PATH.read_text(encoding="utf-8")
        + "\nattack_type: gaussian_update_noise\nmalicious_client_id: client_002\n"
        "attack_start_round: 20\nattack_strength_rho: 1.0\n"
    )
    good_path = tmp_path / "good.yaml"
    good_path.write_text(text, encoding="utf-8")
    config = load_federated_experiment_config(good_path)
    assert config.attack_type == "gaussian_update_noise"
    assert config.malicious_client_id == "client_002"
    assert config.attack_start_round == 20
    assert config.attack_strength_rho == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Client construction from partitions + registration
# ----------------------------------------------------------------------


def test_client_construction_from_partitions() -> None:
    """Each partition produces a FederatedClient whose dataset matches that partition exactly."""
    dataset = _make_dataset(40)
    partitions = partition_dataset_iid(dataset, num_clients=4, seed=0)

    clients = {cid: _make_client(cid, subset) for cid, subset in partitions.items()}

    for client_id, subset in partitions.items():
        assert len(clients[client_id].train_dataset) == len(subset)


def test_correct_client_sample_registration() -> None:
    """Registering each client with its real local sample count matches the partition sizes."""
    dataset = _make_dataset(40)
    partitions = partition_dataset_iid(dataset, num_clients=4, seed=0)

    server_config = FederatedServerConfig(
        server_id="exp_test", seed=0, clients_per_round=4, minimum_updates=4, checkpoint_dir=Path("unused")
    )
    server = FederatedServer(server_config, model_fn=_TinyModel)
    for client_id, subset in partitions.items():
        server.register_client(client_id, num_samples=len(subset))

    assert set(server.list_registered_clients()) == set(partitions.keys())


# ----------------------------------------------------------------------
# Optimizer-reset policy between rounds
# ----------------------------------------------------------------------


def test_optimizer_reset_policy_between_rounds() -> None:
    """Each round's load_global_weights() rebuilds a fresh optimizer (no momentum carried over)."""
    dataset = _make_dataset(40)
    partitions = partition_dataset_iid(dataset, num_clients=2, seed=0)
    clients = {cid: _make_client(cid, subset) for cid, subset in partitions.items()}

    server_config = FederatedServerConfig(
        server_id="exp_test", seed=0, clients_per_round=2, minimum_updates=2, checkpoint_dir=Path("unused")
    )
    server = FederatedServer(server_config, model_fn=_TinyModel)
    for client_id, subset in partitions.items():
        server.register_client(client_id, num_samples=len(subset))

    runner_config = FederatedRunnerConfig(total_rounds=2)
    runner = FederatedRunner(server=server, clients=clients, aggregation_strategy=FedAvg(), config=runner_config)

    any_client = next(iter(clients.values()))
    runner.run_round(0)
    optimizer_after_round_0 = any_client.optimizer
    runner.run_round(1)
    optimizer_after_round_1 = any_client.optimizer

    # A brand-new optimizer object each round proves state was fully reset,
    # not merely mutated in place.
    assert optimizer_after_round_0 is not optimizer_after_round_1
    momentum_buffer_present = any(
        any_client.optimizer.state.get(param, {}).get("momentum_buffer") is not None
        for group in any_client.optimizer.param_groups
        for param in group["params"]
    )
    # Only this round's single train_one_round() call could have populated
    # momentum, so a present buffer here reflects round 1 only.
    assert momentum_buffer_present


# ----------------------------------------------------------------------
# One tiny end-to-end IID federated experiment
# ----------------------------------------------------------------------


def test_one_tiny_end_to_end_iid_federated_experiment() -> None:
    """A full tiny federated run: partition, clients, server, FedAvg, runner, evaluation, final test."""
    dataset = _make_dataset(40)
    partitions = partition_dataset_iid(dataset, num_clients=4, seed=1)
    clients = {cid: _make_client(cid, subset) for cid, subset in partitions.items()}

    server_config = FederatedServerConfig(
        server_id="exp_e2e", seed=1, clients_per_round=4, minimum_updates=4, checkpoint_dir=Path("unused")
    )
    server = FederatedServer(server_config, model_fn=_TinyModel)
    for client_id, subset in partitions.items():
        server.register_client(client_id, num_samples=len(subset))

    initial_weights = {key: value.clone() for key, value in server.get_global_weights().items()}

    val_dataset = _make_dataset(12)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def evaluation_callback(model: nn.Module, round_id: int) -> dict[str, float]:
        metrics, _ = evaluate(model, val_loader, criterion, torch.device("cpu"))
        return {"loss": metrics.loss, "top1": metrics.top1, "top5": metrics.top5}

    runner_config = FederatedRunnerConfig(total_rounds=2, evaluate_every=1)
    runner = FederatedRunner(
        server=server, clients=clients, aggregation_strategy=FedAvg(), config=runner_config,
        evaluation_callback=evaluation_callback,
    )

    results = runner.run()

    assert len(results) == 2
    assert len(runner.get_history()) == 2
    assert all(result.round_status == "completed" for result in results)
    assert all(result.evaluation_metrics.get("top1") is not None for result in results)

    final_weights = server.get_global_weights()
    assert not torch.equal(initial_weights["linear.weight"], final_weights["linear.weight"])

    # Final "test" evaluation executes exactly once, on a separate held-out set.
    test_dataset = _make_dataset(16)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    final_model = _TinyModel()
    final_model.load_state_dict(final_weights)
    final_metrics, predictions = evaluate(
        final_model, test_loader, criterion, torch.device("cpu"), return_predictions=True
    )
    assert predictions is not None
    assert final_metrics.loss >= 0.0


def test_one_tiny_end_to_end_federated_round_with_fedprox() -> None:
    """A full tiny federated round with FedProx-configured clients completes and aggregates normally.

    Confirms FedAvg aggregation requires no special-casing for FedProx
    clients: from the runner/aggregation strategy's perspective, a
    FedProx client's submitted update is just a state dict like any
    other -- the round completes, the global model changes, and
    aggregation behaves exactly as the plain-FedAvg end-to-end test above.
    """
    dataset = _make_dataset(40)
    partitions = partition_dataset_iid(dataset, num_clients=4, seed=1)
    clients = {
        cid: _make_client(cid, subset, local_training_method="fedprox", fedprox_mu=0.01)
        for cid, subset in partitions.items()
    }

    server_config = FederatedServerConfig(
        server_id="exp_fedprox_e2e", seed=1, clients_per_round=4, minimum_updates=4,
        checkpoint_dir=Path("unused"),
    )
    server = FederatedServer(server_config, model_fn=_TinyModel)
    for client_id, subset in partitions.items():
        server.register_client(client_id, num_samples=len(subset))

    initial_weights = {key: value.clone() for key, value in server.get_global_weights().items()}

    runner_config = FederatedRunnerConfig(total_rounds=2, evaluate_every=1)
    runner = FederatedRunner(
        server=server, clients=clients, aggregation_strategy=FedAvg(), config=runner_config,
    )

    results = runner.run()

    assert len(results) == 2
    assert all(result.round_status == "completed" for result in results)
    assert all(result.aggregation_strategy == "fedavg" for result in results)

    final_weights = server.get_global_weights()
    assert not torch.equal(initial_weights["linear.weight"], final_weights["linear.weight"])


def test_one_tiny_end_to_end_federated_round_with_gaussian_attack() -> None:
    """A full tiny federated run with the gaussian_update_noise attack active completes normally.

    Wires the real, production src.attacks.traceability.AttackConfig /
    apply_gaussian_update_noise into a FederatedRunner's
    update_transform_callback exactly the way
    src.experiments.run_federated_cifar100.main() does (Phase 5), confirming
    the production attack path -- not just its own isolated unit tests --
    works end to end through the full client/server/FedAvg/runner stack.
    """
    from src.attacks.traceability import AttackConfig, apply_gaussian_update_noise

    dataset = _make_dataset(40)
    partitions = partition_dataset_iid(dataset, num_clients=4, seed=1)
    clients = {cid: _make_client(cid, subset) for cid, subset in partitions.items()}
    trainable_keys = frozenset(name for name, _ in _TinyModel().named_parameters())

    server_config = FederatedServerConfig(
        server_id="exp_attack_e2e", seed=1, clients_per_round=4, minimum_updates=4,
        checkpoint_dir=Path("unused"),
    )
    server = FederatedServer(server_config, model_fn=_TinyModel)
    for client_id, subset in partitions.items():
        server.register_client(client_id, num_samples=len(subset))

    malicious_client_id = sorted(partitions.keys())[0]
    attack_config = AttackConfig(
        attack_type="gaussian_update_noise",
        malicious_client_id=malicious_client_id,
        attack_start_round=0,
        attack_strength_rho=1.0,
    )

    def update_transform_callback(client_id, round_id, model_state, global_state):  # noqa: ANN001
        if not attack_config.is_active(round_id, client_id):
            return model_state
        return apply_gaussian_update_noise(
            model_state, global_state, trainable_keys,
            rho=attack_config.attack_strength_rho, seed=1, round_id=round_id, client_id=client_id,
        )

    runner_config = FederatedRunnerConfig(total_rounds=2, evaluate_every=1)
    runner = FederatedRunner(
        server=server, clients=clients, aggregation_strategy=FedAvg(), config=runner_config,
        update_transform_callback=update_transform_callback,
    )

    results = runner.run()

    assert len(results) == 2
    assert all(result.round_status == "completed" for result in results)
    assert all(result.aggregation_strategy == "fedavg" for result in results)


# ----------------------------------------------------------------------
# Output-writing helpers
# ----------------------------------------------------------------------


def test_output_files_written_to_temp_directory(tmp_path: Path) -> None:
    """The experiment's output-writing helpers create the expected files."""
    rows = [
        {
            "round_id": 0, "selected_client_ids": "client_000;client_001",
            "successful_client_ids": "client_000;client_001", "failed_client_ids": "",
            "num_successful_clients": 2, "total_samples": 20,
            "train_loss": 1.5, "train_top1": 0.3, "train_top5": 0.6,
            "validation_loss": 1.4, "validation_top1": 0.32, "validation_top5": 0.62,
            "round_seconds": 1.23, "cumulative_seconds": 1.23,
            "aggregation_strategy": "fedavg", "round_status": "completed",
        },
        {
            "round_id": 1, "selected_client_ids": "client_000;client_001",
            "successful_client_ids": "client_000;client_001", "failed_client_ids": "",
            "num_successful_clients": 2, "total_samples": 20,
            "train_loss": 1.1, "train_top1": 0.4, "train_top5": 0.7,
            "validation_loss": 1.0, "validation_top1": 0.42, "validation_top5": 0.72,
            "round_seconds": 1.05, "cumulative_seconds": 2.28,
            "aggregation_strategy": "fedavg", "round_status": "completed",
        },
    ]
    save_round_history_csv(rows, tmp_path / "round_history.csv")
    _line_plot(rows, "validation_loss", "Validation Loss", "Validation Loss", tmp_path / "validation_loss.png")
    save_round_duration_plot(rows, tmp_path / "communication_round_duration.png")

    confusion = torch.eye(NUM_CLASSES).numpy()
    save_confusion_matrix_plot(confusion, tmp_path / "confusion_matrix.png")
    save_per_class_accuracy_csv(["a", "b", "c", "d"], [1.0, 0.5, 0.0, 0.75], tmp_path / "per_class_accuracy.csv")

    dataset = _make_dataset(12)
    partitions = partition_dataset_iid(dataset, num_clients=3, seed=0)
    save_client_partition_outputs(
        partitions, tmp_path / "client_partition_summary.csv", tmp_path / "client_partition_indices.json"
    )

    _save_json({"ok": True}, tmp_path / "final_test_metrics.json")

    for name in (
        "round_history.csv", "validation_loss.png", "communication_round_duration.png",
        "confusion_matrix.png", "per_class_accuracy.csv", "client_partition_summary.csv",
        "client_partition_indices.json", "final_test_metrics.json",
    ):
        assert (tmp_path / name).exists()


def test_round_result_row_flattens_expected_fields() -> None:
    """_round_result_row extracts the fields required by the round-history spec."""
    result = FederatedRoundResult(
        round_id=0, selected_client_ids=("client_000", "client_001"),
        successful_client_ids=("client_000", "client_001"), failed_client_ids=(),
        aggregation_strategy="fedavg", total_samples=20,
        train_metrics={"loss": 1.5, "top1": 0.3, "top5": 0.6},
        evaluation_metrics={"loss": 1.4, "top1": 0.32, "top5": 0.62},
        round_status="completed", started_at_utc="t0", completed_at_utc="t1",
        metadata={"elapsed_seconds": 1.23},
    )
    row = _round_result_row(result, cumulative_seconds=1.23)

    assert row["round_id"] == 0
    assert row["num_successful_clients"] == 2
    assert row["train_loss"] == 1.5
    assert row["validation_top1"] == 0.32
    assert row["cumulative_seconds"] == 1.23
    assert row["round_status"] == "completed"


# ----------------------------------------------------------------------
# Checkpoint round-trip (tiny setup)
# ----------------------------------------------------------------------


def test_resume_checkpoint_path_works_in_tiny_setup(tmp_path: Path) -> None:
    """The runner's checkpoint save/load composes correctly with partition-derived clients."""
    dataset = _make_dataset(40)
    partitions = partition_dataset_iid(dataset, num_clients=2, seed=0)
    clients = {cid: _make_client(cid, subset) for cid, subset in partitions.items()}

    server_config = FederatedServerConfig(
        server_id="exp_ckpt", seed=0, clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path
    )
    server = FederatedServer(server_config, model_fn=_TinyModel)
    for client_id, subset in partitions.items():
        server.register_client(client_id, num_samples=len(subset))

    runner_config = FederatedRunnerConfig(total_rounds=2)
    runner = FederatedRunner(server=server, clients=clients, aggregation_strategy=FedAvg(), config=runner_config)
    runner.run_round(0)

    checkpoint_path = tmp_path / "runner_latest.pt"
    runner.save_runner_checkpoint(checkpoint_path)

    restored_clients = {cid: _make_client(cid, subset) for cid, subset in partitions.items()}
    restored_server = FederatedServer(
        FederatedServerConfig(
            server_id="exp_ckpt", seed=0, clients_per_round=2, minimum_updates=2, checkpoint_dir=tmp_path
        ),
        model_fn=_TinyModel,
    )
    for client_id, subset in partitions.items():
        restored_server.register_client(client_id, num_samples=len(subset))
    restored_runner = FederatedRunner(
        server=restored_server, clients=restored_clients, aggregation_strategy=FedAvg(), config=runner_config
    )
    restored_runner.load_runner_checkpoint(checkpoint_path)

    assert len(restored_runner.get_history()) == 1
    results = restored_runner.run()
    assert [r.round_id for r in results] == [1]
