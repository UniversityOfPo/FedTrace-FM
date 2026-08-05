"""First real IID federated CIFAR-100 experiment (Sprint 4.1).

Run from the project root with:
    python -m src.experiments.run_federated_cifar100 --config <config-path>

Wires together only already-existing, reusable components: the CIFAR-100
data pipeline (``src.data.cifar100``), the generic IID partitioner
(``src.data.federated_partition``), the CIFAR-compatible ResNet-18
(``src.models.resnet18_cifar``), the reusable training/evaluation engine
(``src.training.engine``, via ``FederatedClient``), and the federated
client/server/FedAvg/runner stack (``src.federated``). No new federated
abstraction is introduced here; no Non-IID partitioning, attacks,
attribution, or foundation-model logic exists in this script.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from src.data.cifar100 import build_datasets, get_class_names, split_train_validation  # noqa: E402
from src.data.federated_partition import partition_dataset_iid  # noqa: E402
from src.evaluation.metrics import (  # noqa: E402
    compute_balanced_accuracy,
    compute_confusion_matrix,
    macro_precision_recall_f1,
    per_class_accuracy,
)
from src.federated.aggregation.fedavg import FedAvg  # noqa: E402
from src.federated.client import FederatedClient, FederatedClientConfig  # noqa: E402
from src.federated.runner import FederatedRoundResult, FederatedRunner, FederatedRunnerConfig  # noqa: E402
from src.federated.server import FederatedServer, FederatedServerConfig  # noqa: E402
from src.models.resnet18_cifar import count_parameters, resnet18_cifar  # noqa: E402
from src.training.engine import evaluate, resolve_device  # noqa: E402
from src.utils import checkpoint as checkpoint_utils  # noqa: E402
from src.utils.config import ConfigError, load_data_config, load_yaml  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/cifar100_fedavg_iid.yaml")

_VALID_OPTIMIZERS: tuple[str, ...] = ("sgd", "adamw")
_VALID_SCHEDULERS: tuple[str, ...] = ("cosine", "none")
_VALID_ROUND_LR_SCHEDULES: tuple[str, ...] = ("constant", "cosine")
_VALID_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")
_SUPPORTED_MODELS: tuple[str, ...] = ("resnet18_cifar",)
_SUPPORTED_AGGREGATIONS: tuple[str, ...] = ("fedavg",)
_SUPPORTED_PARTITIONS: tuple[str, ...] = ("iid",)

_REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_id", "experiment_name", "dataset_config", "output_dir", "checkpoint_dir",
    "num_clients", "clients_per_round", "minimum_updates", "communication_rounds",
    "local_epochs", "batch_size", "num_workers", "model", "num_classes",
    "optimizer", "learning_rate", "momentum", "weight_decay", "scheduler", "mixed_precision",
    "aggregation", "partition", "validation_fraction", "evaluate_every", "checkpoint_every",
    "seed", "device", "fail_round_on_client_error", "continue_after_failed_round",
)


@dataclass(frozen=True)
class FederatedExperimentConfig:
    """Typed, validated configuration for the IID FedAvg CIFAR-100 experiment.

    ``round_lr_schedule``/``min_learning_rate`` (Sprint 5.2) are optional
    YAML fields, defaulting to ``"constant"``/``0.0`` when absent, so
    existing configs (e.g. `EXP_002`-`EXP_004`) that predate this plumbing
    remain valid and behaviorally unchanged. ``total_communication_rounds``
    is deliberately not a separate field here -- ``main()`` forwards this
    config's own ``communication_rounds`` into every
    ``FederatedClientConfig`` instead, so the two never risk disagreeing.
    """

    experiment_id: str
    experiment_name: str
    dataset_config: Path
    output_dir: Path
    checkpoint_dir: Path
    num_clients: int
    clients_per_round: int
    minimum_updates: int
    communication_rounds: int
    local_epochs: int
    batch_size: int
    num_workers: int
    model: str
    num_classes: int
    optimizer: str
    learning_rate: float
    momentum: float
    weight_decay: float
    scheduler: str
    mixed_precision: bool
    aggregation: str
    partition: str
    validation_fraction: float
    evaluate_every: int
    checkpoint_every: int
    seed: int
    device: str
    fail_round_on_client_error: bool
    continue_after_failed_round: bool
    round_lr_schedule: str = "constant"
    min_learning_rate: float = 0.0


def load_federated_experiment_config(path: Path | str) -> FederatedExperimentConfig:
    """Load and validate a federated CIFAR-100 experiment configuration.

    Args:
        path: Path to a YAML experiment configuration file.

    Returns:
        A validated, typed ``FederatedExperimentConfig``.

    Raises:
        ConfigError: If the file is missing, malformed, missing required
            fields, or contains invalid/out-of-range/unsupported values.
    """
    path = Path(path)
    raw = load_yaml(path)

    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ConfigError(
            f"Configuration file {path} is missing required field(s): {', '.join(missing)}"
        )

    try:
        config = FederatedExperimentConfig(
            experiment_id=str(raw["experiment_id"]),
            experiment_name=str(raw["experiment_name"]),
            dataset_config=Path(raw["dataset_config"]),
            output_dir=Path(raw["output_dir"]),
            checkpoint_dir=Path(raw["checkpoint_dir"]),
            num_clients=int(raw["num_clients"]),
            clients_per_round=int(raw["clients_per_round"]),
            minimum_updates=int(raw["minimum_updates"]),
            communication_rounds=int(raw["communication_rounds"]),
            local_epochs=int(raw["local_epochs"]),
            batch_size=int(raw["batch_size"]),
            num_workers=int(raw["num_workers"]),
            model=str(raw["model"]).lower(),
            num_classes=int(raw["num_classes"]),
            optimizer=str(raw["optimizer"]).lower(),
            learning_rate=float(raw["learning_rate"]),
            momentum=float(raw["momentum"]),
            weight_decay=float(raw["weight_decay"]),
            scheduler=str(raw["scheduler"]).lower(),
            mixed_precision=bool(raw["mixed_precision"]),
            aggregation=str(raw["aggregation"]).lower(),
            partition=str(raw["partition"]).lower(),
            validation_fraction=float(raw["validation_fraction"]),
            evaluate_every=int(raw["evaluate_every"]),
            checkpoint_every=int(raw["checkpoint_every"]),
            seed=int(raw["seed"]),
            device=str(raw["device"]).lower(),
            fail_round_on_client_error=bool(raw["fail_round_on_client_error"]),
            continue_after_failed_round=bool(raw["continue_after_failed_round"]),
            round_lr_schedule=str(raw.get("round_lr_schedule", "constant")).lower(),
            min_learning_rate=float(raw.get("min_learning_rate", 0.0)),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid field value in configuration file {path}: {exc}") from exc

    if not config.dataset_config.exists():
        raise ConfigError(f"dataset_config file not found: {config.dataset_config}")
    if config.num_clients <= 0:
        raise ConfigError(f"num_clients must be > 0, got {config.num_clients}.")
    if config.clients_per_round <= 0 or config.clients_per_round > config.num_clients:
        raise ConfigError(
            f"clients_per_round must be in [1, num_clients={config.num_clients}], "
            f"got {config.clients_per_round}."
        )
    if config.minimum_updates <= 0 or config.minimum_updates > config.clients_per_round:
        raise ConfigError(
            f"minimum_updates must be in [1, clients_per_round={config.clients_per_round}], "
            f"got {config.minimum_updates}."
        )
    if config.communication_rounds <= 0:
        raise ConfigError(f"communication_rounds must be > 0, got {config.communication_rounds}.")
    if config.local_epochs <= 0:
        raise ConfigError(f"local_epochs must be > 0, got {config.local_epochs}.")
    if config.batch_size <= 0:
        raise ConfigError(f"batch_size must be > 0, got {config.batch_size}.")
    if config.num_workers < 0:
        raise ConfigError(f"num_workers must be >= 0, got {config.num_workers}.")
    if config.model not in _SUPPORTED_MODELS:
        raise ConfigError(f"model must be one of {_SUPPORTED_MODELS}, got '{config.model}'.")
    if config.num_classes <= 0:
        raise ConfigError(f"num_classes must be > 0, got {config.num_classes}.")
    if config.optimizer not in _VALID_OPTIMIZERS:
        raise ConfigError(f"optimizer must be one of {_VALID_OPTIMIZERS}, got '{config.optimizer}'.")
    if config.learning_rate <= 0:
        raise ConfigError(f"learning_rate must be positive, got {config.learning_rate}.")
    if not (0.0 <= config.momentum < 1.0):
        raise ConfigError(f"momentum must be in [0, 1), got {config.momentum}.")
    if config.weight_decay < 0:
        raise ConfigError(f"weight_decay must be zero or positive, got {config.weight_decay}.")
    if config.scheduler not in _VALID_SCHEDULERS:
        raise ConfigError(f"scheduler must be one of {_VALID_SCHEDULERS}, got '{config.scheduler}'.")
    if config.aggregation not in _SUPPORTED_AGGREGATIONS:
        raise ConfigError(
            f"aggregation must be one of {_SUPPORTED_AGGREGATIONS} in this sprint, "
            f"got '{config.aggregation}'."
        )
    if config.partition not in _SUPPORTED_PARTITIONS:
        raise ConfigError(
            f"partition must be one of {_SUPPORTED_PARTITIONS} in this sprint "
            f"(non-IID partitioning is not implemented yet), got '{config.partition}'."
        )
    if not (0.0 < config.validation_fraction < 0.5):
        raise ConfigError(
            f"validation_fraction must be > 0 and < 0.5, got {config.validation_fraction}."
        )
    if config.evaluate_every <= 0:
        raise ConfigError(f"evaluate_every must be > 0, got {config.evaluate_every}.")
    if config.checkpoint_every <= 0:
        raise ConfigError(f"checkpoint_every must be > 0, got {config.checkpoint_every}.")
    if config.seed < 0:
        raise ConfigError(f"seed must be >= 0, got {config.seed}.")
    if config.device not in _VALID_DEVICES:
        raise ConfigError(f"device must be one of {_VALID_DEVICES}, got '{config.device}'.")
    if config.round_lr_schedule not in _VALID_ROUND_LR_SCHEDULES:
        raise ConfigError(
            f"round_lr_schedule must be one of {_VALID_ROUND_LR_SCHEDULES}, "
            f"got '{config.round_lr_schedule}'."
        )
    if config.min_learning_rate < 0:
        raise ConfigError(
            f"min_learning_rate must be zero or positive, got {config.min_learning_rate}."
        )
    if config.min_learning_rate > config.learning_rate:
        raise ConfigError(
            f"min_learning_rate ({config.min_learning_rate}) must be <= learning_rate "
            f"({config.learning_rate})."
        )

    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the experiment entry point."""
    parser = argparse.ArgumentParser(description="Run the IID FedAvg CIFAR-100 experiment.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to the experiment YAML.")
    return parser.parse_args()


def _configure_logging(log_path: Path) -> None:
    """Configure root logging to write to both the console and ``log_path``."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode="w", encoding="utf-8")],
        force=True,
    )


def _git_commit_hash(repo_root: Path) -> str | None:
    """Best-effort current git commit hash, read directly from .git/ files (no git command run).

    Returns None (rather than raising) if the repository metadata cannot
    be read for any reason -- this is provenance information, not a
    required input to the experiment.
    """
    try:
        head_path = repo_root / ".git" / "HEAD"
        head_content = head_path.read_text(encoding="utf-8").strip()
        if head_content.startswith("ref:"):
            ref_relative = head_content.split(" ", 1)[1].strip()
            ref_path = repo_root / ".git" / ref_relative
            return ref_path.read_text(encoding="utf-8").strip()
        return head_content  # detached HEAD: HEAD file already holds the hash
    except OSError:
        return None


def _save_json(payload: dict[str, Any], path: Path) -> None:
    """Save a plain dictionary as an indented JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    logger.info("Saved %s", path)


def _round_result_row(result: FederatedRoundResult, cumulative_seconds: float) -> dict[str, Any]:
    """Flatten one FederatedRoundResult into a CSV-friendly row."""
    return {
        "round_id": result.round_id,
        "selected_client_ids": ";".join(result.selected_client_ids),
        "successful_client_ids": ";".join(result.successful_client_ids),
        "failed_client_ids": ";".join(result.failed_client_ids),
        "num_successful_clients": len(result.successful_client_ids),
        "total_samples": result.total_samples,
        "train_loss": result.train_metrics.get("loss"),
        "train_top1": result.train_metrics.get("top1"),
        "train_top5": result.train_metrics.get("top5"),
        "validation_loss": result.evaluation_metrics.get("loss"),
        "validation_top1": result.evaluation_metrics.get("top1"),
        "validation_top5": result.evaluation_metrics.get("top5"),
        "round_seconds": result.metadata.get("elapsed_seconds"),
        "cumulative_seconds": cumulative_seconds,
        "aggregation_strategy": result.aggregation_strategy,
        "round_status": result.round_status,
    }


def save_round_history_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Save per-round history as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved round history to %s", path)


def _line_plot(
    rows: list[dict[str, Any]], y_key: str, ylabel: str, title: str, path: Path, scale: float = 1.0
) -> None:
    """Save a simple communication-round-on-x-axis line plot for rows where ``y_key`` is present."""
    plotted = [(row["round_id"], row[y_key]) for row in rows if row.get(y_key) is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    if plotted:
        xs, ys = zip(*plotted)
        ax.plot(xs, [y * scale for y in ys], marker="o")
    ax.set_xlabel("Communication Round")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def save_round_duration_plot(rows: list[dict[str, Any]], path: Path) -> None:
    """Save a per-round wall-clock duration bar chart (communication round on x-axis)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    rounds = [row["round_id"] for row in rows]
    durations = [row["round_seconds"] or 0.0 for row in rows]
    ax.bar(rounds, durations)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Round Duration (s)")
    ax.set_title("Communication Round Duration")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def save_confusion_matrix_plot(confusion, path: Path) -> None:
    """Save a confusion matrix heatmap (matplotlib only, default colormap)."""
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(confusion, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix (Final Global Model, Test Set)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", path)


def save_per_class_accuracy_csv(class_names: list[str], class_acc, path: Path) -> None:
    """Save per-class accuracy as a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "class_name", "accuracy"])
        for idx, name in enumerate(class_names):
            writer.writerow([idx, name, float(class_acc[idx])])
    logger.info("Saved %s", path)


def save_client_partition_outputs(
    partitions: dict[str, Any], summary_path: Path, indices_path: Path
) -> None:
    """Save per-client sample counts (CSV) and index lists (JSON).

    Indices are relative to the 45,000-image training subset (i.e., after
    the train/validation split), not the original 50,000-image CIFAR-100
    training set.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["client_id", "num_samples"])
        for client_id in sorted(partitions):
            writer.writerow([client_id, len(partitions[client_id])])
    logger.info("Saved %s", summary_path)

    indices_payload = {
        client_id: list(int(i) for i in partitions[client_id].indices) for client_id in sorted(partitions)
    }
    _save_json(indices_payload, indices_path)


def main() -> None:
    """Run the IID FedAvg CIFAR-100 experiment end to end."""
    args = parse_args()

    try:
        config = load_federated_experiment_config(args.config)
    except ConfigError as exc:
        logging.basicConfig(level=logging.INFO)
        logging.getLogger(__name__).error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    output_dir = config.output_dir
    checkpoint_dir = config.checkpoint_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    _configure_logging(output_dir / "run.log")
    logger.info("Starting federated experiment '%s' (%s)", config.experiment_name, config.experiment_id)

    set_seed(config.seed)

    device = resolve_device(config.device)
    gpu_name: str | None = None
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Using GPU: %s (%.1f GB VRAM)", gpu_name, vram_gb)
    else:
        logger.info("CUDA not available; using CPU.")

    # --- Data: reuse the existing pipeline and the existing, reproducible
    # 45,000/5,000 stratified train/validation split. Only the 45,000-image
    # training subset is partitioned across clients; the validation subset
    # stays centralized (used by the server-side evaluation callback), and
    # the official test set is loaded separately and touched exactly once,
    # at the very end.
    data_config = load_data_config(config.dataset_config)
    _, test_dataset = build_datasets(data_config.data_root, download=data_config.download)
    train_subset, val_subset = split_train_validation(
        data_config.data_root, download=data_config.download,
        validation_fraction=config.validation_fraction, seed=config.seed,
    )
    class_names = get_class_names(test_dataset)
    if len(class_names) != config.num_classes:
        raise ValueError(
            f"Configured num_classes ({config.num_classes}) does not match the dataset's "
            f"actual class count ({len(class_names)})."
        )

    partitions = partition_dataset_iid(train_subset, num_clients=config.num_clients, seed=config.seed)
    logger.info(
        "Partitioned %d training samples across %d IID clients.", len(train_subset), config.num_clients
    )

    def model_fn() -> nn.Module:
        """Fresh, uninitialized CIFAR-compatible ResNet-18 -- used for every client and the server."""
        return resnet18_cifar(num_classes=config.num_classes)

    # --- Clients: one per partition, each with its own optimizer/scheduler.
    clients: dict[str, FederatedClient] = {}
    for client_id, subset in partitions.items():
        client_config = FederatedClientConfig(
            client_id=client_id,
            local_epochs=config.local_epochs,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            pin_memory=data_config.pin_memory,
            optimizer=config.optimizer,
            learning_rate=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            scheduler=config.scheduler,
            mixed_precision=config.mixed_precision,
            device=config.device,
            seed=config.seed,
            log_interval=0,
            round_lr_schedule=config.round_lr_schedule,
            min_learning_rate=config.min_learning_rate,
            total_communication_rounds=config.communication_rounds,
        )
        client = FederatedClient(client_id=client_id, model_fn=model_fn, train_dataset=subset, config=client_config)
        client.initialize_model()
        clients[client_id] = client

    # --- Server + registration.
    server_config = FederatedServerConfig(
        server_id=config.experiment_id, seed=config.seed, clients_per_round=config.clients_per_round,
        minimum_updates=config.minimum_updates, checkpoint_dir=checkpoint_dir, strict_round_matching=True,
    )
    server = FederatedServer(server_config, model_fn=model_fn)
    for client_id, subset in partitions.items():
        server.register_client(client_id, num_samples=len(subset))

    total_params, trainable_params = count_parameters(model_fn())
    logger.info("Model parameters: total=%d trainable=%d", total_params, trainable_params)

    # --- Aggregation strategy (only FedAvg is supported/validated this sprint).
    strategy = FedAvg()

    # --- Centralized validation loader + evaluation callback.
    val_loader = DataLoader(
        val_subset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=data_config.pin_memory,
    )
    criterion = nn.CrossEntropyLoss()

    def evaluation_callback(model: nn.Module, round_id: int) -> dict[str, float]:
        """Evaluate the current global model on the shared, centralized validation set."""
        model.to(device)
        metrics, _ = evaluate(model, val_loader, criterion, device, use_amp=config.mixed_precision)
        return {"loss": metrics.loss, "top1": metrics.top1, "top5": metrics.top5}

    runner_checkpoint_path = checkpoint_dir / "runner_latest.pt"

    def round_callback(result: FederatedRoundResult) -> None:
        """Log progress and, on the configured cadence, checkpoint the runner's latest state."""
        logger.info(
            "Round %d: status=%s successful=%d/%d val_top1=%s",
            result.round_id, result.round_status, len(result.successful_client_ids),
            len(result.selected_client_ids), result.evaluation_metrics.get("top1"),
        )
        if result.round_id % config.checkpoint_every == 0:
            runner.save_runner_checkpoint(runner_checkpoint_path)

    runner_config = FederatedRunnerConfig(
        total_rounds=config.communication_rounds, evaluate_every=config.evaluate_every,
        checkpoint_every=config.checkpoint_every, fail_round_on_client_error=config.fail_round_on_client_error,
        continue_after_failed_round=config.continue_after_failed_round,
    )
    runner = FederatedRunner(
        server=server, clients=clients, aggregation_strategy=strategy, config=runner_config,
        evaluation_callback=evaluation_callback, round_callback=round_callback,
    )

    experiment_start = time.perf_counter()
    runner.run()
    total_experiment_seconds = time.perf_counter() - experiment_start

    # Guarantee a final, up-to-date "latest" checkpoint regardless of cadence.
    runner.save_runner_checkpoint(runner_checkpoint_path)

    history = runner.get_history()

    # --- Final global model: the server's current global weights after all
    # rounds (each successful round already explicitly applied its
    # aggregation result; a failed round never touched it).
    final_model = model_fn().to(device)
    final_model.load_state_dict(server.get_global_weights())

    final_checkpoint_path = checkpoint_dir / "final_global_model.pt"
    checkpoint_utils.save_checkpoint(
        {
            "model_state_dict": server.get_global_weights(),
            "experiment_id": config.experiment_id,
            "seed": config.seed,
            "model_name": config.model,
            "num_classes": config.num_classes,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        },
        final_checkpoint_path,
    )

    # --- Final test evaluation: exactly once, on the untouched test set.
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.num_workers, pin_memory=data_config.pin_memory,
    )
    final_test_metrics, predictions = evaluate(
        final_model, test_loader, criterion, device, use_amp=config.mixed_precision, return_predictions=True
    )
    targets_arr, preds_arr = predictions
    confusion = compute_confusion_matrix(targets_arr, preds_arr, num_classes=config.num_classes)
    class_acc = per_class_accuracy(confusion)
    macro_precision, macro_recall, macro_f1 = macro_precision_recall_f1(targets_arr, preds_arr, config.num_classes)
    balanced_acc = compute_balanced_accuracy(targets_arr, preds_arr)

    print(f"Final test loss: {final_test_metrics.loss:.4f}")
    print(f"Final test top-1: {final_test_metrics.top1:.4f}")
    print(f"Final test top-5: {final_test_metrics.top5:.4f}")

    # --- Outputs.
    rows: list[dict[str, Any]] = []
    cumulative_seconds = 0.0
    for result in history:
        cumulative_seconds += result.metadata.get("elapsed_seconds") or 0.0
        rows.append(_round_result_row(result, cumulative_seconds))

    if rows:
        save_round_history_csv(rows, output_dir / "round_history.csv")
        _line_plot(rows, "validation_loss", "Validation Loss", "Validation Loss", output_dir / "validation_loss.png")
        _line_plot(
            rows, "validation_top1", "Validation Top-1 (%)", "Validation Top-1 Accuracy",
            output_dir / "validation_top1.png", scale=100.0,
        )
        _line_plot(
            rows, "validation_top5", "Validation Top-5 (%)", "Validation Top-5 Accuracy",
            output_dir / "validation_top5.png", scale=100.0,
        )
        _line_plot(
            rows, "train_loss", "Weighted Local Training Loss", "Local Training Loss",
            output_dir / "local_training_loss.png",
        )
        save_round_duration_plot(rows, output_dir / "communication_round_duration.png")

    save_confusion_matrix_plot(confusion, output_dir / "confusion_matrix.png")
    save_per_class_accuracy_csv(class_names, class_acc, output_dir / "per_class_accuracy.csv")
    save_client_partition_outputs(
        partitions, output_dir / "client_partition_summary.csv", output_dir / "client_partition_indices.json"
    )

    final_test_payload = {
        "loss": final_test_metrics.loss,
        "top1": final_test_metrics.top1,
        "top5": final_test_metrics.top5,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "balanced_accuracy": balanced_acc,
    }
    _save_json(final_test_payload, output_dir / "final_test_metrics.json")

    completed_rows = [row for row in rows if row["round_status"] == "completed"]
    validation_rows = [row for row in rows if row["validation_top1"] is not None]
    best_row = max(validation_rows, key=lambda row: row["validation_top1"], default=None)
    final_validation_row = validation_rows[-1] if validation_rows else None

    client_checkpoint_names = [
        f"{runner_checkpoint_path.stem}_client_{client_id}.pt" for client_id in sorted(clients)
    ]

    summary = {
        "experiment_id": config.experiment_id,
        "experiment_name": config.experiment_name,
        "dataset": "CIFAR-100",
        "partition": config.partition,
        "model": config.model,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "num_clients": config.num_clients,
        "clients_per_round": config.clients_per_round,
        "minimum_updates": config.minimum_updates,
        "communication_rounds_configured": config.communication_rounds,
        "communication_rounds_completed": len(completed_rows),
        "local_epochs": config.local_epochs,
        "aggregation_strategy": strategy.name,
        "optimizer": config.optimizer,
        "learning_rate": config.learning_rate,
        "batch_size": config.batch_size,
        "seed": config.seed,
        "device": str(device),
        "gpu_name": gpu_name,
        "client_sample_counts": {cid: len(partitions[cid]) for cid in sorted(partitions)},
        "best_validation": (
            {"round_id": best_row["round_id"], "top1": best_row["validation_top1"]}
            if best_row is not None else None
        ),
        "final_validation": (
            {
                "loss": final_validation_row["validation_loss"],
                "top1": final_validation_row["validation_top1"],
                "top5": final_validation_row["validation_top5"],
            }
            if final_validation_row is not None else None
        ),
        "final_test": {
            "loss": final_test_metrics.loss,
            "top1": final_test_metrics.top1,
            "top5": final_test_metrics.top5,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_acc,
        },
        "total_experiment_seconds": total_experiment_seconds,
        "checkpoint_paths": {
            "runner_latest": str(runner_checkpoint_path),
            "runner_latest_server": str(runner_checkpoint_path.with_name(f"{runner_checkpoint_path.stem}_server.pt")),
            "runner_latest_clients": [str(runner_checkpoint_path.with_name(name)) for name in client_checkpoint_names],
            "final_global_model": str(final_checkpoint_path),
        },
        "git_commit": _git_commit_hash(Path(__file__).resolve().parents[2]),
    }
    _save_json(summary, output_dir / "summary.json")

    logger.info("Federated experiment '%s' complete.", config.experiment_name)


if __name__ == "__main__":
    main()
