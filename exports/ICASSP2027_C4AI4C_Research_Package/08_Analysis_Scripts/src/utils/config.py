"""Typed YAML configuration loading utilities.

This module separates generic YAML loading (``load_yaml``) from the
schema-specific validation used by the CIFAR-100 data pipeline
(``load_data_config`` / ``DataConfig``), so the loader can be reused for
future configuration schemas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or invalid."""


@dataclass(frozen=True)
class DataConfig:
    """Typed configuration for the CIFAR-100 data pipeline."""

    dataset_name: str
    data_root: Path
    download: bool
    batch_size: int
    num_workers: int
    pin_memory: bool
    seed: int
    output_dir: Path
    num_preview_images: int


_REQUIRED_FIELDS: tuple[str, ...] = (
    "dataset_name",
    "data_root",
    "download",
    "batch_size",
    "num_workers",
    "pin_memory",
    "seed",
    "output_dir",
    "num_preview_images",
)


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file into a plain dictionary.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed top-level mapping.

    Raises:
        ConfigError: If the file does not exist, is not a file, cannot be
            parsed, or does not contain a top-level mapping.
    """
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ConfigError(f"Configuration path is not a file: {path}")

    try:
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Failed to parse YAML file {path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(
            f"Configuration file {path} must contain a top-level mapping, got {type(raw).__name__}."
        )

    return raw


def load_data_config(path: Path | str) -> DataConfig:
    """Load and validate the CIFAR-100 data pipeline configuration.

    Args:
        path: Path to a YAML configuration file.

    Returns:
        A validated, typed ``DataConfig`` instance.

    Raises:
        ConfigError: If the file is missing, malformed, missing required
            fields, or contains invalid field values.
    """
    path = Path(path)
    raw = load_yaml(path)

    missing = [field for field in _REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ConfigError(
            f"Configuration file {path} is missing required field(s): {', '.join(missing)}"
        )

    try:
        config = DataConfig(
            dataset_name=str(raw["dataset_name"]),
            data_root=Path(raw["data_root"]),
            download=bool(raw["download"]),
            batch_size=int(raw["batch_size"]),
            num_workers=int(raw["num_workers"]),
            pin_memory=bool(raw["pin_memory"]),
            seed=int(raw["seed"]),
            output_dir=Path(raw["output_dir"]),
            num_preview_images=int(raw["num_preview_images"]),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid field value in configuration file {path}: {exc}") from exc

    if config.batch_size <= 0:
        raise ConfigError(f"batch_size must be a positive integer, got {config.batch_size}.")
    if config.num_workers < 0:
        raise ConfigError(f"num_workers must be zero or positive, got {config.num_workers}.")
    if config.num_preview_images <= 0:
        raise ConfigError(
            f"num_preview_images must be a positive integer, got {config.num_preview_images}."
        )

    logger.debug("Loaded configuration from %s: %s", path, config)
    return config


@dataclass(frozen=True)
class TrainingConfig:
    """Typed configuration for the centralized CIFAR-100 ResNet-18 baseline."""

    experiment_name: str
    dataset_config: Path
    output_dir: Path
    checkpoint_dir: Path
    num_classes: int
    epochs: int
    batch_size: int
    num_workers: int
    validation_fraction: float
    learning_rate: float
    momentum: float
    weight_decay: float
    optimizer: str
    scheduler: str
    warmup_epochs: int
    label_smoothing: float
    mixed_precision: bool
    gradient_clip_norm: float | None
    seed: int
    device: str
    save_last_checkpoint: bool
    save_best_checkpoint: bool
    resume_from: Path | None
    log_interval: int


_TRAINING_REQUIRED_FIELDS: tuple[str, ...] = (
    "experiment_name",
    "dataset_config",
    "output_dir",
    "checkpoint_dir",
    "num_classes",
    "epochs",
    "batch_size",
    "num_workers",
    "validation_fraction",
    "learning_rate",
    "momentum",
    "weight_decay",
    "optimizer",
    "scheduler",
    "warmup_epochs",
    "label_smoothing",
    "mixed_precision",
    "gradient_clip_norm",
    "seed",
    "device",
    "save_last_checkpoint",
    "save_best_checkpoint",
    "resume_from",
    "log_interval",
)

_VALID_OPTIMIZERS: tuple[str, ...] = ("sgd", "adamw")
_VALID_SCHEDULERS: tuple[str, ...] = ("cosine", "none")
_VALID_DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")


def load_training_config(path: Path | str) -> TrainingConfig:
    """Load and validate the centralized training configuration.

    Args:
        path: Path to a YAML training configuration file.

    Returns:
        A validated, typed ``TrainingConfig`` instance.

    Raises:
        ConfigError: If the file is missing, malformed, missing required
            fields, or contains invalid/out-of-range field values.
    """
    path = Path(path)
    raw = load_yaml(path)

    missing = [field for field in _TRAINING_REQUIRED_FIELDS if field not in raw]
    if missing:
        raise ConfigError(
            f"Configuration file {path} is missing required field(s): {', '.join(missing)}"
        )

    gradient_clip_norm = raw["gradient_clip_norm"]
    resume_from = raw["resume_from"]

    try:
        config = TrainingConfig(
            experiment_name=str(raw["experiment_name"]),
            dataset_config=Path(raw["dataset_config"]),
            output_dir=Path(raw["output_dir"]),
            checkpoint_dir=Path(raw["checkpoint_dir"]),
            num_classes=int(raw["num_classes"]),
            epochs=int(raw["epochs"]),
            batch_size=int(raw["batch_size"]),
            num_workers=int(raw["num_workers"]),
            validation_fraction=float(raw["validation_fraction"]),
            learning_rate=float(raw["learning_rate"]),
            momentum=float(raw["momentum"]),
            weight_decay=float(raw["weight_decay"]),
            optimizer=str(raw["optimizer"]).lower(),
            scheduler=str(raw["scheduler"]).lower(),
            warmup_epochs=int(raw["warmup_epochs"]),
            label_smoothing=float(raw["label_smoothing"]),
            mixed_precision=bool(raw["mixed_precision"]),
            gradient_clip_norm=float(gradient_clip_norm) if gradient_clip_norm is not None else None,
            seed=int(raw["seed"]),
            device=str(raw["device"]).lower(),
            save_last_checkpoint=bool(raw["save_last_checkpoint"]),
            save_best_checkpoint=bool(raw["save_best_checkpoint"]),
            resume_from=Path(resume_from) if resume_from is not None else None,
            log_interval=int(raw["log_interval"]),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid field value in configuration file {path}: {exc}") from exc

    if not config.dataset_config.exists():
        raise ConfigError(f"dataset_config file not found: {config.dataset_config}")
    if config.num_classes <= 0:
        raise ConfigError(f"num_classes must be a positive integer, got {config.num_classes}.")
    if config.epochs <= 0:
        raise ConfigError(f"epochs must be a positive integer, got {config.epochs}.")
    if config.batch_size <= 0:
        raise ConfigError(f"batch_size must be a positive integer, got {config.batch_size}.")
    if config.num_workers < 0:
        raise ConfigError(f"num_workers must be zero or positive, got {config.num_workers}.")
    if not (0.0 < config.validation_fraction < 0.5):
        raise ConfigError(
            f"validation_fraction must be > 0 and < 0.5, got {config.validation_fraction}."
        )
    if config.learning_rate <= 0:
        raise ConfigError(f"learning_rate must be positive, got {config.learning_rate}.")
    if not (0.0 <= config.momentum < 1.0):
        raise ConfigError(f"momentum must be in [0, 1), got {config.momentum}.")
    if config.weight_decay < 0:
        raise ConfigError(f"weight_decay must be zero or positive, got {config.weight_decay}.")
    if config.optimizer not in _VALID_OPTIMIZERS:
        raise ConfigError(f"optimizer must be one of {_VALID_OPTIMIZERS}, got '{config.optimizer}'.")
    if config.scheduler not in _VALID_SCHEDULERS:
        raise ConfigError(f"scheduler must be one of {_VALID_SCHEDULERS}, got '{config.scheduler}'.")
    if config.warmup_epochs < 0:
        raise ConfigError(f"warmup_epochs must be zero or positive, got {config.warmup_epochs}.")
    if config.warmup_epochs >= config.epochs:
        raise ConfigError(
            f"warmup_epochs ({config.warmup_epochs}) must be less than epochs ({config.epochs})."
        )
    if not (0.0 <= config.label_smoothing < 1.0):
        raise ConfigError(f"label_smoothing must be in [0, 1), got {config.label_smoothing}.")
    if config.gradient_clip_norm is not None and config.gradient_clip_norm <= 0:
        raise ConfigError(
            f"gradient_clip_norm must be positive when set, got {config.gradient_clip_norm}."
        )
    if config.device not in _VALID_DEVICES:
        raise ConfigError(f"device must be one of {_VALID_DEVICES}, got '{config.device}'.")
    if config.resume_from is not None and not config.resume_from.exists():
        raise ConfigError(f"resume_from checkpoint not found: {config.resume_from}")
    if config.log_interval <= 0:
        raise ConfigError(f"log_interval must be a positive integer, got {config.log_interval}.")

    logger.debug("Loaded training configuration from %s: %s", path, config)
    return config
