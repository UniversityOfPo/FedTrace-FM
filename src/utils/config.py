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
