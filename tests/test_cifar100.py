"""Lightweight tests for the CIFAR-100 data inspection pipeline.

These tests reuse the locally downloaded dataset (module-scoped fixtures)
rather than re-downloading it per test.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data.cifar100 import build_dataloaders, build_datasets, get_class_names
from src.utils.config import ConfigError, load_data_config

CONFIG_PATH = Path("configs/cifar100_data.yaml")

EXPECTED_NUM_CLASSES = 100
EXPECTED_TRAIN_SIZE = 50000
EXPECTED_TEST_SIZE = 10000


@pytest.fixture(scope="module")
def data_config():
    """Load the CIFAR-100 data configuration once for all tests."""
    return load_data_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def datasets_pair(data_config):
    """Build train/test datasets once for all tests in this module."""
    return build_datasets(data_config.data_root, download=data_config.download)


def test_load_data_config(data_config) -> None:
    """The configuration file loads with the expected required fields."""
    assert data_config.dataset_name == "CIFAR100"
    assert data_config.batch_size > 0
    assert data_config.seed == 42


def test_load_data_config_missing_file() -> None:
    """Loading a nonexistent configuration file raises a clear error."""
    with pytest.raises(ConfigError):
        load_data_config(Path("configs/does_not_exist.yaml"))


def test_class_count(datasets_pair) -> None:
    """CIFAR-100 exposes exactly 100 class names."""
    train_dataset, _ = datasets_pair
    class_names = get_class_names(train_dataset)
    assert len(class_names) == EXPECTED_NUM_CLASSES


def test_dataset_sizes(datasets_pair) -> None:
    """Train and test splits have the expected standard CIFAR-100 sizes."""
    train_dataset, test_dataset = datasets_pair
    assert len(train_dataset) == EXPECTED_TRAIN_SIZE
    assert len(test_dataset) == EXPECTED_TEST_SIZE


def test_label_range(datasets_pair) -> None:
    """All training labels fall within the valid [0, 100) range."""
    train_dataset, _ = datasets_pair
    labels = train_dataset.targets
    assert min(labels) == 0
    assert max(labels) == EXPECTED_NUM_CLASSES - 1


def test_dataloader_batch_shape(data_config, datasets_pair) -> None:
    """A single training batch has the expected image and label dimensions."""
    train_dataset, test_dataset = datasets_pair
    train_loader, _ = build_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=data_config.batch_size,
        num_workers=0,
        pin_memory=False,
    )
    images, labels = next(iter(train_loader))
    assert images.shape == (data_config.batch_size, 3, 32, 32)
    assert labels.shape == (data_config.batch_size,)
