"""Lightweight tests for the CIFAR-100 data inspection pipeline.

These tests reuse the locally downloaded dataset (module-scoped fixtures)
rather than re-downloading it per test.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from src.data.cifar100 import (
    build_dataloaders,
    build_datasets,
    get_class_names,
    split_train_validation,
)
from src.utils.config import ConfigError, load_data_config

CONFIG_PATH = Path("configs/cifar100_data.yaml")

EXPECTED_NUM_CLASSES = 100
EXPECTED_TRAIN_SIZE = 50000
EXPECTED_TEST_SIZE = 10000
SPLIT_VALIDATION_FRACTION = 0.1
SPLIT_SEED = 42


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


@pytest.fixture(scope="module")
def split_pair(data_config):
    """Build one reproducible stratified train/validation split for reuse across tests."""
    return split_train_validation(
        data_config.data_root,
        download=data_config.download,
        validation_fraction=SPLIT_VALIDATION_FRACTION,
        seed=SPLIT_SEED,
    )


def test_split_expected_sizes(split_pair) -> None:
    """A 0.1 validation_fraction on 50,000 training samples yields 45,000/5,000."""
    train_subset, val_subset = split_pair
    assert len(train_subset) == 45000
    assert len(val_subset) == 5000


def test_split_no_overlap(split_pair) -> None:
    """Train and validation subsets share no sample indices."""
    train_subset, val_subset = split_pair
    assert set(train_subset.indices).isdisjoint(set(val_subset.indices))


def test_split_reproducible_with_same_seed(data_config) -> None:
    """Calling the split twice with the same seed yields identical indices."""
    train_a, val_a = split_train_validation(
        data_config.data_root,
        download=data_config.download,
        validation_fraction=SPLIT_VALIDATION_FRACTION,
        seed=SPLIT_SEED,
    )
    train_b, val_b = split_train_validation(
        data_config.data_root,
        download=data_config.download,
        validation_fraction=SPLIT_VALIDATION_FRACTION,
        seed=SPLIT_SEED,
    )
    assert list(train_a.indices) == list(train_b.indices)
    assert list(val_a.indices) == list(val_b.indices)


def test_split_differs_with_different_seed(data_config) -> None:
    """Changing the seed changes which samples land in the validation subset."""
    _, val_a = split_train_validation(
        data_config.data_root,
        download=data_config.download,
        validation_fraction=SPLIT_VALIDATION_FRACTION,
        seed=SPLIT_SEED,
    )
    _, val_b = split_train_validation(
        data_config.data_root,
        download=data_config.download,
        validation_fraction=SPLIT_VALIDATION_FRACTION,
        seed=SPLIT_SEED + 1,
    )
    assert list(val_a.indices) != list(val_b.indices)


def test_split_preserves_class_representation(split_pair) -> None:
    """All 100 classes remain present, and roughly proportionally sized, in both subsets."""
    train_subset, val_subset = split_pair

    train_targets = [train_subset.dataset.targets[i] for i in train_subset.indices]
    val_targets = [val_subset.dataset.targets[i] for i in val_subset.indices]

    assert len(set(train_targets)) == EXPECTED_NUM_CLASSES
    assert len(set(val_targets)) == EXPECTED_NUM_CLASSES

    # CIFAR-100 has exactly 500 training samples per class; with
    # validation_fraction=0.1 each class should contribute ~50 samples to
    # validation. Allow slack around the ideal value for stratified-split
    # rounding rather than requiring an exact count.
    val_counts = Counter(val_targets)
    for class_index in range(EXPECTED_NUM_CLASSES):
        assert 40 <= val_counts[class_index] <= 60
