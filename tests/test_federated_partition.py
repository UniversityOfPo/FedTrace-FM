"""Lightweight tests for generic IID dataset partitioning (Sprint 4.1).

Uses a tiny synthetic dataset so the suite runs quickly on CPU. No
CIFAR-100 data is loaded or downloaded here.
"""

from __future__ import annotations

import torch
from torch.utils.data import TensorDataset

from src.data.federated_partition import partition_dataset_iid

import pytest


def _make_dataset(num_samples: int = 23) -> TensorDataset:
    images = torch.randn(num_samples, 4)
    labels = torch.randint(0, 10, (num_samples,))
    return TensorDataset(images, labels)


def test_num_clients_must_be_positive() -> None:
    """num_clients <= 0 is rejected with a clear error."""
    with pytest.raises(ValueError):
        partition_dataset_iid(_make_dataset(), num_clients=0, seed=0)


def test_num_clients_must_not_exceed_dataset_size() -> None:
    """num_clients larger than the dataset is rejected."""
    dataset = _make_dataset(num_samples=5)
    with pytest.raises(ValueError):
        partition_dataset_iid(dataset, num_clients=6, seed=0)


def test_deterministic_same_seed() -> None:
    """The same seed produces an identical partition."""
    dataset = _make_dataset()
    partition_a = partition_dataset_iid(dataset, num_clients=4, seed=42)
    partition_b = partition_dataset_iid(dataset, num_clients=4, seed=42)

    assert set(partition_a.keys()) == set(partition_b.keys())
    for client_id in partition_a:
        assert list(partition_a[client_id].indices) == list(partition_b[client_id].indices)


def test_different_seed_differs() -> None:
    """A different seed produces a different partition."""
    dataset = _make_dataset()
    partition_a = partition_dataset_iid(dataset, num_clients=4, seed=42)
    partition_b = partition_dataset_iid(dataset, num_clients=4, seed=1)

    all_indices_a = [tuple(partition_a[cid].indices) for cid in sorted(partition_a)]
    all_indices_b = [tuple(partition_b[cid].indices) for cid in sorted(partition_b)]
    assert all_indices_a != all_indices_b


def test_no_overlap_and_complete_coverage() -> None:
    """Every source index appears in exactly one client's partition."""
    dataset = _make_dataset(num_samples=23)
    partitions = partition_dataset_iid(dataset, num_clients=5, seed=7)

    all_indices: list[int] = []
    for subset in partitions.values():
        all_indices.extend(subset.indices)

    assert len(all_indices) == len(dataset)
    assert len(set(all_indices)) == len(all_indices)  # no duplicates -> no overlap
    assert set(all_indices) == set(range(len(dataset)))  # full coverage


def test_balanced_partition_sizes() -> None:
    """Partition sizes differ by at most one sample."""
    dataset = _make_dataset(num_samples=23)
    partitions = partition_dataset_iid(dataset, num_clients=5, seed=7)

    sizes = [len(subset) for subset in partitions.values()]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == len(dataset)


def test_stable_client_ids() -> None:
    """Client ids are stable, human-readable, and zero-padded."""
    dataset = _make_dataset(num_samples=23)
    partitions = partition_dataset_iid(dataset, num_clients=5, seed=7)

    assert set(partitions.keys()) == {
        "client_000", "client_001", "client_002", "client_003", "client_004"
    }


def test_original_dataset_not_mutated() -> None:
    """Partitioning does not modify the source dataset."""
    dataset = _make_dataset(num_samples=23)
    images_before = dataset.tensors[0].clone()
    labels_before = dataset.tensors[1].clone()

    partition_dataset_iid(dataset, num_clients=5, seed=7)

    assert torch.equal(dataset.tensors[0], images_before)
    assert torch.equal(dataset.tensors[1], labels_before)
