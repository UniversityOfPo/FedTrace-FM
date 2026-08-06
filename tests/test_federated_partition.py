"""Lightweight tests for generic dataset partitioning (Sprints 4.1, 6.1).

Uses tiny synthetic datasets so the suite runs quickly on CPU. No
CIFAR-100 data is loaded or downloaded here.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import TensorDataset

import src.data.federated_partition as federated_partition
from src.data.federated_partition import partition_dataset_dirichlet, partition_dataset_iid

import pytest


def _make_dataset(num_samples: int = 23) -> TensorDataset:
    images = torch.randn(num_samples, 4)
    labels = torch.randint(0, 10, (num_samples,))
    return TensorDataset(images, labels)


def _make_balanced_dataset(
    num_classes: int, samples_per_class: int
) -> tuple[TensorDataset, torch.Tensor]:
    """A dataset with an exact, known, class-balanced label distribution.

    Balanced (rather than random) labels make Dirichlet-partition tests
    deterministic about how many samples of each class exist to allocate,
    independent of any label-generation randomness.
    """
    num_samples = num_classes * samples_per_class
    images = torch.randn(num_samples, 4)
    labels = torch.arange(num_classes).repeat_interleave(samples_per_class)
    return TensorDataset(images, labels), labels


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


# ----------------------------------------------------------------------
# Dirichlet Non-IID partitioning (Sprint 6.1)
# ----------------------------------------------------------------------


def test_dirichlet_num_clients_must_be_positive() -> None:
    """num_clients <= 0 is rejected with a clear error."""
    dataset, labels = _make_balanced_dataset(num_classes=5, samples_per_class=20)
    with pytest.raises(ValueError):
        partition_dataset_dirichlet(
            dataset, labels=labels, num_clients=0, alpha=0.5, seed=0, min_samples_per_client=1
        )


def test_dirichlet_alpha_must_be_positive() -> None:
    """alpha <= 0 is rejected with a clear error."""
    dataset, labels = _make_balanced_dataset(num_classes=5, samples_per_class=20)
    with pytest.raises(ValueError):
        partition_dataset_dirichlet(
            dataset, labels=labels, num_clients=4, alpha=0.0, seed=0, min_samples_per_client=1
        )


def test_dirichlet_min_samples_per_client_must_be_positive() -> None:
    """min_samples_per_client <= 0 is rejected with a clear error."""
    dataset, labels = _make_balanced_dataset(num_classes=5, samples_per_class=20)
    with pytest.raises(ValueError):
        partition_dataset_dirichlet(
            dataset, labels=labels, num_clients=4, alpha=0.5, seed=0, min_samples_per_client=0
        )


def test_dirichlet_labels_length_must_match_dataset() -> None:
    """labels shorter/longer than the dataset is rejected with a clear error."""
    dataset, labels = _make_balanced_dataset(num_classes=5, samples_per_class=20)
    with pytest.raises(ValueError):
        partition_dataset_dirichlet(
            dataset, labels=labels[:-1], num_clients=4, alpha=0.5, seed=0, min_samples_per_client=1
        )


def test_dirichlet_num_clients_must_not_exceed_dataset_size() -> None:
    """num_clients larger than the dataset is rejected."""
    dataset, labels = _make_balanced_dataset(num_classes=2, samples_per_class=2)  # 4 samples
    with pytest.raises(ValueError):
        partition_dataset_dirichlet(
            dataset, labels=labels, num_clients=5, alpha=0.5, seed=0, min_samples_per_client=1
        )


def test_dirichlet_min_samples_times_num_clients_must_not_exceed_dataset_size() -> None:
    """A min_samples_per_client that is unsatisfiable by construction is rejected up front."""
    dataset, labels = _make_balanced_dataset(num_classes=5, samples_per_class=20)  # 100 samples
    with pytest.raises(ValueError):
        # 30 * 4 = 120 > 100: impossible regardless of how balanced the split is.
        partition_dataset_dirichlet(
            dataset, labels=labels, num_clients=4, alpha=0.5, seed=0, min_samples_per_client=30
        )


def test_dirichlet_deterministic_same_seed() -> None:
    """The same seed produces an identical Dirichlet partition."""
    dataset, labels = _make_balanced_dataset(num_classes=10, samples_per_class=50)
    partition_a = partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.5, seed=42, min_samples_per_client=10
    )
    partition_b = partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.5, seed=42, min_samples_per_client=10
    )

    assert set(partition_a.keys()) == set(partition_b.keys())
    for client_id in partition_a:
        assert list(partition_a[client_id].indices) == list(partition_b[client_id].indices)


def test_dirichlet_different_seed_differs() -> None:
    """A different seed produces a different Dirichlet partition."""
    dataset, labels = _make_balanced_dataset(num_classes=10, samples_per_class=50)
    partition_a = partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.5, seed=42, min_samples_per_client=10
    )
    partition_b = partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.5, seed=1, min_samples_per_client=10
    )

    all_indices_a = [tuple(partition_a[cid].indices) for cid in sorted(partition_a)]
    all_indices_b = [tuple(partition_b[cid].indices) for cid in sorted(partition_b)]
    assert all_indices_a != all_indices_b


def test_dirichlet_no_overlap_and_complete_coverage() -> None:
    """Every source index appears in exactly one client's Dirichlet partition."""
    dataset, labels = _make_balanced_dataset(num_classes=10, samples_per_class=50)
    partitions = partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.5, seed=7, min_samples_per_client=10
    )

    all_indices: list[int] = []
    for subset in partitions.values():
        all_indices.extend(subset.indices)

    assert len(all_indices) == len(dataset)
    assert len(set(all_indices)) == len(all_indices)  # no duplicates -> no overlap
    assert set(all_indices) == set(range(len(dataset)))  # full coverage


def test_dirichlet_respects_minimum_samples_per_client() -> None:
    """Every client's Dirichlet partition meets the configured minimum size."""
    dataset, labels = _make_balanced_dataset(num_classes=10, samples_per_class=50)
    min_samples = 25
    partitions = partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.3, seed=3, min_samples_per_client=min_samples
    )

    for subset in partitions.values():
        assert len(subset) >= min_samples


def test_dirichlet_stable_client_ids() -> None:
    """Client ids are stable, human-readable, and zero-padded, matching the IID convention."""
    dataset, labels = _make_balanced_dataset(num_classes=10, samples_per_class=50)
    partitions = partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.5, seed=7, min_samples_per_client=10
    )

    assert set(partitions.keys()) == {
        "client_000", "client_001", "client_002", "client_003", "client_004"
    }


def test_dirichlet_original_dataset_not_mutated() -> None:
    """Dirichlet partitioning does not modify the source dataset."""
    dataset, labels = _make_balanced_dataset(num_classes=10, samples_per_class=50)
    images_before = dataset.tensors[0].clone()
    labels_before = dataset.tensors[1].clone()

    partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.5, seed=7, min_samples_per_client=10
    )

    assert torch.equal(dataset.tensors[0], images_before)
    assert torch.equal(dataset.tensors[1], labels_before)


def test_dirichlet_class_distributions_are_meaningfully_heterogeneous() -> None:
    """A low-alpha Dirichlet split produces visibly uneven per-client class coverage.

    Under IID partitioning every client would have (near-)complete class
    coverage. A low-alpha (strongly skewed) Dirichlet split should leave
    at least one client missing at least one class entirely -- direct
    evidence the partition is genuinely Non-IID, not merely labeled as such.
    """
    num_classes = 10
    dataset, labels = _make_balanced_dataset(num_classes=num_classes, samples_per_class=100)
    partitions = partition_dataset_dirichlet(
        dataset, labels=labels, num_clients=5, alpha=0.1, seed=11, min_samples_per_client=20
    )

    labels_array = labels.numpy()
    missing_class_counts = [
        num_classes - len({int(labels_array[i]) for i in subset.indices})
        for subset in partitions.values()
    ]
    assert max(missing_class_counts) > 0


class _AlwaysFirstClientGenerator:
    """A minimal fake RNG that always allocates 100% of every class to client 0.

    Used to deterministically exercise the bounded-retry-exhaustion path
    without depending on the real algorithm's statistical behavior (which
    cannot be predicted in advance without actually running it).
    """

    def shuffle(self, array: np.ndarray) -> None:
        return None  # no-op: shuffling doesn't matter for this test

    def dirichlet(self, alpha_vector: np.ndarray) -> np.ndarray:
        proportions = np.zeros(len(alpha_vector))
        proportions[0] = 1.0
        return proportions


def test_dirichlet_bounded_retries_raise_clear_error_when_unsatisfiable(monkeypatch) -> None:
    """Exhausting max_attempts without satisfying min_samples_per_client raises a clear RuntimeError.

    A pathological RNG that starves every client but the first guarantees
    every attempt fails min_samples_per_client for clients 1-3, so the
    bounded-retry loop is guaranteed to exhaust max_attempts and raise --
    deterministically, regardless of alpha.
    """
    dataset, labels = _make_balanced_dataset(num_classes=4, samples_per_class=10)  # 40 samples
    monkeypatch.setattr(
        federated_partition.np.random, "default_rng", lambda seed: _AlwaysFirstClientGenerator()
    )

    with pytest.raises(RuntimeError):
        federated_partition.partition_dataset_dirichlet(
            dataset, labels=labels, num_clients=4, alpha=0.5, seed=0,
            min_samples_per_client=5, max_attempts=3,
        )
