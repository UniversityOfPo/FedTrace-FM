"""Generic dataset partitioning for federated learning (Sprints 4.1, 6.1).

Provides ``partition_dataset_iid`` (a seeded-shuffle IID split) and
``partition_dataset_dirichlet`` (a seeded, label-skew Non-IID split), both
of which split any map-style ``torch.utils.data.Dataset`` (one with a known
``len()``) into disjoint ``Subset``s. Neither contains any CIFAR-specific
logic -- callers supply ``labels`` explicitly to
``partition_dataset_dirichlet`` rather than this module inspecting the
dataset's internals. Shard-based Non-IID partitioning is not implemented
yet and is not in scope here.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset, Subset

logger = logging.getLogger(__name__)

_MIN_CLIENT_ID_DIGITS = 3


def partition_dataset_iid(
    dataset: Dataset, num_clients: int, seed: int
) -> dict[str, Subset]:
    """Partition ``dataset`` into ``num_clients`` disjoint, near-equal IID subsets.

    Every source sample is assigned to exactly one client (a seeded
    permutation of all indices is split into ``num_clients`` contiguous
    chunks), so partition sizes differ by at most one sample and the
    union of all partitions' indices exactly reconstructs
    ``range(len(dataset))``. ``dataset`` itself is never modified — each
    returned ``Subset`` only stores a reference to it plus its own index
    list, and ``Subset.indices``/``len(subset)`` already provide the
    per-client index list and sample count, so no separate metadata
    structure is needed to record them.

    Args:
        dataset: A map-style dataset with a well-defined ``len()``.
        num_clients: Number of client partitions to create. Must be
            positive and must not exceed ``len(dataset)`` (each client
            must receive at least one sample).
        seed: Seed controlling the shuffle. The same seed always produces
            the same partition for a given ``(dataset, num_clients)``; a
            different seed produces a different shuffle (and therefore,
            almost always, a different partition).

    Returns:
        A mapping of stable, human-readable client id (``"client_000"``,
        ``"client_001"``, ...) to that client's ``Subset``.

    Raises:
        ValueError: If ``num_clients`` is not positive, or exceeds
            ``len(dataset)``.
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}.")

    dataset_size = len(dataset)  # type: ignore[arg-type]
    if num_clients > dataset_size:
        raise ValueError(
            f"num_clients ({num_clients}) must not exceed dataset size ({dataset_size})."
        )

    rng = np.random.default_rng(seed)
    shuffled_indices = rng.permutation(dataset_size)
    chunks = np.array_split(shuffled_indices, num_clients)

    id_width = max(_MIN_CLIENT_ID_DIGITS, len(str(num_clients - 1)))
    partitions: dict[str, Subset] = {}
    for client_index, indices in enumerate(chunks):
        client_id = f"client_{client_index:0{id_width}d}"
        partitions[client_id] = Subset(dataset, indices.tolist())

    logger.info(
        "Partitioned %d samples into %d IID client(s) (seed=%d, sizes %d-%d).",
        dataset_size, num_clients, seed,
        min(len(p) for p in partitions.values()),
        max(len(p) for p in partitions.values()),
    )
    return partitions


def partition_dataset_dirichlet(
    dataset: Dataset,
    labels: Sequence[int] | np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int,
    min_samples_per_client: int,
    max_attempts: int = 100,
) -> dict[str, Subset]:
    """Partition ``dataset`` into ``num_clients`` Non-IID subsets via a Dirichlet label split.

    For each class, a per-client proportion vector is drawn from
    ``Dirichlet(alpha, ..., alpha)`` (one concentration parameter per
    client) and that class's samples are assigned to clients according to
    those proportions. Smaller ``alpha`` produces more skewed (more
    Non-IID) per-client class distributions; large ``alpha`` approaches the
    IID case. This function holds no CIFAR-specific logic -- ``labels``
    must be supplied by the caller, aligned index-for-index with
    ``dataset`` (i.e. ``labels[i]`` is the label of ``dataset[i]``).

    Because a single Dirichlet draw can (especially at low ``alpha``)
    starve one or more clients below ``min_samples_per_client``, the full
    partition is retried -- continuing to draw from the same seeded
    generator, so the overall process remains deterministic for a given
    ``seed`` -- until every client meets the minimum, or ``max_attempts``
    is exhausted (in which case a clear ``RuntimeError`` is raised rather
    than silently returning a degenerate partition).

    Args:
        dataset: A map-style dataset with a well-defined ``len()``.
        labels: The integer class label for every sample in ``dataset``,
            in the same order/indexing as ``dataset`` itself.
            ``len(labels)`` must equal ``len(dataset)``.
        num_clients: Number of client partitions to create. Must be
            positive and must not exceed ``len(dataset)``.
        alpha: Dirichlet concentration parameter. Must be positive.
        seed: Seed controlling both the per-class index shuffle and the
            Dirichlet draws. The same seed always produces the same
            partition for a given ``(dataset, labels, num_clients, alpha,
            min_samples_per_client)``; a different seed produces a
            different draw sequence (and therefore, almost always, a
            different partition).
        min_samples_per_client: Minimum number of samples every client
            must receive, across all classes combined. Must be positive,
            and ``min_samples_per_client * num_clients`` must not exceed
            ``len(dataset)`` (otherwise no split, however balanced, could
            possibly satisfy it).
        max_attempts: Maximum number of full re-partition attempts before
            giving up. Each attempt redraws every class's Dirichlet
            proportions.

    Returns:
        A mapping of stable, human-readable client id (``"client_000"``,
        ``"client_001"``, ...) to that client's ``Subset``. Every source
        sample is assigned to exactly one client.

    Raises:
        ValueError: If ``num_clients``, ``alpha``, or
            ``min_samples_per_client`` is not positive; if ``len(labels)``
            does not equal ``len(dataset)``; if ``num_clients`` exceeds
            ``len(dataset)``; or if ``min_samples_per_client * num_clients``
            exceeds ``len(dataset)``.
        RuntimeError: If no partition satisfying ``min_samples_per_client``
            for every client was found within ``max_attempts`` attempts.
    """
    if num_clients <= 0:
        raise ValueError(f"num_clients must be > 0, got {num_clients}.")
    if alpha <= 0:
        raise ValueError(f"alpha must be > 0, got {alpha}.")
    if min_samples_per_client <= 0:
        raise ValueError(f"min_samples_per_client must be > 0, got {min_samples_per_client}.")

    dataset_size = len(dataset)  # type: ignore[arg-type]
    labels_array = np.asarray(labels)
    if len(labels_array) != dataset_size:
        raise ValueError(
            f"labels length ({len(labels_array)}) must equal dataset length ({dataset_size})."
        )
    if num_clients > dataset_size:
        raise ValueError(
            f"num_clients ({num_clients}) must not exceed dataset size ({dataset_size})."
        )
    if min_samples_per_client * num_clients > dataset_size:
        raise ValueError(
            f"min_samples_per_client * num_clients ({min_samples_per_client} * {num_clients} = "
            f"{min_samples_per_client * num_clients}) must not exceed dataset size ({dataset_size})."
        )

    rng = np.random.default_rng(seed)
    class_labels = np.unique(labels_array)

    attempt = 0
    client_indices: list[list[int]] = [[] for _ in range(num_clients)]
    for attempt in range(1, max_attempts + 1):
        client_indices = [[] for _ in range(num_clients)]
        for class_label in class_labels:
            class_indices = np.where(labels_array == class_label)[0]
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            split_points = (np.cumsum(proportions)[:-1] * len(class_indices)).astype(int)
            for client_index, split in enumerate(np.split(class_indices, split_points)):
                client_indices[client_index].extend(int(i) for i in split)

        sizes = [len(indices) for indices in client_indices]
        if min(sizes) >= min_samples_per_client:
            break
    else:
        raise RuntimeError(
            f"Could not find a Dirichlet(alpha={alpha}) partition into {num_clients} clients "
            f"satisfying min_samples_per_client={min_samples_per_client} after {max_attempts} "
            "attempt(s). Consider lowering min_samples_per_client or increasing alpha."
        )

    id_width = max(_MIN_CLIENT_ID_DIGITS, len(str(num_clients - 1)))
    partitions: dict[str, Subset] = {}
    for client_index, indices in enumerate(client_indices):
        client_id = f"client_{client_index:0{id_width}d}"
        partitions[client_id] = Subset(dataset, indices)

    logger.info(
        "Partitioned %d samples into %d Dirichlet(alpha=%.3g) client(s) "
        "(seed=%d, attempt=%d/%d, sizes %d-%d).",
        dataset_size, num_clients, alpha, seed, attempt, max_attempts,
        min(len(p) for p in partitions.values()),
        max(len(p) for p in partitions.values()),
    )
    return partitions
