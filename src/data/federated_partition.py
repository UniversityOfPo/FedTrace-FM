"""Generic IID dataset partitioning for federated learning (Sprint 4.1).

Provides ``partition_dataset_iid``, which splits any map-style
``torch.utils.data.Dataset`` (one with a known ``len()``) into disjoint,
near-equal-size ``Subset``s via a seeded shuffle. Contains no CIFAR-specific
logic and no Dirichlet/shard (non-IID) partitioning -- those are deferred
to a later sprint.
"""

from __future__ import annotations

import logging

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
