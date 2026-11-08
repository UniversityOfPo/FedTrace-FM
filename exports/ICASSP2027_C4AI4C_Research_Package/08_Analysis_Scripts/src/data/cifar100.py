"""CIFAR-100 dataset loading utilities.

Provides training/evaluation transforms, dataset construction (with
automatic download), DataLoader construction, and summary statistics for
the CIFAR-100 dataset. No model training or federated logic lives here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# Standard CIFAR-100 per-channel normalization statistics.
CIFAR100_MEAN: tuple[float, float, float] = (0.5071, 0.4867, 0.4408)
CIFAR100_STD: tuple[float, float, float] = (0.2675, 0.2565, 0.2761)


def build_train_transforms() -> transforms.Compose:
    """Return the training-time augmentation and normalization pipeline."""
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def build_eval_transforms() -> transforms.Compose:
    """Return the evaluation-time (no augmentation) normalization pipeline."""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )


def build_datasets(
    data_root: Path | str, download: bool = True
) -> tuple[datasets.CIFAR100, datasets.CIFAR100]:
    """Instantiate the CIFAR-100 train and test datasets.

    The dataset is downloaded automatically into ``data_root`` on first
    execution when ``download`` is True and the data is not already present.

    Args:
        data_root: Directory where CIFAR-100 is stored or should be
            downloaded to.
        download: Whether to download the dataset if it is not already
            present under ``data_root``.

    Returns:
        A tuple of ``(train_dataset, test_dataset)``.

    Raises:
        RuntimeError: If the dataset is absent under ``data_root`` and
            ``download`` is False.
    """
    data_root = Path(data_root)
    data_root.mkdir(parents=True, exist_ok=True)

    try:
        train_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=True,
            download=download,
            transform=build_train_transforms(),
        )
        test_dataset = datasets.CIFAR100(
            root=str(data_root),
            train=False,
            download=download,
            transform=build_eval_transforms(),
        )
    except RuntimeError as exc:
        raise RuntimeError(
            f"CIFAR-100 dataset not found under {data_root} and download=False. "
            "Set download=True in the configuration or place the dataset manually."
        ) from exc

    logger.info(
        "Loaded CIFAR-100 from %s: %d train samples, %d test samples",
        data_root,
        len(train_dataset),
        len(test_dataset),
    )
    return train_dataset, test_dataset


def split_train_validation(
    data_root: Path | str,
    download: bool,
    validation_fraction: float,
    seed: int,
) -> tuple[Subset, Subset]:
    """Split the CIFAR-100 training set into stratified train/validation subsets.

    Two independent ``CIFAR100`` instances are constructed over the same
    underlying training data on disk: one with training-time augmentations,
    one with evaluation-only transforms. The returned train subset indexes
    into the augmented instance and the validation subset indexes into the
    clean instance, so each subset gets the correct transform without ever
    mutating a shared dataset object's ``.transform`` in place. The official
    CIFAR-100 test set is never touched by this function.

    The split itself is computed once (via a stratified shuffle keyed on
    ``seed``) and applied identically to both instances, so corresponding
    indices in the two subsets refer to the same underlying images.

    Args:
        data_root: Directory where CIFAR-100 is stored or downloaded to.
        download: Whether to download the dataset if not already present.
        validation_fraction: Fraction of the 50,000 training images to
            reserve for validation (expected to already be validated as
            ``0 < validation_fraction < 0.5`` by the caller's configuration).
        seed: Random seed controlling the stratified split. The same seed
            always yields the same split; a different seed yields a
            different split.

    Returns:
        A tuple ``(train_subset, val_subset)`` of ``torch.utils.data.Subset``
        instances, stratified so every class is proportionally represented
        in both subsets.
    """
    data_root = Path(data_root)

    train_transform_dataset = datasets.CIFAR100(
        root=str(data_root), train=True, download=download, transform=build_train_transforms()
    )
    eval_transform_dataset = datasets.CIFAR100(
        root=str(data_root), train=True, download=download, transform=build_eval_transforms()
    )

    targets = train_transform_dataset.targets
    indices = list(range(len(targets)))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_fraction,
        random_state=seed,
        stratify=targets,
        shuffle=True,
    )

    train_subset = Subset(train_transform_dataset, train_indices)
    val_subset = Subset(eval_transform_dataset, val_indices)

    logger.info(
        "Split CIFAR-100 training set: %d train / %d validation "
        "(validation_fraction=%.3f, seed=%d)",
        len(train_subset),
        len(val_subset),
        validation_fraction,
        seed,
    )
    return train_subset, val_subset


def build_dataloaders(
    train_dataset: Dataset,
    test_dataset: Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """Build a shuffled and a non-shuffled DataLoader around the given datasets.

    Accepts full CIFAR-100 datasets or ``Subset`` instances (e.g. the
    train/validation subsets from ``split_train_validation``) interchangeably,
    since both satisfy the ``Dataset`` protocol.

    Args:
        train_dataset: Dataset used to build the shuffled loader.
        test_dataset: Dataset used to build the non-shuffled loader.
        batch_size: Number of samples per batch.
        num_workers: Number of worker subprocesses for data loading.
        pin_memory: Whether to use pinned (page-locked) host memory.

    Returns:
        A tuple of ``(train_loader, test_loader)``.
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, test_loader


def get_class_names(dataset: datasets.CIFAR100) -> list[str]:
    """Return the ordered list of CIFAR-100 class names for a dataset."""
    return list(dataset.classes)


@dataclass(frozen=True)
class DatasetSummary:
    """Summary statistics describing a loaded CIFAR-100 dataset split."""

    num_samples: int
    num_classes: int
    image_shape: tuple[int, ...]
    label_min: int
    label_max: int


def summarize_dataset(dataset: datasets.CIFAR100) -> DatasetSummary:
    """Compute basic summary statistics for a CIFAR-100 dataset split.

    Args:
        dataset: A CIFAR-100 dataset instance (train or test split).

    Returns:
        A ``DatasetSummary`` with sample/class counts, image shape, and
        label range.
    """
    image, _ = dataset[0]
    labels = dataset.targets
    return DatasetSummary(
        num_samples=len(dataset),
        num_classes=len(dataset.classes),
        image_shape=tuple(image.shape),
        label_min=int(min(labels)),
        label_max=int(max(labels)),
    )
