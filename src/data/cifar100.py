"""CIFAR-100 dataset loading utilities.

Provides training/evaluation transforms, dataset construction (with
automatic download), DataLoader construction, and summary statistics for
the CIFAR-100 dataset. No model training or federated logic lives here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
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


def build_dataloaders(
    train_dataset: datasets.CIFAR100,
    test_dataset: datasets.CIFAR100,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test DataLoaders around the given datasets.

    Args:
        train_dataset: The CIFAR-100 training dataset.
        test_dataset: The CIFAR-100 test dataset.
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
