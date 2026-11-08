"""CLI script to inspect the CIFAR-100 dataset.

Loads the YAML configuration, downloads CIFAR-100 if absent, validates
class/label coverage, and writes a sample image grid, per-class count CSV,
and a JSON summary into the configured output directory.

Run from the project root with:
    python -m src.data.inspect_dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.data.cifar100 import (  # noqa: E402
    CIFAR100_MEAN,
    CIFAR100_STD,
    build_dataloaders,
    build_datasets,
    get_class_names,
    summarize_dataset,
)
from src.utils.config import ConfigError, load_data_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path("configs/cifar100_data.yaml")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the inspection script."""
    parser = argparse.ArgumentParser(description="Inspect the CIFAR-100 dataset.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the CIFAR-100 data YAML configuration.",
    )
    return parser.parse_args()


def unnormalize(image: torch.Tensor) -> torch.Tensor:
    """Reverse CIFAR-100 normalization so an image can be displayed.

    Args:
        image: A normalized CHW image tensor.

    Returns:
        The unnormalized image tensor, clamped to [0, 1].
    """
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1)
    std = torch.tensor(CIFAR100_STD).view(3, 1, 1)
    return (image * std + mean).clamp(0, 1)


def verify_labels(labels: list[int], num_classes: int) -> None:
    """Verify all classes are represented and every label is in range.

    Args:
        labels: The full list of integer labels for a dataset split.
        num_classes: The expected number of classes.

    Raises:
        ValueError: If any label falls outside ``[0, num_classes)`` or a
            class has zero samples.
    """
    label_set = set(labels)
    invalid = sorted(label for label in label_set if label < 0 or label >= num_classes)
    if invalid:
        raise ValueError(f"Found invalid label(s) outside [0, {num_classes}): {invalid}")

    missing_classes = sorted(set(range(num_classes)) - label_set)
    if missing_classes:
        raise ValueError(f"Missing classes in dataset: {missing_classes}")

    logger.info("Verified all %d classes are present with valid labels.", num_classes)


def save_sample_grid(
    dataset: Any, class_names: list[str], output_path: Path, num_images: int
) -> None:
    """Save a square grid of unnormalized sample images with class labels.

    Args:
        dataset: A CIFAR-100 dataset to draw samples from (first
            ``num_images`` items, in dataset order).
        class_names: Ordered list of class names for label lookup.
        output_path: Destination PNG file path.
        num_images: Number of images to display; must be a perfect square.

    Raises:
        ValueError: If ``num_images`` is not a perfect square.
    """
    grid_size = int(round(num_images**0.5))
    if grid_size * grid_size != num_images:
        raise ValueError(f"num_preview_images ({num_images}) must be a perfect square.")

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 1.5, grid_size * 1.5))
    for idx, ax in enumerate(axes.flat):
        image, label = dataset[idx]
        image_np = unnormalize(image).permute(1, 2, 0).numpy()
        ax.imshow(image_np)
        ax.set_title(class_names[label], fontsize=6)
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    logger.info("Saved sample grid to %s", output_path)


def save_class_counts_csv(labels: list[int], class_names: list[str], output_path: Path) -> Counter:
    """Save per-class training sample counts as a CSV file.

    Args:
        labels: The full list of integer training labels.
        class_names: Ordered list of class names.
        output_path: Destination CSV file path.

    Returns:
        The computed per-class ``Counter`` of label occurrences.
    """
    counts = Counter(labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_index", "class_name", "count"])
        for class_index, class_name in enumerate(class_names):
            writer.writerow([class_index, class_name, counts.get(class_index, 0)])
    logger.info("Saved per-class counts to %s", output_path)
    return counts


def save_summary_json(summary: dict[str, Any], output_path: Path) -> None:
    """Save a dataset summary dictionary as a JSON file.

    Args:
        summary: JSON-serializable summary dictionary.
        output_path: Destination JSON file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    logger.info("Saved dataset summary to %s", output_path)


def main() -> None:
    """Run the CIFAR-100 dataset inspection pipeline end to end."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()

    try:
        config = load_data_config(args.config)
    except ConfigError as exc:
        logger.error("Configuration error: %s", exc)
        raise SystemExit(1) from exc

    set_seed(config.seed)

    train_dataset, test_dataset = build_datasets(config.data_root, download=config.download)
    train_loader, _ = build_dataloaders(
        train_dataset,
        test_dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    class_names = get_class_names(train_dataset)
    train_summary = summarize_dataset(train_dataset)
    test_summary = summarize_dataset(test_dataset)

    print(f"Train samples: {train_summary.num_samples}")
    print(f"Test samples: {test_summary.num_samples}")
    print(f"Number of classes: {train_summary.num_classes}")
    print(f"Image tensor shape: {tuple(train_summary.image_shape)}")
    print(f"Label range: [{train_summary.label_min}, {train_summary.label_max}]")

    train_labels = list(train_dataset.targets)
    verify_labels(train_labels, num_classes=train_summary.num_classes)
    print("Label verification passed: all classes present, no invalid labels.")

    class_counts = Counter(train_labels)
    counts_str = ", ".join(
        f"{class_names[i]}={class_counts.get(i, 0)}" for i in range(train_summary.num_classes)
    )
    logger.info("Per-class training counts: %s", counts_str)

    images, labels = next(iter(train_loader))
    print(f"Loaded one training batch: images={tuple(images.shape)}, labels={tuple(labels.shape)}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    save_sample_grid(
        train_dataset,
        class_names,
        output_dir / "cifar100_samples.png",
        num_images=config.num_preview_images,
    )
    save_class_counts_csv(train_labels, class_names, output_dir / "cifar100_class_counts.csv")

    summary = {
        "dataset_name": config.dataset_name,
        "seed": config.seed,
        "train": {
            "num_samples": train_summary.num_samples,
            "num_classes": train_summary.num_classes,
            "image_shape": list(train_summary.image_shape),
            "label_min": train_summary.label_min,
            "label_max": train_summary.label_max,
        },
        "test": {
            "num_samples": test_summary.num_samples,
            "num_classes": test_summary.num_classes,
            "image_shape": list(test_summary.image_shape),
            "label_min": test_summary.label_min,
            "label_max": test_summary.label_max,
        },
        "class_names": class_names,
    }
    save_summary_json(summary, output_dir / "cifar100_summary.json")

    logger.info("Dataset inspection complete.")


if __name__ == "__main__":
    main()
