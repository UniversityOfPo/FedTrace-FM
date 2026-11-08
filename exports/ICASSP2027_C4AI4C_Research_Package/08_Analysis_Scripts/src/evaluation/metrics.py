"""Classification evaluation metrics.

``topk_accuracy`` is implemented with plain tensor ops so it is cheap enough
to call every batch during training/evaluation. The remaining metrics
(confusion matrix, per-class accuracy, macro precision/recall/F1, balanced
accuracy) operate on full prediction/label arrays collected after an
evaluation pass, and are intended for end-of-run reporting rather than
per-batch tracking. This module is independent of the training entry point
and of the training engine (the engine imports from here, not vice versa).
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix as sk_confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def topk_accuracy(
    outputs: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...] = (1, 5)
) -> tuple[float, ...]:
    """Compute top-k accuracy for one or more values of ``k``.

    Args:
        outputs: Model logits of shape ``[batch_size, num_classes]``.
        targets: Ground-truth class indices of shape ``[batch_size]``.
        ks: The k values to compute accuracy for. Any ``k`` larger than the
            number of classes is clamped down to the number of classes.

    Returns:
        A tuple of accuracies (fractions in [0, 1]), one per entry in ``ks``,
        in the same order.
    """
    max_k = min(max(ks), outputs.size(1))
    batch_size = targets.size(0)

    _, pred = outputs.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results = []
    for k in ks:
        k = min(k, max_k)
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append((correct_k / batch_size).item())
    return tuple(results)


def compute_confusion_matrix(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> np.ndarray:
    """Compute a ``[num_classes, num_classes]`` confusion matrix.

    Args:
        targets: Ground-truth class indices.
        predictions: Predicted class indices.
        num_classes: Total number of classes (fixes matrix shape even if a
            class is absent from this particular batch of labels).

    Returns:
        The confusion matrix, rows = true labels, columns = predicted labels.
    """
    return sk_confusion_matrix(targets, predictions, labels=list(range(num_classes)))


def per_class_accuracy(confusion: np.ndarray) -> np.ndarray:
    """Compute per-class accuracy (recall) from a confusion matrix.

    Classes with zero support (no true samples) are safely reported as 0.0
    accuracy rather than raising a division error or producing NaN.

    Args:
        confusion: A ``[num_classes, num_classes]`` confusion matrix.

    Returns:
        A ``[num_classes]`` array of per-class accuracies.
    """
    support = confusion.sum(axis=1)
    correct = np.diag(confusion)
    return np.where(support > 0, correct / np.maximum(support, 1), 0.0)


def macro_precision_recall_f1(
    targets: np.ndarray, predictions: np.ndarray, num_classes: int
) -> tuple[float, float, float]:
    """Compute macro-averaged precision, recall, and F1 score.

    Classes with no predicted or true samples contribute 0 to the macro
    average (``zero_division=0``) instead of raising or emitting warnings.

    Args:
        targets: Ground-truth class indices.
        predictions: Predicted class indices.
        num_classes: Total number of classes.

    Returns:
        A tuple ``(macro_precision, macro_recall, macro_f1)``.
    """
    labels = list(range(num_classes))
    precision = precision_score(targets, predictions, labels=labels, average="macro", zero_division=0)
    recall = recall_score(targets, predictions, labels=labels, average="macro", zero_division=0)
    f1 = f1_score(targets, predictions, labels=labels, average="macro", zero_division=0)
    return float(precision), float(recall), float(f1)


def compute_balanced_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    """Compute balanced accuracy (macro-averaged recall across classes).

    Args:
        targets: Ground-truth class indices.
        predictions: Predicted class indices.

    Returns:
        Balanced accuracy as a fraction in [0, 1].
    """
    return float(balanced_accuracy_score(targets, predictions))
