"""Lightweight tests for the training engine, metrics, and checkpointing.

These use tiny synthetic data and a single batch/epoch so they run quickly
on CPU. No real dataset is loaded and no full training is run.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.metrics import topk_accuracy
from src.models.resnet18_cifar import resnet18_cifar
from src.training.engine import (
    build_grad_scaler,
    build_optimizer,
    evaluate,
    resolve_device,
    train_one_epoch,
)
from src.utils.checkpoint import build_checkpoint, load_checkpoint, restore_training_state, save_checkpoint


def _make_tiny_loader(num_samples: int = 8, num_classes: int = 10) -> DataLoader:
    """Build a tiny synthetic image/label DataLoader for fast CPU tests."""
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, num_classes, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=4, shuffle=False)


def test_topk_accuracy_synthetic() -> None:
    """Top-1/top-k accuracy match hand-computed values on a tiny example."""
    outputs = torch.tensor(
        [
            [10.0, 1.0, 2.0, 3.0],  # top-1 pred=0, correct
            [1.0, 2.0, 10.0, 3.0],  # top-1 pred=2, target=1 -> wrong
            [1.0, 2.0, 3.0, 10.0],  # top-1 pred=3, correct
        ]
    )
    targets = torch.tensor([0, 1, 3])

    top1, top5 = topk_accuracy(outputs, targets, ks=(1, 5))
    assert top1 == pytest.approx(2 / 3)
    assert top5 == pytest.approx(1.0)  # k clamped to num_classes=4, so always correct


def test_resolve_device_cpu() -> None:
    """resolve_device('cpu') always returns a CPU device."""
    device = resolve_device("cpu")
    assert device.type == "cpu"


def test_train_one_epoch_updates_parameters() -> None:
    """One training epoch changes at least one model parameter."""
    device = resolve_device("cpu")
    model = resnet18_cifar(num_classes=10).to(device)
    loader = _make_tiny_loader(num_classes=10)
    optimizer = build_optimizer(model, "sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()
    scaler = build_grad_scaler(device, use_amp=False)

    before = [p.detach().clone() for p in model.parameters()]
    train_one_epoch(
        model, loader, optimizer, criterion, device, scaler=scaler, use_amp=False, log_interval=0
    )
    after = [p.detach().clone() for p in model.parameters()]

    assert any(not torch.equal(b, a) for b, a in zip(before, after))


def test_evaluate_does_not_update_parameters() -> None:
    """One evaluation epoch leaves every model parameter unchanged."""
    device = resolve_device("cpu")
    model = resnet18_cifar(num_classes=10).to(device)
    loader = _make_tiny_loader(num_classes=10)
    criterion = nn.CrossEntropyLoss()

    before = [p.detach().clone() for p in model.parameters()]
    evaluate(model, loader, criterion, device, use_amp=False)
    after = [p.detach().clone() for p in model.parameters()]

    assert all(torch.equal(b, a) for b, a in zip(before, after))


def test_mixed_precision_disabled_on_cpu() -> None:
    """The gradient scaler is disabled on CPU even when use_amp=True."""
    device = resolve_device("cpu")
    scaler = build_grad_scaler(device, use_amp=True)
    assert scaler.is_enabled() is False


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    """A saved checkpoint reloads into a fresh model with identical weights."""
    device = resolve_device("cpu")
    model = resnet18_cifar(num_classes=10).to(device)
    optimizer = build_optimizer(model, "sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.0)

    payload = build_checkpoint(
        epoch=0,
        model=model,
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        best_val_top1=0.5,
        config={"example": True},
        seed=42,
        model_name="resnet18_cifar",
        num_classes=10,
    )
    checkpoint_path = tmp_path / "test.pt"
    save_checkpoint(payload, checkpoint_path)
    assert checkpoint_path.exists()

    loaded = load_checkpoint(checkpoint_path, map_location="cpu")
    assert loaded["epoch"] == 0
    assert loaded["best_val_top1"] == 0.5
    assert loaded["seed"] == 42

    new_model = resnet18_cifar(num_classes=10).to(device)
    restore_training_state(loaded, new_model)

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.equal(p1, p2)
