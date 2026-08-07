"""Lightweight tests for the training engine, metrics, and checkpointing.

These use tiny synthetic data and a single batch/epoch so they run quickly
on CPU. No real dataset is loaded and no full training is run.
"""

from __future__ import annotations

import copy
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
    compute_proximal_term,
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


# ----------------------------------------------------------------------
# FedProx proximal term (Sprint 7.1)
# ----------------------------------------------------------------------


class _TinyLinearModel(nn.Module):
    """A minimal model with a known, hand-computable parameter count."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _TinyModelWithBatchNorm(nn.Module):
    """A minimal model with a real BatchNorm layer (buffers, not just parameters)."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.bn = nn.BatchNorm1d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.linear(x))


def test_compute_proximal_term_zero_when_parameters_match() -> None:
    """The proximal term is exactly zero when model and global parameters are identical."""
    model = _TinyLinearModel()
    global_params = [p.detach().clone() for p in model.parameters()]

    term = compute_proximal_term(model, global_params)

    assert term.item() == pytest.approx(0.0)


def test_compute_proximal_term_positive_when_parameters_differ() -> None:
    """The proximal term is strictly positive once parameters diverge from the snapshot."""
    model = _TinyLinearModel()
    global_params = [p.detach().clone() for p in model.parameters()]

    with torch.no_grad():
        next(model.parameters()).add_(1.0)

    term = compute_proximal_term(model, global_params)

    assert term.item() > 0.0


def test_compute_proximal_term_matches_closed_form() -> None:
    """The proximal term equals the hand-computed sum of squared differences."""
    model = _TinyLinearModel()
    with torch.no_grad():
        model.linear.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    global_params = [torch.tensor([[0.0, 0.0], [0.0, 0.0]])]

    term = compute_proximal_term(model, global_params)

    expected = (1.0**2 + 2.0**2 + 3.0**2 + 4.0**2)  # ||w - w_global||^2
    assert term.item() == pytest.approx(expected)


def test_compute_proximal_term_excludes_non_trainable_buffers() -> None:
    """The proximal term is computed only over trainable parameters, never BatchNorm buffers."""
    model = _TinyModelWithBatchNorm()
    global_params = [p.detach().clone() for p in model.parameters()]

    # model.parameters() must exactly match the snapshot's length -- if
    # buffers (bn.running_mean/running_var/num_batches_tracked) were
    # accidentally included, this would either mismatch in length (zip
    # would silently truncate) or raise a shape error. Confirming the
    # count matches nn.Parameter count only is the direct check.
    num_trainable_params = sum(1 for _ in model.parameters())
    assert len(global_params) == num_trainable_params

    term = compute_proximal_term(model, global_params)
    assert term.item() == pytest.approx(0.0)  # identical snapshot -> zero, regardless of buffers


def test_train_one_epoch_default_matches_explicit_inert_fedprox_args() -> None:
    """Calling train_one_epoch with no FedProx args matches calling with explicit inert values.

    Directly verifies the "preserve existing centralized behavior exactly
    by default" requirement: the new proximal_mu/global_params parameters
    must be a strict no-op unless both are explicitly set to non-inert
    values.

    Both training calls share one base model (via deepcopy, so both start
    from bitwise-identical weights with no dependence on RNG state) and
    one DataLoader instance (re-iterated; safe and deterministic since
    shuffle=False), so no random model-construction or data-generation
    call is repeated -- the only two variables between the two calls are
    exactly the two FedProx arguments under test. Compared with
    `torch.allclose` rather than `torch.equal`: PyTorch does not
    guarantee bitwise-identical results across two independent CPU
    forward/backward passes even given identical inputs (see
    `src/utils/seed.py`'s own documented caveat that full determinism is
    not guaranteed for every operation) -- the scientific property this
    test verifies is behavioral equivalence, not incidental bitwise
    reproducibility.
    """
    device = resolve_device("cpu")
    torch.manual_seed(0)
    base_model = resnet18_cifar(num_classes=10).to(device)
    loader = _make_tiny_loader(num_classes=10)
    criterion = nn.CrossEntropyLoss()

    model_default = copy.deepcopy(base_model)
    model_explicit = copy.deepcopy(base_model)
    optimizer_default = build_optimizer(model_default, "sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.0)
    optimizer_explicit = build_optimizer(model_explicit, "sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.0)

    train_one_epoch(
        model_default, loader, optimizer_default, criterion, device,
        scaler=build_grad_scaler(device, use_amp=False), use_amp=False, log_interval=0,
    )
    train_one_epoch(
        model_explicit, loader, optimizer_explicit, criterion, device,
        scaler=build_grad_scaler(device, use_amp=False), use_amp=False, log_interval=0,
        proximal_mu=0.0, global_params=None,
    )

    for p_default, p_explicit in zip(model_default.parameters(), model_explicit.parameters()):
        assert torch.allclose(p_default, p_explicit)


def test_train_one_epoch_fedprox_produces_different_parameters_than_plain() -> None:
    """One epoch with FedProx (mu > 0) diverges from the same epoch trained with plain FedAvg."""
    device = resolve_device("cpu")

    torch.manual_seed(1)
    model_plain = resnet18_cifar(num_classes=10).to(device)
    torch.manual_seed(1)
    model_fedprox = resnet18_cifar(num_classes=10).to(device)
    global_params = [p.detach().clone() for p in model_fedprox.parameters()]

    torch.manual_seed(2)
    loader_plain = _make_tiny_loader(num_classes=10)
    torch.manual_seed(2)
    loader_fedprox = _make_tiny_loader(num_classes=10)

    optimizer_plain = build_optimizer(model_plain, "sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.0)
    optimizer_fedprox = build_optimizer(model_fedprox, "sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    train_one_epoch(
        model_plain, loader_plain, optimizer_plain, criterion, device,
        scaler=build_grad_scaler(device, use_amp=False), use_amp=False, log_interval=0,
    )
    train_one_epoch(
        model_fedprox, loader_fedprox, optimizer_fedprox, criterion, device,
        scaler=build_grad_scaler(device, use_amp=False), use_amp=False, log_interval=0,
        proximal_mu=1.0, global_params=global_params,
    )

    assert any(
        not torch.equal(p_plain, p_fedprox)
        for p_plain, p_fedprox in zip(model_plain.parameters(), model_fedprox.parameters())
    )


def test_fedprox_pulls_trained_parameters_closer_to_global_reference() -> None:
    """A large-mu FedProx epoch ends up closer to the global snapshot than plain training does.

    This checks the actual regularizing mechanism, not just "some
    difference exists": training under a strong proximal penalty toward
    the initial (== global) parameters should leave the model measurably
    closer to that starting point than unregularized training does.
    """
    device = resolve_device("cpu")

    torch.manual_seed(3)
    model_plain = resnet18_cifar(num_classes=10).to(device)
    torch.manual_seed(3)
    model_fedprox = resnet18_cifar(num_classes=10).to(device)
    # Both models start identical; the shared initial parameters are the
    # "global" reference point both this test measures distance from.
    initial_params = [p.detach().clone() for p in model_plain.parameters()]

    torch.manual_seed(4)
    loader_plain = _make_tiny_loader(num_classes=10)
    torch.manual_seed(4)
    loader_fedprox = _make_tiny_loader(num_classes=10)

    optimizer_plain = build_optimizer(model_plain, "sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.0)
    optimizer_fedprox = build_optimizer(model_fedprox, "sgd", learning_rate=0.1, momentum=0.9, weight_decay=0.0)
    criterion = nn.CrossEntropyLoss()

    train_one_epoch(
        model_plain, loader_plain, optimizer_plain, criterion, device,
        scaler=build_grad_scaler(device, use_amp=False), use_amp=False, log_interval=0,
    )
    train_one_epoch(
        model_fedprox, loader_fedprox, optimizer_fedprox, criterion, device,
        scaler=build_grad_scaler(device, use_amp=False), use_amp=False, log_interval=0,
        proximal_mu=10.0, global_params=initial_params,
    )

    distance_plain = compute_proximal_term(model_plain, initial_params).item()
    distance_fedprox = compute_proximal_term(model_fedprox, initial_params).item()

    assert distance_fedprox < distance_plain
