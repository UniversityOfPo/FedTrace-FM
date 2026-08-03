"""Lightweight tests for the CIFAR-compatible ResNet-18 model.

No training is run here; these only check architecture and shape
properties, so they execute quickly on CPU.
"""

from __future__ import annotations

import torch
from torch import nn

from src.models.resnet18_cifar import count_parameters, resnet18_cifar


def test_output_shape_default_num_classes() -> None:
    """Model output shape is [batch_size, 100] with the default class count."""
    model = resnet18_cifar(num_classes=100)
    model.eval()
    images = torch.randn(4, 3, 32, 32)
    with torch.no_grad():
        outputs = model(images)
    assert outputs.shape == (4, 100)


def test_output_shape_custom_num_classes() -> None:
    """Model output shape follows a custom num_classes value."""
    model = resnet18_cifar(num_classes=10)
    model.eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        outputs = model(images)
    assert outputs.shape == (2, 10)


def test_stem_is_cifar_compatible() -> None:
    """The stem uses a 3x3, stride-1, padding-1 convolution (not the 7x7/stride-2 ImageNet stem)."""
    model = resnet18_cifar(num_classes=100)
    stem_conv = model.stem[0]
    assert isinstance(stem_conv, nn.Conv2d)
    assert stem_conv.kernel_size == (3, 3)
    assert stem_conv.stride == (1, 1)
    assert stem_conv.padding == (1, 1)


def test_no_max_pooling_layer() -> None:
    """No nn.MaxPool2d module exists anywhere in the network."""
    model = resnet18_cifar(num_classes=100)
    assert not any(isinstance(module, nn.MaxPool2d) for module in model.modules())


def test_parameter_count_nonzero() -> None:
    """Parameter counting reports nonzero total and trainable counts."""
    model = resnet18_cifar(num_classes=100)
    total, trainable = count_parameters(model)
    assert total > 0
    assert trainable > 0
    assert trainable <= total
