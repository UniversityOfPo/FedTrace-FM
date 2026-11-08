"""CIFAR-compatible ResNet-18 (He et al., adapted stem for 32x32 inputs).

Differs from the standard torchvision ImageNet ResNet-18 only in the stem:
a 3x3 stride-1 convolution replaces the 7x7 stride-2 convolution, and the
initial 3x3 stride-2 max-pool is removed. CIFAR images are 32x32, and the
ImageNet stem would downsample them to 8x8 before the first residual stage
even runs, discarding most spatial detail. The residual block structure
(two BasicBlocks per stage, four stages) is otherwise the standard
ResNet-18 architecture. No pretrained (ImageNet) weights are used.
"""

from __future__ import annotations

import torch
from torch import nn


class BasicBlock(nn.Module):
    """Standard ResNet basic residual block (two 3x3 convolutions)."""

    expansion: int = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """Build one residual block.

        Args:
            in_channels: Number of input feature-map channels.
            out_channels: Number of output feature-map channels.
            stride: Stride of the first convolution; > 1 downsamples spatially.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample: nn.Module | None = None
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels * self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block to input ``x``."""
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return self.relu(out)


class ResNetCifar(nn.Module):
    """CIFAR-compatible ResNet built from stacked ``BasicBlock``s."""

    def __init__(self, layers: list[int], num_classes: int = 100) -> None:
        """Build the network.

        Args:
            layers: Number of ``BasicBlock``s in each of the four stages
                (e.g. ``[2, 2, 2, 2]`` for ResNet-18).
            num_classes: Number of output classes for the final linear layer.
        """
        super().__init__()
        self.in_channels = 64

        # CIFAR stem: 3x3 stride-1 convolution, no max-pool (see module docstring).
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, layers[0], stride=1)
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, out_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
        """Build one stage of ``num_blocks`` residual blocks."""
        strides = [stride] + [1] * (num_blocks - 1)
        blocks: list[nn.Module] = []
        for block_stride in strides:
            blocks.append(BasicBlock(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * BasicBlock.expansion
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Initialize convolution weights (Kaiming) and BatchNorm affine params."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass, returning class logits of shape [N, num_classes]."""
        out = self.stem(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return self.fc(out)


def resnet18_cifar(num_classes: int = 100) -> ResNetCifar:
    """Build a randomly initialized, CIFAR-compatible ResNet-18.

    Args:
        num_classes: Number of output classes (defaults to 100 for CIFAR-100).

    Returns:
        A ``ResNetCifar`` instance with the standard ResNet-18 layer
        configuration ``[2, 2, 2, 2]``. No pretrained weights are loaded.
    """
    return ResNetCifar(layers=[2, 2, 2, 2], num_classes=num_classes)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model.

    Args:
        model: The model to inspect.

    Returns:
        A tuple ``(total_parameters, trainable_parameters)``.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
