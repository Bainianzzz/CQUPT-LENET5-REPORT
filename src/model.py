from __future__ import annotations

import torch
from torch import nn


class LeNet5(nn.Module):
    """
    LeNet-5 (MNIST variant):
    Input: (N, 1, 28, 28)
    Conv(5x5) -> ReLU -> AvgPool(2x2)
    Conv(5x5) -> ReLU -> AvgPool(2x2)
    Flatten -> FC -> ReLU -> FC -> ReLU -> FC(10)
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),  # 28->24
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 24->12
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),  # 12->8
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2, stride=2),  # 8->4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),  # 16*4*4 = 256
            nn.Linear(16 * 4 * 4, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

