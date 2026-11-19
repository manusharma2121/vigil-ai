"""
VIGIL AI – 3D CNN Architecture
Lightweight 3D Convolutional Neural Network for video classification.

Architecture:
  - 4 × Conv3D blocks with BatchNorm and MaxPool3D
  - Global Average Pooling
  - FC classification head

Input shape : (batch, C, frames, H, W)  e.g. (8, 3, 16, 112, 112)
Output shape: (batch, 2) – logits for [Non-Violent, Violent]
"""
import torch
import torch.nn as nn


class Conv3DBlock(nn.Module):
    """3D Convolution + BatchNorm + ReLU + optional MaxPool."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple = (3, 3, 3),
        padding: tuple = (1, 1, 1),
        pool: bool = True,
        pool_kernel: tuple = (1, 2, 2),
    ):
        super().__init__()
        layers = [
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool3d(kernel_size=pool_kernel, stride=pool_kernel))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Model3DCNN(nn.Module):
    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super().__init__()

        self.features = nn.Sequential(
            Conv3DBlock(3, 32, pool_kernel=(1, 2, 2)),          # 32 channels
            Conv3DBlock(32, 64, pool_kernel=(2, 2, 2)),         # 64 channels
            Conv3DBlock(64, 128, pool_kernel=(2, 2, 2)),        # 128 channels
            Conv3DBlock(128, 256, pool_kernel=(2, 2, 2)),       # 256 channels
        )

        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, C, frames, H, W)
        Returns:
            logits: (batch, num_classes)
        """
        x = self.features(x)       # (B, 256, t', h', w')
        x = self.global_pool(x)    # (B, 256, 1, 1, 1)
        logits = self.classifier(x)
        return logits


def build_3dcnn(num_classes: int = 2, **kwargs) -> Model3DCNN:
    """Factory function for 3D-CNN model."""
    return Model3DCNN(num_classes=num_classes, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
