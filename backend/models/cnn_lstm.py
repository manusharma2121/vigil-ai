"""
VIGIL AI – CNN + LSTM Architecture
Lightweight video classification model suitable for CPU inference.

Architecture:
  - MobileNetV2 (pretrained ImageNet) as the frame-level CNN encoder
  - Bidirectional LSTM for temporal sequence learning
  - FC head with dropout for binary classification

Input shape : (batch, frames, C, H, W)  e.g. (8, 16, 3, 112, 112)
Output shape: (batch, 2) – logits for [Non-Violent, Violent]
"""
import torch
import torch.nn as nn
import torchvision.models as models


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True,
        freeze_cnn: bool = False,
    ):
        super().__init__()

        # ── CNN Encoder (MobileNetV2 backbone) ───────────────────────────────
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        # Remove the final classifier; keep the feature extractor
        self.cnn = mobilenet.features
        self.cnn_pool = nn.AdaptiveAvgPool2d((1, 1))
        cnn_out_size = 1280  # MobileNetV2 final feature channels

        if freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False

        # ── Temporal Projection ───────────────────────────────────────────────
        self.temporal_proj = nn.Sequential(
            nn.Linear(cnn_out_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Bidirectional LSTM ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_out = hidden_size * (2 if bidirectional else 1)

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, frames, C, H, W)
        Returns:
            logits: (batch, num_classes)
        """
        batch_size, num_frames, C, H, W = x.shape

        # Flatten batch and frames for CNN processing
        x = x.view(batch_size * num_frames, C, H, W)
        features = self.cnn(x)                         # (B*T, 1280, h, w)
        features = self.cnn_pool(features)             # (B*T, 1280, 1, 1)
        features = features.view(batch_size * num_frames, -1)  # (B*T, 1280)

        # Project to smaller temporal dimension
        features = self.temporal_proj(features)        # (B*T, 512)
        features = features.view(batch_size, num_frames, 512)  # (B, T, 512)

        # Temporal learning with LSTM
        lstm_out, _ = self.lstm(features)              # (B, T, hidden*2)
        # Use last time step
        last_hidden = lstm_out[:, -1, :]               # (B, hidden*2)

        # Classify
        logits = self.classifier(last_hidden)          # (B, num_classes)
        return logits


def build_cnn_lstm(
    num_classes: int = 2,
    freeze_cnn: bool = False,
    **kwargs,
) -> CNNLSTM:
    """Factory function for CNN+LSTM model."""
    return CNNLSTM(num_classes=num_classes, freeze_cnn=freeze_cnn, **kwargs)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
