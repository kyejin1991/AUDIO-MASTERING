from __future__ import annotations
import torch
from torch import nn


class TransformerBottleneck(nn.Module):
    def __init__(self, channels: int, heads: int = 4, layers: int = 2, dropout: float = 0.0):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=max(1, heads),
            dim_feedforward=channels * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=max(1, layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [batch, channels, time] -> [batch, time, channels]
        seq = x.transpose(1, 2)
        seq = self.encoder(seq)
        return seq.transpose(1, 2)


