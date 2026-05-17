from __future__ import annotations
import torch
from torch import nn
from .attention import TransformerBottleneck


class CrossDomainTransformer(nn.Module):
    def __init__(self, wav_channels: int, spec_channels: int, hidden_channels: int, heads: int = 4, layers: int = 2):
        super().__init__()
        self.wav_proj = nn.Conv1d(wav_channels, hidden_channels, kernel_size=1)
        self.spec_proj = nn.Conv1d(spec_channels, hidden_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            nn.Conv1d(hidden_channels * 2, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
        )
        self.transformer = TransformerBottleneck(hidden_channels, heads=heads, layers=layers)

    def forward(self, wav_features: torch.Tensor, spec_features: torch.Tensor) -> torch.Tensor:
        if spec_features.shape[-1] != wav_features.shape[-1]:
            spec_features = torch.nn.functional.interpolate(spec_features, size=wav_features.shape[-1], mode="linear", align_corners=False)
        wav = self.wav_proj(wav_features)
        spec = self.spec_proj(spec_features)
        return self.transformer(self.fuse(torch.cat([wav, spec], dim=1)))


