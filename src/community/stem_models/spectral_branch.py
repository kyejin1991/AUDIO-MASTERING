from __future__ import annotations
import torch
from torch import nn
from .stft import TorchSTFT


class SpectralEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, hidden_channels: int = 32, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.stft = TorchSTFT(n_fft=n_fft, hop_length=hop_length)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden_channels),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))

    def forward(self, audio: torch.Tensor, target_frames: int) -> torch.Tensor:
        spec = self.stft(audio)
        features = self.net(spec)
        features = self.pool(features).squeeze(-2)
        return torch.nn.functional.interpolate(features, size=target_frames, mode="linear", align_corners=False)


class SpectralDecoder(nn.Module):
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        self.mask_hint = nn.Sequential(
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1),
        )

    def forward(self, fused: torch.Tensor, target_len: int) -> torch.Tensor:
        hint = self.mask_hint(fused)
        return torch.nn.functional.interpolate(hint, size=target_len, mode="linear", align_corners=False)


