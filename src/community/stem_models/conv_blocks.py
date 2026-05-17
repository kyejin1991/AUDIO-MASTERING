from __future__ import annotations
import torch
from torch import nn


class ConvNormAct1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            ConvNormAct1d(channels, channels, kernel_size),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class UpBlock1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.proj = ConvNormAct1d(in_channels, out_channels)

    def forward(self, x: torch.Tensor, target_len: int) -> torch.Tensor:
        x = torch.nn.functional.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return self.proj(x)


