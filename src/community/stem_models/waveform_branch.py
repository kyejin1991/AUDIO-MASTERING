from __future__ import annotations
import torch
from torch import nn
from .conv_blocks import ConvNormAct1d, ResidualBlock1d, UpBlock1d


class WaveformEncoder(nn.Module):
    def __init__(self, in_channels: int = 2, hidden_channels: int = 32, depth: int = 3):
        super().__init__()
        blocks = []
        channels = in_channels
        for idx in range(depth):
            out = hidden_channels * (2 ** idx)
            blocks.append(ConvNormAct1d(channels, out, kernel_size=7, stride=2))
            blocks.append(ResidualBlock1d(out))
            channels = out
        self.net = nn.ModuleList(blocks)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skips = []
        for block in self.net:
            x = block(x)
            if isinstance(block, ResidualBlock1d):
                skips.append(x)
        return x, skips


class WaveformDecoder(nn.Module):
    def __init__(self, out_channels: int = 2, hidden_channels: int = 32, depth: int = 3):
        super().__init__()
        channels = hidden_channels * (2 ** (depth - 1))
        blocks = []
        for idx in reversed(range(depth)):
            out = hidden_channels * (2 ** max(idx - 1, 0))
            if idx == 0:
                out = hidden_channels
            blocks.append(UpBlock1d(channels, out))
            channels = out
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Conv1d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, skips: list[torch.Tensor], target_len: int) -> torch.Tensor:
        for idx, block in enumerate(self.blocks):
            skip_idx = len(skips) - idx - 2
            next_len = target_len if idx == len(self.blocks) - 1 else skips[max(0, skip_idx)].shape[-1]
            x = block(x, next_len)
            if 0 <= skip_idx < len(skips) and skips[skip_idx].shape[1] == x.shape[1]:
                x = x + skips[skip_idx][..., : x.shape[-1]]
        return self.proj(x)[..., :target_len]


