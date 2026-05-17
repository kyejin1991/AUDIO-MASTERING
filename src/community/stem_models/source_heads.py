from __future__ import annotations
import torch
from torch import nn


class SourceHeads(nn.Module):
    def __init__(self, channels: int, sources: tuple[str, ...], out_channels: int = 2):
        super().__init__()
        self.sources = tuple(sources)
        self.heads = nn.ModuleDict({
            source: nn.Conv1d(channels, out_channels, kernel_size=1)
            for source in self.sources
        })

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}


