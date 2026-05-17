from __future__ import annotations
import torch
from torch import nn
from .model_config import StemModelConfig
from .hybrid_unet import HybridUNetCore
from .source_heads import SourceHeads


class HybridStemSeparator(nn.Module):
    """
    Compact HT-Demucs-like skeleton for Task 031C.

    It includes waveform and spectrogram branches, a cross-domain Transformer
    bottleneck, skip-connected decoding, and named source heads. The bundled
    default weights are intentionally untrained; real separation quality requires
    a trained checkpoint loaded through weight_loader.py.
    """

    def __init__(self, config: StemModelConfig):
        super().__init__()
        self.config = config
        self.core = HybridUNetCore(config)
        self.source_heads = SourceHeads(config.hidden_channels, config.sources, out_channels=config.channels)
        self._init_safe_untrained_heads()

    def _init_safe_untrained_heads(self):
        if not self.config.untrained:
            return
        for head in self.source_heads.heads.values():
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head.bias)

    def forward(self, mixture: torch.Tensor) -> dict[str, torch.Tensor]:
        if mixture.ndim != 3:
            raise ValueError("mixture must have shape [batch, channels, samples]")
        if mixture.shape[1] != self.config.channels:
            raise ValueError(f"expected {self.config.channels} channels, got {mixture.shape[1]}")
        features = self.core(mixture)
        stems = self.source_heads(features)
        return {name: torch.tanh(stem)[..., : mixture.shape[-1]] for name, stem in stems.items()}


