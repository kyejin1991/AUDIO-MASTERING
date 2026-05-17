from __future__ import annotations
import torch
from torch import nn
from .model_config import StemModelConfig
from .waveform_branch import WaveformEncoder, WaveformDecoder
from .spectral_branch import SpectralEncoder, SpectralDecoder
from .cross_domain_transformer import CrossDomainTransformer


class HybridUNetCore(nn.Module):
    def __init__(self, config: StemModelConfig):
        super().__init__()
        self.config = config
        self.waveform_encoder = WaveformEncoder(config.channels, config.hidden_channels, config.depth)
        wav_channels = self.waveform_encoder.out_channels
        self.spectral_encoder = SpectralEncoder(config.channels, config.hidden_channels, config.n_fft, config.hop_length)
        self.cross_domain_transformer = CrossDomainTransformer(
            wav_channels=wav_channels,
            spec_channels=config.hidden_channels,
            hidden_channels=wav_channels,
            heads=config.transformer_heads,
            layers=config.transformer_layers,
        )
        self.waveform_decoder = WaveformDecoder(
            out_channels=config.hidden_channels,
            hidden_channels=config.hidden_channels,
            depth=config.depth,
        )
        self.spectral_decoder = SpectralDecoder(config.hidden_channels)
        self.output_channels = config.hidden_channels

    def forward(self, mixture: torch.Tensor) -> torch.Tensor:
        target_len = mixture.shape[-1]
        wav_features, skips = self.waveform_encoder(mixture)
        spec_features = self.spectral_encoder(mixture, target_frames=wav_features.shape[-1])
        fused = self.cross_domain_transformer(wav_features, spec_features)
        wav_out = self.waveform_decoder(fused, skips, target_len=target_len)
        spec_hint = self.spectral_decoder(spec_features, target_len=target_len)
        return wav_out + spec_hint


