from __future__ import annotations
import torch
from torch import nn


class TorchSTFT(nn.Module):
    def __init__(self, n_fft: int = 1024, hop_length: int = 256):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.register_buffer("window", torch.hann_window(self.n_fft), persistent=False)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # audio: [batch, channels, samples]
        b, c, t = audio.shape
        flat = audio.reshape(b * c, t)
        spec = torch.stft(
            flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window.to(audio.device),
            return_complex=True,
            center=True,
        )
        mag = torch.log1p(spec.abs())
        return mag.reshape(b, c, mag.shape[-2], mag.shape[-1])

    def inverse(self, spec: torch.Tensor, length: int) -> torch.Tensor:
        raise NotImplementedError("Task 031C keeps iSTFT reconstruction in the waveform decoder path.")


