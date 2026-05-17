from __future__ import annotations
import numpy as np
from scipy import signal


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if int(source_sr) == int(target_sr):
        return audio
    ratio = float(target_sr) / float(source_sr)
    target_len = max(1, int(round(audio.shape[0] * ratio)))
    return signal.resample_poly(audio, target_sr, source_sr, axis=0)[:target_len]


