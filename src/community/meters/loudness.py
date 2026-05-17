from __future__ import annotations
import numpy as np
from scipy import signal
from community.meters.common import db20

try:
    import pyloudnorm as pyln
except Exception:
    pyln = None

def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    if pyln is not None:
        try:
            return float(pyln.Meter(sr).integrated_loudness(audio))
        except Exception:
            pass
    rms = np.sqrt(np.mean(audio ** 2) + 1e-12)
    return float(db20(rms) - 1.0)

def windowed_lufs(audio: np.ndarray, sr: int, window_sec: float) -> list[float]:
    frame_size = max(1, int(sr * window_sec))
    vals = []
    for start in range(0, len(audio), frame_size):
        chunk = audio[start:start+frame_size]
        if len(chunk) < sr * 0.05:
            continue
        vals.append(integrated_lufs(chunk, sr))
    return vals or [integrated_lufs(audio, sr)]

def true_peak_dbtp(audio: np.ndarray, sr: int, oversample: int = 4) -> float:
    up = signal.resample_poly(audio, oversample, 1, axis=0)
    peak = np.max(np.abs(up)) if up.size else 0.0
    return float(db20(peak))
