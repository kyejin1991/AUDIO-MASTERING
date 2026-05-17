from __future__ import annotations
import numpy as np


def peak_normalize(audio: np.ndarray, peak_dbfs: float = -1.0) -> tuple[np.ndarray, float]:
    peak = float(np.max(np.abs(audio)) + 1e-12)
    target = float(10 ** (peak_dbfs / 20.0))
    gain = min(1.0, target / peak)
    return (audio * gain).astype(np.float32), gain


def standardize_audio(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    x = np.asarray(audio, dtype=np.float32)
    if x.ndim == 1:
        x = np.repeat(x[:, None], 2, axis=1)
    if x.shape[1] == 1:
        x = np.repeat(x, 2, axis=1)
    elif x.shape[1] > 2:
        x = x[:, :2]
    mean = np.mean(x, axis=0, keepdims=True)
    scale = float(np.max(np.abs(x - mean)) + 1e-6)
    return (x - mean) / scale, {"mean": mean.astype(np.float32), "scale": scale}


def restore_audio(audio: np.ndarray, stats: dict) -> np.ndarray:
    return audio * float(stats["scale"]) + stats["mean"]


