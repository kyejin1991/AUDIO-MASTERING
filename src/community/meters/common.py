import numpy as np

EPS = 1e-12

def db20(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x), EPS))

def undb20(x):
    return 10.0 ** (x / 20.0)

def mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)

def frame_audio(x: np.ndarray, frame_size: int, hop_size: int):
    if len(x) < frame_size:
        return
    for start in range(0, len(x) - frame_size + 1, hop_size):
        yield x[start:start+frame_size]


