from __future__ import annotations
import numpy as np
from scipy import signal

EPS = 1e-12

def sanitize(audio: np.ndarray) -> np.ndarray:
    return np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

def clip_audio(audio: np.ndarray, limit: float = 1.0) -> np.ndarray:
    return np.clip(sanitize(audio), -limit, limit)

def db20(x):
    return 20.0 * np.log10(np.maximum(np.asarray(x), EPS))

def undb20(x):
    return 10.0 ** (x / 20.0)

def mono(audio):
    return audio if audio.ndim == 1 else np.mean(audio, axis=1)


def oversample_audio(audio, factor: int):
    factor = int(max(1, factor))
    if factor == 1:
        return sanitize(audio)
    return signal.resample_poly(sanitize(audio), factor, 1, axis=0)


def downsample_audio(audio, factor: int):
    factor = int(max(1, factor))
    if factor == 1:
        return sanitize(audio)
    return signal.resample_poly(sanitize(audio), 1, factor, axis=0)


def smooth_control_signal(values: np.ndarray, sr: int, attack_ms: float = 5.0, release_ms: float = 120.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    attack = np.exp(-1.0 / max(1.0, (attack_ms / 1000.0) * sr))
    release = np.exp(-1.0 / max(1.0, (release_ms / 1000.0) * sr))
    out = np.zeros_like(values)
    prev = 0.0
    for i, x in enumerate(values):
        coeff = attack if x > prev else release
        prev = coeff * prev + (1.0 - coeff) * x
        out[i] = prev
    return out

def highpass(audio, sr, cutoff_hz, order=2):
    cutoff_hz = max(5.0, min(float(cutoff_hz), sr/2 - 100))
    sos = signal.butter(order, cutoff_hz, btype="highpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)

def lowpass(audio, sr, cutoff_hz, order=2):
    cutoff_hz = max(20.0, min(float(cutoff_hz), sr/2 - 100))
    sos = signal.butter(order, cutoff_hz, btype="lowpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)

def bandpass(audio, sr, low_hz, high_hz, order=2):
    low_hz = max(10.0, float(low_hz))
    high_hz = min(float(high_hz), sr/2 - 100)
    if low_hz >= high_hz:
        return np.zeros_like(audio)
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)

def peaking_eq(audio, sr, freq, gain_db, q=1.0):
    if abs(float(gain_db)) < 1e-4:
        return audio
    freq = max(20.0, min(float(freq), sr/2 - 200))
    q = max(0.1, float(q))
    A = 10 ** (float(gain_db) / 40.0)
    w0 = 2 * np.pi * freq / sr
    alpha = np.sin(w0) / (2 * q)
    cosw = np.cos(w0)
    b0 = 1 + alpha * A
    b1 = -2 * cosw
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw
    a2 = 1 - alpha / A
    b = np.array([b0, b1, b2]) / a0
    a = np.array([1, a1/a0, a2/a0])
    return signal.filtfilt(b, a, audio, axis=0)

def highshelf(audio, sr, freq, gain_db, slope=0.8):
    if abs(float(gain_db)) < 1e-4:
        return audio
    freq = max(200.0, min(float(freq), sr/2 - 200))
    A = 10 ** (float(gain_db) / 40.0)
    w0 = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w0), np.sin(w0)
    slope = max(0.1, float(slope))
    alpha = sinw/2 * np.sqrt((A + 1/A) * (1/slope - 1) + 2)
    beta = 2*np.sqrt(A)*alpha
    b0 = A*((A+1)+(A-1)*cosw+beta)
    b1 = -2*A*((A-1)+(A+1)*cosw)
    b2 = A*((A+1)+(A-1)*cosw-beta)
    a0 = (A+1)-(A-1)*cosw+beta
    a1 = 2*((A-1)-(A+1)*cosw)
    a2 = (A+1)-(A-1)*cosw-beta
    b = np.array([b0,b1,b2])/a0
    a = np.array([1,a1/a0,a2/a0])
    return signal.filtfilt(b,a,audio,axis=0)

def lowshelf(audio, sr, freq, gain_db, slope=0.8):
    if abs(float(gain_db)) < 1e-4:
        return audio
    freq = max(20.0, min(float(freq), sr/2 - 200))
    A = 10 ** (float(gain_db) / 40.0)
    w0 = 2*np.pi*freq/sr
    cosw, sinw = np.cos(w0), np.sin(w0)
    slope = max(0.1, float(slope))
    alpha = sinw/2 * np.sqrt((A + 1/A) * (1/slope - 1) + 2)
    beta = 2*np.sqrt(A)*alpha
    b0 = A*((A+1)-(A-1)*cosw+beta)
    b1 = 2*A*((A-1)-(A+1)*cosw)
    b2 = A*((A+1)-(A-1)*cosw-beta)
    a0 = (A+1)+(A-1)*cosw+beta
    a1 = -2*((A-1)+(A+1)*cosw)
    a2 = (A+1)+(A-1)*cosw-beta
    b = np.array([b0,b1,b2])/a0
    a = np.array([1,a1/a0,a2/a0])
    return signal.filtfilt(b,a,audio,axis=0)

def mid_side(audio):
    l, r = audio[:, 0], audio[:, 1]
    mid = (l + r) * 0.5
    side = (l - r) * 0.5
    return mid, side

def from_mid_side(mid, side):
    return np.stack([mid + side, mid - side], axis=1)

def apply_gain_db(audio, gain_db):
    return audio * undb20(gain_db)

def static_compress(x, threshold_db=-18.0, ratio=2.0, makeup_db=0.0):
    mag = np.abs(x) + EPS
    level = db20(mag)
    over = np.maximum(level - threshold_db, 0.0)
    gr = over * (1.0 - 1.0 / max(float(ratio), 1.0))
    gain = undb20(-gr + makeup_db)
    return x * gain, float(np.max(gr)) if np.size(gr) else 0.0

def soft_limiter(audio, ceiling_db=-1.0):
    ceiling = undb20(ceiling_db)
    peak = float(np.max(np.abs(audio)) + EPS)
    if peak <= ceiling:
        return audio, 0.0
    drive = max(1.0, peak / ceiling)
    y = np.tanh(audio * drive) / np.tanh(drive)
    peak2 = float(np.max(np.abs(y)) + EPS)
    y = y * (ceiling / peak2)
    reduction = db20(peak / ceiling)
    return sanitize(y), float(reduction)

def normalize_peak(audio, ceiling_db=-1.0):
    ceiling = undb20(ceiling_db)
    peak = np.max(np.abs(audio)) + EPS
    if peak > ceiling:
        return audio * (ceiling / peak)
    return audio


