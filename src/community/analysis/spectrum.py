from pathlib import Path
import json
import numpy as np
from scipy import signal
from community.meters.common import mono

BANDS = {
    "sub_20_60": (20, 60),
    "bass_60_150": (60, 150),
    "low_mid_150_500": (150, 500),
    "mid_500_2500": (500, 2500),
    "presence_2500_8000": (2500, 8000),
    "air_8000_18000": (8000, 18000),
}

def welch_spectrum(audio: np.ndarray, sr: int):
    x = mono(audio)
    nperseg = min(16384, max(1024, len(x)//2))
    freqs, pxx = signal.welch(x, fs=sr, nperseg=nperseg)
    return freqs, pxx + 1e-20

def band_energy(freqs, pxx, low, high):
    mask = (freqs >= low) & (freqs < high)
    return float(np.sum(pxx[mask]) + 1e-20)

def spectral_centroid(freqs, pxx):
    return float(np.sum(freqs * pxx) / np.sum(pxx))

def spectral_rolloff(freqs, pxx, pct=0.85):
    cumsum = np.cumsum(pxx)
    threshold = cumsum[-1] * pct
    idx = int(np.searchsorted(cumsum, threshold))
    return float(freqs[min(idx, len(freqs)-1)])

def analyze_spectrum(audio: np.ndarray, sr: int) -> dict:
    freqs, pxx = welch_spectrum(audio, sr)
    raw = {name: band_energy(freqs, pxx, lo, hi) for name, (lo, hi) in BANDS.items()}
    total = sum(raw.values()) + 1e-20
    bands = {name: raw[name] / total for name in raw}
    low_end = bands["sub_20_60"] + bands["bass_60_150"]
    brightness = bands["presence_2500_8000"] + bands["air_8000_18000"]

    return {
        "task": "Task 006 - Spectrum Analyzer",
        "bands": {k: round(float(v), 6) for k, v in bands.items()},
        "raw_band_energy": {k: float(v) for k, v in raw.items()},
        "band_energy_sum": round(float(sum(bands.values())), 6),
        "mud_index": round(float(bands["low_mid_150_500"]), 6),
        "harshness_index": round(float(bands["presence_2500_8000"]), 6),
        "brightness_index": round(float(brightness), 6),
        "low_end_index": round(float(low_end), 6),
        "air_index": round(float(bands["air_8000_18000"]), 6),
        "spectral_centroid_hz": round(spectral_centroid(freqs, pxx), 3),
        "spectral_rolloff_hz": round(spectral_rolloff(freqs, pxx), 3),
    }

def save_spectrum_analysis(audio: np.ndarray, sr: int, analysis_dir: str | Path) -> dict:
    report = analyze_spectrum(audio, sr)
    path = Path(analysis_dir) / "spectrum.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report



