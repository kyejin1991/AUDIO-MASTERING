from pathlib import Path
import json
import numpy as np
from scipy import signal
from community.meters.common import db20

def safe_corr(x, y):
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 1.0
    return float(np.corrcoef(x, y)[0, 1])

def bandpass(audio, sr, low, high):
    sos = signal.butter(2, [low, high], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, audio, axis=0)

def ms_metrics(audio):
    l = audio[:, 0]
    r = audio[:, 1]
    mid = (l + r) * 0.5
    side = (l - r) * 0.5
    mid_energy = float(np.mean(mid ** 2))
    side_energy = float(np.mean(side ** 2))
    total = mid_energy + side_energy + 1e-20
    return mid, side, mid_energy, side_energy, total

def analyze_stereo(audio: np.ndarray, sr: int) -> dict:
    if audio.ndim == 1 or audio.shape[1] == 1:
        audio = np.repeat(audio.reshape(-1, 1), 2, axis=1)

    l, r = audio[:, 0], audio[:, 1]
    mid, side, mid_energy, side_energy, total = ms_metrics(audio)
    stereo_width = float(np.sqrt(side_energy) / (np.sqrt(mid_energy) + 1e-12))
    phase_correlation = safe_corr(l, r)

    stereo_rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    mono_audio = (l + r) * 0.5
    mono_rms = float(np.sqrt(np.mean(mono_audio ** 2)) + 1e-12)
    mono_collapse_loss_db = float(db20(mono_rms / stereo_rms))

    low = bandpass(audio, sr, 20, 120)
    _, _, low_mid_e, low_side_e, low_total = ms_metrics(low)
    low_end_stereo_leakage = float(low_side_e / (low_total + 1e-20))

    band_width = {}
    for name, lo, hi in [
        ("low_20_120", 20, 120),
        ("low_mid_120_500", 120, 500),
        ("mid_500_4000", 500, 4000),
        ("high_4000_18000", 4000, 18000),
    ]:
        b = bandpass(audio, sr, lo, hi)
        _, _, me, se, _ = ms_metrics(b)
        band_width[name] = round(float(np.sqrt(se) / (np.sqrt(me) + 1e-12)), 6)

    warnings = []
    if phase_correlation < 0:
        warnings.append("phase_correlation_below_zero")
    if mono_collapse_loss_db < -6:
        warnings.append("mono_collapse_loss_over_6db")
    if low_end_stereo_leakage > 0.15:
        warnings.append("wide_low_end_detected")

    return {
        "task": "Task 007 - Stereo / Phase Analyzer",
        "stereo_width": round(stereo_width, 6),
        "mid_energy": round(float(mid_energy / total), 6),
        "side_energy": round(float(side_energy / total), 6),
        "mid_side_ratio": round(float(mid_energy / (side_energy + 1e-20)), 6),
        "phase_correlation": round(float(phase_correlation), 6),
        "mono_collapse_loss_db": round(mono_collapse_loss_db, 6),
        "low_end_stereo_leakage": round(low_end_stereo_leakage, 6),
        "band_stereo_width": band_width,
        "warnings": warnings,
    }

def save_stereo_analysis(audio: np.ndarray, sr: int, analysis_dir: str | Path) -> dict:
    report = analyze_stereo(audio, sr)
    path = Path(analysis_dir) / "stereo.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report



