from pathlib import Path
import json
import numpy as np
from scipy import signal
from community.meters.common import db20, mono, frame_audio

def short_rms_db(x, sr, window_sec=0.05):
    frame = max(256, int(sr * window_sec))
    vals = []
    for fr in frame_audio(x, frame, frame):
        vals.append(float(db20(np.sqrt(np.mean(fr ** 2)) + 1e-12)))
    return np.array(vals, dtype=float) if vals else np.array([float(db20(np.sqrt(np.mean(x ** 2)) + 1e-12))])

def transient_density(x, sr):
    # high-pass-ish onset proxy using absolute derivative peaks
    diff = np.abs(np.diff(x, prepend=x[0]))
    if len(diff) < 10:
        return 0.0
    threshold = np.percentile(diff, 95)
    peaks, _ = signal.find_peaks(diff, height=threshold, distance=max(1, int(sr * 0.03)))
    seconds = len(x) / sr
    # 10 transients/sec around dense, normalize to 0~1
    return float(np.clip((len(peaks) / max(seconds, 1e-9)) / 10.0, 0, 1))

def analyze_dynamics(audio: np.ndarray, sr: int) -> dict:
    x = mono(audio)
    rms_dist = short_rms_db(x, sr, 0.05)
    dynamic_range_approx = float(np.percentile(rms_dist, 95) - np.percentile(rms_dist, 10))
    peak = float(np.max(np.abs(audio)) + 1e-12)
    rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
    crest_factor_db = float(db20(peak / rms))
    td = transient_density(x, sr)
    punch_score = float(np.clip((crest_factor_db / 16.0) * 0.6 + td * 0.4, 0, 1))
    overcompression_score = float(np.clip((1.0 - dynamic_range_approx / 10.0) * 0.55 + (1.0 - crest_factor_db / 14.0) * 0.45, 0, 1))
    limiter_damage_risk = "high" if overcompression_score > 0.72 else ("medium" if overcompression_score > 0.45 else "low")
    needs_unlimiter = bool(overcompression_score > 0.58 and crest_factor_db < 9.0)

    return {
        "task": "Task 008 - Dynamics / Transient Analyzer",
        "dynamic_range_approx": round(dynamic_range_approx, 6),
        "crest_factor_db": round(crest_factor_db, 6),
        "transient_density": round(td, 6),
        "punch_score": round(punch_score, 6),
        "overcompression_score": round(overcompression_score, 6),
        "limiter_damage_risk": limiter_damage_risk,
        "needs_unlimiter": needs_unlimiter,
        "short_window_rms_db": {
            "p10": round(float(np.percentile(rms_dist, 10)), 6),
            "p50": round(float(np.percentile(rms_dist, 50)), 6),
            "p95": round(float(np.percentile(rms_dist, 95)), 6),
            "max": round(float(np.max(rms_dist)), 6),
            "min": round(float(np.min(rms_dist)), 6),
        }
    }

def save_dynamics_analysis(audio: np.ndarray, sr: int, analysis_dir: str | Path) -> dict:
    report = analyze_dynamics(audio, sr)
    path = Path(analysis_dir) / "dynamics.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report



