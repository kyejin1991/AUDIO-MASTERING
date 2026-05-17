from pathlib import Path
import json
import numpy as np
from community.meters.common import db20
from community.meters.loudness import integrated_lufs, windowed_lufs, true_peak_dbtp

def analyze_loudness(audio: np.ndarray, sr: int, youtube_target_lufs: float = -14.0, ceiling_dbtp: float = -1.0) -> dict:
    integ = integrated_lufs(audio, sr)
    short_vals = windowed_lufs(audio, sr, 3.0)
    moment_vals = windowed_lufs(audio, sr, 0.4)
    lra = float(np.percentile(short_vals, 95) - np.percentile(short_vals, 10)) if len(short_vals) > 1 else 0.0
    tp = true_peak_dbtp(audio, sr)
    sp = float(db20(np.max(np.abs(audio)) + 1e-12))
    diff = float(youtube_target_lufs - integ)

    return {
        "task": "Task 005 - Loudness Meter",
        "integrated_lufs": round(integ, 4),
        "short_term_lufs_max": round(float(np.max(short_vals)), 4),
        "short_term_lufs_min": round(float(np.min(short_vals)), 4),
        "short_term_lufs_mean": round(float(np.mean(short_vals)), 4),
        "momentary_lufs_max": round(float(np.max(moment_vals)), 4),
        "momentary_lufs_min": round(float(np.min(moment_vals)), 4),
        "loudness_range": round(lra, 4),
        "true_peak_dbtp": round(tp, 4),
        "sample_peak_dbfs": round(sp, 4),
        "youtube_target_lufs": youtube_target_lufs,
        "youtube_lufs_diff": round(integ - youtube_target_lufs, 4),
        "needs_loudness_gain": bool(integ < youtube_target_lufs - 1.0),
        "recommended_gain_db": round(diff, 4),
        "true_peak_risk": bool(tp > ceiling_dbtp),
        "ceiling_dbtp": ceiling_dbtp,
    }

def save_loudness_analysis(audio: np.ndarray, sr: int, analysis_dir: str | Path) -> dict:
    report = analyze_loudness(audio, sr)
    path = Path(analysis_dir) / "loudness.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
