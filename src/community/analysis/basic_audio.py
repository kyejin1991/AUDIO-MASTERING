from pathlib import Path
import json
import numpy as np
from community.meters.common import db20, mono, frame_audio

def analyze_basic_audio(audio: np.ndarray, sr: int) -> dict:
    peak_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    peak_dbfs = float(db20(peak_abs))
    rms_dbfs = float(db20(rms))
    crest_factor_db = float(peak_dbfs - rms_dbfs)
    dc_offsets = np.mean(audio, axis=0) if audio.ndim == 2 else np.array([np.mean(audio)])
    clipping_samples = int(np.sum(np.abs(audio) >= 0.999))
    headroom_db = float(-peak_dbfs) if peak_dbfs <= 0 else float(-peak_dbfs)

    x = mono(audio)
    frame_size = max(1024, int(sr * 0.05))
    silent = 0
    total = 0
    frame_rms_values = []
    for frame in frame_audio(x, frame_size, frame_size):
        frame_rms = np.sqrt(np.mean(frame ** 2))
        frame_rms_values.append(frame_rms)
        if db20(frame_rms) < -60.0:
            silent += 1
        total += 1
    silence_ratio = float(silent / total) if total else 0.0
    if frame_rms_values:
        noise_floor_dbfs = float(db20(np.percentile(frame_rms_values, 15)))
    else:
        noise_floor_dbfs = float(db20(rms + 1e-12))

    return {
        "task": "Task 004 - Basic Audio Analyzer",
        "duration_sec": round(float(audio.shape[0] / sr), 6),
        "sample_rate": int(sr),
        "channels": int(audio.shape[1] if audio.ndim == 2 else 1),
        "samples": int(audio.shape[0]),
        "peak_abs": round(peak_abs, 8),
        "peak_dbfs": round(peak_dbfs, 4),
        "rms_dbfs": round(rms_dbfs, 4),
        "crest_factor_db": round(crest_factor_db, 4),
        "dc_offset_l": round(float(dc_offsets[0]), 10),
        "dc_offset_r": round(float(dc_offsets[1] if len(dc_offsets) > 1 else dc_offsets[0]), 10),
        "clipping_samples": clipping_samples,
        "silence_ratio": round(silence_ratio, 6),
        "headroom_db": round(headroom_db, 4),
        "noise_floor_dbfs": round(noise_floor_dbfs, 4),
    }

def save_basic_audio_analysis(audio: np.ndarray, sr: int, analysis_dir: str | Path) -> dict:
    report = analyze_basic_audio(audio, sr)
    path = Path(analysis_dir) / "basic_audio.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report



