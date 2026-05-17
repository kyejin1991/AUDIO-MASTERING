from pathlib import Path
import json
import numpy as np
import soundfile as sf

def sanitize_audio(audio: np.ndarray) -> np.ndarray:
    return np.nan_to_num(audio.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    if audio.shape[1] > 2:
        return audio[:, :2]
    return audio

def read_working_wav(working_wav: str | Path) -> tuple[np.ndarray, int, dict]:
    """
    Task 003: working WAV瑜??덉젙?곸쑝濡?float64 stereo ndarray濡?濡쒕뱶.
    """
    path = Path(working_wav).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Working WAV not found: {path}")

    audio, sr = sf.read(path, always_2d=True)
    audio = ensure_stereo(sanitize_audio(audio))

    peak_abs = float(np.max(np.abs(audio))) if audio.size else 0.0
    clipped_for_safety = False
    if peak_abs > 4.0:
        audio = np.clip(audio, -4.0, 4.0)
        peak_abs = float(np.max(np.abs(audio)))
        clipped_for_safety = True

    info = {
        "task": "Task 003 - Audio Reader",
        "path": str(path),
        "sample_rate": int(sr),
        "channels": int(audio.shape[1]),
        "samples": int(audio.shape[0]),
        "duration_sec": round(float(audio.shape[0] / sr), 6) if sr else 0.0,
        "dtype": str(audio.dtype),
        "peak_abs": peak_abs,
        "has_nan": bool(np.isnan(audio).any()),
        "has_inf": bool(np.isinf(audio).any()),
        "safety_clip_applied": clipped_for_safety,
        "stereo_guaranteed": bool(audio.ndim == 2 and audio.shape[1] == 2),
    }
    return audio, int(sr), info

def save_audio_reader_info(audio_info: dict, analysis_dir: str | Path) -> dict:
    path = Path(analysis_dir) / "audio_reader_info.json"
    path.write_text(json.dumps(audio_info, indent=2, ensure_ascii=False), encoding="utf-8")
    return audio_info

