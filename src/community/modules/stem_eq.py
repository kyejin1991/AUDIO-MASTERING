from __future__ import annotations

from pathlib import Path

import numpy as np

from community.core.project_loader import load_project
from community.core.project_logger import ProjectLogger
from community.modules.audio_modules import process_deesser
from community.modules.dsp_utils import (
    apply_gain_db,
    from_mid_side,
    highpass,
    highshelf,
    lowpass,
    mid_side,
    normalize_peak,
    peaking_eq,
    sanitize,
    undb20,
)
from community.modules.stem_separator import analyze_audio, load_stem_wav, save_wav, write_json


def _normalized_amount(value: float, max_unit: float = 100.0) -> float:
    value = float(value)
    if abs(value) <= 1.0:
        return float(np.clip(value, -1.0, 1.0))
    return float(np.clip(value / max_unit, -1.0, 1.0))


def _transient_enhance(audio: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 1e-6:
        return audio
    mono = np.mean(audio, axis=1)
    diff = np.abs(np.diff(mono, prepend=mono[0]))
    peak = float(np.max(diff) + 1e-12)
    env = np.clip(diff / peak, 0.0, 1.0)
    gain = 1.0 + (0.45 * amount * env[:, None])
    return sanitize(audio * gain)


def _tighten_bass(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    if abs(amount) < 1e-6:
        return audio
    y = highpass(audio, sr, 28 + max(0.0, amount) * 12.0, order=2)
    y = peaking_eq(y, sr, 180.0, gain_db=-3.0 * max(0.0, amount), q=0.9)
    y = peaking_eq(y, sr, 85.0, gain_db=1.2 * max(0.0, amount), q=1.0)
    return sanitize(y)


def _adjust_width(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    if abs(amount) < 1e-6:
        return audio
    mid, side = mid_side(audio)
    low_mid = lowpass(np.repeat(mid[:, None], 2, axis=1), sr, 140)[:, 0]
    low_side = lowpass(np.repeat(side[:, None], 2, axis=1), sr, 140)[:, 0] * 0.2
    high_mid = mid - low_mid
    high_side = side - low_side
    high_side = high_side * (1.0 + (0.55 * amount))
    return sanitize(from_mid_side(low_mid + high_mid, low_side + high_side))


def apply_stem_eq(
    stems: dict[str, np.ndarray],
    sr: int,
    params: dict[str, float] | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    params = params or {}
    processed = {name: sanitize(audio.copy()) for name, audio in stems.items()}

    vocals_gain_db = float(params.get("vocals_gain_db", params.get("vocal_gain_db", 0.0)))
    vocals_brightness_db = float(params.get("vocals_brightness_db", params.get("vocal_brightness_db", 0.0)))
    vocal_deess_amount = _normalized_amount(float(params.get("vocal_deess", 0.0)))
    bass_gain_db = float(params.get("bass_gain_db", 0.0))
    bass_tightness = _normalized_amount(float(params.get("bass_tightness", 0.0)))
    drums_punch = _normalized_amount(float(params.get("drum_punch", 0.0)))
    drums_brightness_db = float(params.get("drum_brightness_db", 0.0))
    instrument_width = _normalized_amount(float(params.get("instrument_width", 0.0)))

    report = {
        "task": "Task 032 - Stem EQ",
        "applied": {
            "vocals_gain_db": vocals_gain_db,
            "vocals_brightness_db": vocals_brightness_db,
            "vocal_deess": vocal_deess_amount,
            "bass_gain_db": bass_gain_db,
            "bass_tightness": bass_tightness,
            "drum_punch": drums_punch,
            "drum_brightness_db": drums_brightness_db,
            "instrument_width": instrument_width,
        },
        "per_stem": {},
    }

    if "vocals" in processed:
        vocals = processed["vocals"]
        vocals = apply_gain_db(vocals, vocals_gain_db)
        vocals = highshelf(vocals, sr, 5500.0, vocals_brightness_db, slope=0.8)
        deess_report = None
        if vocal_deess_amount > 0:
            vocals, deess_report = process_deesser(
                vocals,
                sr,
                {
                    "freq_range_hz": [5000, 9000],
                    "threshold_db": -34.0 + (10.0 * (1.0 - vocal_deess_amount)),
                    "max_reduction_db": 1.0 + (3.0 * vocal_deess_amount),
                },
            )
        processed["vocals"] = sanitize(vocals)
        report["per_stem"]["vocals"] = {
            "gain_db": vocals_gain_db,
            "brightness_db": vocals_brightness_db,
            "deess_report": deess_report,
        }

    if "bass" in processed:
        bass = processed["bass"]
        bass = apply_gain_db(bass, bass_gain_db)
        bass = _tighten_bass(bass, sr, bass_tightness)
        processed["bass"] = sanitize(bass)
        report["per_stem"]["bass"] = {
            "gain_db": bass_gain_db,
            "tightness": bass_tightness,
        }

    if "drums" in processed:
        drums = processed["drums"]
        drums = _transient_enhance(drums, drums_punch)
        drums = highshelf(drums, sr, 7000.0, drums_brightness_db, slope=0.9)
        processed["drums"] = sanitize(drums)
        report["per_stem"]["drums"] = {
            "punch": drums_punch,
            "brightness_db": drums_brightness_db,
        }

    for name in ["music", "ambience", "other"]:
        if name in processed:
            processed[name] = _adjust_width(processed[name], sr, instrument_width)
            report["per_stem"][name] = {
                "width": instrument_width,
            }

    return processed, report


def run_stem_eq(project_json: str | Path, params: dict | None = None) -> dict:
    params = params or {}
    project_load = load_project(project_json, save_status=True)
    project = project_load["project"]
    paths = project["paths"]
    logger = ProjectLogger(paths["project_log"])
    logger.info("Task 032 Stem EQ started.")

    root = Path(paths["root"])
    analysis_dir = Path(paths["analysis_dir"])
    stems_dir = root / "stems"
    output_dir = root / "stem_eq"
    processed_stems_dir = output_dir / "stems_processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_stems_dir.mkdir(parents=True, exist_ok=True)

    if not stems_dir.exists():
        raise FileNotFoundError(f"Stem directory not found: {stems_dir}")

    stem_names = ["vocals", "drums", "bass", "music", "ambience", "other"]
    stems: dict[str, np.ndarray] = {}
    sr = None
    target_len = None
    for name in stem_names:
        path = stems_dir / f"{name}.wav"
        if not path.exists():
            continue
        audio, stem_sr = load_stem_wav(path)
        stems[name] = audio
        sr = stem_sr if sr is None else sr
        target_len = len(audio) if target_len is None else max(target_len, len(audio))

    if not stems or sr is None or target_len is None:
        raise FileNotFoundError(f"No usable stems found in {stems_dir}")

    aligned_stems = {}
    for name, audio in stems.items():
        if len(audio) < target_len:
            audio = np.pad(audio, ((0, target_len - len(audio)), (0, 0)))
        elif len(audio) > target_len:
            audio = audio[:target_len]
        aligned_stems[name] = sanitize(audio)

    before_mix = np.zeros((target_len, 2), dtype=np.float64)
    for stem in aligned_stems.values():
        before_mix += stem

    processed_stems, stem_report = apply_stem_eq(aligned_stems, sr, params=params)

    output_mix = np.zeros_like(before_mix)
    processed_files = {}
    for name, stem in processed_stems.items():
        out_path = processed_stems_dir / f"{name}.wav"
        save_wav(out_path, stem, sr)
        processed_files[name] = str(out_path)
        output_mix += stem

    peak_before = float(np.max(np.abs(output_mix)) + 1e-12)
    clipping_protection_applied = peak_before > undb20(-1.0)
    output_mix = normalize_peak(output_mix, -1.0)
    output_mix = sanitize(output_mix)

    output_wav = output_dir / "stem_eq_master.wav"
    save_wav(output_wav, output_mix, sr)

    analysis_before = analyze_audio(before_mix, sr)
    analysis_after = analyze_audio(output_mix, sr)
    report = {
        "task": "Task 032 - Stem EQ",
        "status": "success",
        "project_id": project["project_id"],
        "output_wav": str(output_wav),
        "processed_stem_files": processed_files,
        "processing": stem_report,
        "clipping_protection_applied": clipping_protection_applied,
        "clip_guard_peak_before": peak_before,
        "clipping_samples_after": analysis_after["basic_audio"]["clipping_samples"],
        "analysis_before": analysis_before,
        "analysis_after": analysis_after,
    }
    write_json(analysis_dir / "stem_eq_report.json", report)
    write_json(analysis_dir / "stem_eq_master_analysis.json", analysis_after)
    (analysis_dir / "stem_eq_report.md").write_text(
        "\n".join(
            [
                "# Stem EQ Report",
                "",
                f"- Output: `{output_wav}`",
                f"- Clipping protection applied: `{clipping_protection_applied}`",
                f"- Vocals gain: `{stem_report['applied']['vocals_gain_db']}` dB",
                f"- Bass gain: `{stem_report['applied']['bass_gain_db']}` dB",
                f"- Drum punch: `{stem_report['applied']['drum_punch']}`",
                f"- Instrument width: `{stem_report['applied']['instrument_width']}`",
            ]
        ),
        encoding="utf-8",
    )
    logger.info("Task 032 Stem EQ completed.")
    return {
        "report": report,
        "analysis_after": analysis_after,
    }



