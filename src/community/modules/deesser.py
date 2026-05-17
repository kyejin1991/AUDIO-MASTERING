from __future__ import annotations

from pathlib import Path

import numpy as np

from community.core.audio_reader import read_working_wav
from community.core.project_loader import load_project
from community.core.project_logger import ProjectLogger
from community.modules.audio_modules import process_deesser
from community.modules.dsp_utils import bandpass, db20, normalize_peak, sanitize
from community.modules.stem_separator import analyze_audio, load_stem_wav, save_wav, write_json


def _band_level_db(audio: np.ndarray, sr: int, low_hz: float = 5000.0, high_hz: float = 9000.0) -> float:
    band = bandpass(audio, sr, low_hz, high_hz)
    rms = float(np.sqrt(np.mean(band ** 2)) + 1e-12)
    return float(db20(rms))


def run_deesser(project_json: str | Path, params: dict | None = None) -> dict:
    params = params or {}
    project_load = load_project(project_json, save_status=True)
    project = project_load["project"]
    paths = project["paths"]
    logger = ProjectLogger(paths["project_log"])
    logger.info("Task 033 De-esser started.")

    analysis_dir = Path(paths["analysis_dir"])
    root = Path(paths["root"])
    output_dir = root / "deesser"
    output_dir.mkdir(parents=True, exist_ok=True)

    force_full_mix = bool(params.get("force_full_mix", False))
    stems_dir = root / "stems"
    vocal_stem_path = stems_dir / "vocals.wav"

    if vocal_stem_path.exists() and not force_full_mix:
        vocals, sr = load_stem_wav(vocal_stem_path)
        before = _band_level_db(vocals, sr)
        processed_vocals, deess_report = process_deesser(vocals, sr, params)
        after = _band_level_db(processed_vocals, sr)

        recombined = np.zeros_like(processed_vocals)
        processed_stem_files = {"vocals": None}
        for stem_path in stems_dir.glob("*.wav"):
            stem_audio, stem_sr = load_stem_wav(stem_path, target_len=len(processed_vocals))
            if stem_sr != sr:
                continue
            if stem_path.name == "vocals.wav":
                stem_audio = processed_vocals
            recombined += stem_audio
            processed_stem_files[stem_path.stem] = str(stem_path)

        output = sanitize(normalize_peak(recombined, -1.0))
        output_wav = output_dir / "deesser_from_stems.wav"
        save_wav(output_wav, output, sr)
        mode = "vocal_stem"
    else:
        audio, sr, _ = read_working_wav(paths["working_wav"])
        before = _band_level_db(audio, sr)
        output, deess_report = process_deesser(audio, sr, params)
        output = sanitize(normalize_peak(output, -1.0))
        after = _band_level_db(output, sr)
        output_wav = output_dir / "deesser_full_mix.wav"
        save_wav(output_wav, output, sr)
        processed_stem_files = {}
        mode = "full_mix_fallback"

    analysis_after = analyze_audio(output, sr)
    report = {
        "task": "Task 033 - De-esser",
        "status": "success",
        "processing_mode": mode,
        "output_wav": str(output_wav),
        "used_vocal_stem": mode == "vocal_stem",
        "gain_reduction_log": {
            "max_gain_reduction_db": deess_report["max_gain_reduction_db"],
            "freq_range_hz": deess_report["freq_range_hz"],
        },
        "harshness_before_db": before,
        "harshness_after_db": after,
        "harshness_delta_db": after - before,
        "processed_stem_files": processed_stem_files,
        "analysis_after": analysis_after,
    }
    write_json(analysis_dir / "deesser_report.json", report)
    (analysis_dir / "deesser_report.md").write_text(
        "\n".join(
            [
                "# De-esser Report",
                "",
                f"- Mode: `{mode}`",
                f"- Output: `{output_wav}`",
                f"- Max gain reduction: `{deess_report['max_gain_reduction_db']}` dB",
                f"- Harshness delta: `{report['harshness_delta_db']}` dB",
            ]
        ),
        encoding="utf-8",
    )
    logger.info(f"Task 033 De-esser completed. mode={mode}")
    return {
        "report": report,
        "analysis_after": analysis_after,
    }


__all__ = ["process_deesser", "run_deesser"]



