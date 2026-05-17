from __future__ import annotations

from pathlib import Path

import numpy as np

from community.core.audio_reader import read_working_wav
from community.core.project_loader import load_project
from community.core.project_logger import ProjectLogger
from community.modules.dsp_utils import (
    downsample_audio,
    highpass,
    highshelf,
    lowpass,
    lowshelf,
    normalize_peak,
    oversample_audio,
    peaking_eq,
    sanitize,
    smooth_control_signal,
)
from community.modules.stem_separator import analyze_audio, save_wav, write_json
from community.modules.vintage_limiter import process_vintage_limiter


def process_vintage_tape(audio, sr: int, params: dict | None = None):
    params = params or {}
    analog_mode = str(params.get("analog_mode", params.get("mode", "tape"))).lower()
    drive = float(params.get("drive", 1.15))
    mix = float(params.get("mix", 0.4))
    oversampling = int(params.get("oversampling", 8))
    if oversampling not in {1, 2, 4, 8}:
        oversampling = 8
    bias_amount = float(params.get("bias_amount", 0.08 if analog_mode == "tube" else 0.05))
    attack_ms = float(params.get("bias_attack_ms", 4.0))
    release_ms = float(params.get("bias_release_ms", 120.0))

    os_audio = oversample_audio(audio, oversampling)
    sr_os = sr * oversampling
    env = smooth_control_signal(np.mean(np.abs(os_audio), axis=1), sr_os, attack_ms=attack_ms, release_ms=release_ms)
    env_norm = env / (np.percentile(env, 99) + 1e-12)
    env_norm = np.clip(env_norm, 0.0, 1.25)
    dynamic_drive = drive * (0.85 + 0.55 * env_norm)
    bias = bias_amount * (env_norm - float(np.mean(env_norm)))

    x = os_audio
    if analog_mode == "tube":
        shaped = (x * dynamic_drive[:, None] + bias[:, None]) / (1.0 + np.abs(x * dynamic_drive[:, None] + bias[:, None]))
    elif analog_mode == "transformer":
        low = lowpass(x, sr_os, 240.0)
        high = x - low
        low_sat = np.tanh(low * (dynamic_drive[:, None] * 1.25 + 0.15))
        high_sat = np.tanh((high + bias[:, None] * 0.3) * (dynamic_drive[:, None] * 0.75))
        shaped = low_sat + high_sat
    else:
        shaped = np.tanh((x + bias[:, None]) * dynamic_drive[:, None])

    wet = downsample_audio(shaped, oversampling)
    out = sanitize((audio * (1.0 - mix)) + (wet * mix))
    out = normalize_peak(out, -1.0)
    return out, {
        "task": "Task 034 - Vintage Tape",
        "analog_mode": analog_mode,
        "drive": drive,
        "mix": mix,
        "oversampling_factor": oversampling,
        "effective_sample_rate": sr_os,
        "dynamic_bias_enabled": True,
        "bias_amount": round(bias_amount, 6),
        "dynamic_drive_range": [round(float(np.min(dynamic_drive)), 6), round(float(np.max(dynamic_drive)), 6)],
        "dynamic_bias_range": [round(float(np.min(bias)), 6), round(float(np.max(bias)), 6)],
    }


def process_vintage_eq(audio, sr: int, params: dict | None = None):
    params = params or {}
    low_shelf_db = float(params.get("low_shelf_db", 1.1))
    mid_dip_db = float(params.get("mid_dip_db", -0.65))
    high_shelf_db = float(params.get("high_shelf_db", -0.9))
    out = lowshelf(audio, sr, 140.0, low_shelf_db, slope=0.8)
    out = peaking_eq(out, sr, 2400.0, mid_dip_db, q=0.8)
    out = highshelf(out, sr, 8200.0, high_shelf_db, slope=0.75)
    out = sanitize(normalize_peak(out, -1.0))
    return out, {
        "task": "Task 034 - Vintage EQ",
        "low_shelf_db": low_shelf_db,
        "mid_dip_db": mid_dip_db,
        "high_shelf_db": high_shelf_db,
    }


def _apply_noise(audio: np.ndarray, noise_level: float) -> np.ndarray:
    if noise_level <= 0:
        return audio
    rng = np.random.default_rng(34)
    noise = rng.normal(0.0, noise_level, size=audio.shape)
    return sanitize(audio + noise)


def _apply_wow_flutter(audio: np.ndarray, sr: int, amount: float) -> np.ndarray:
    if amount <= 0:
        return audio
    t = np.arange(len(audio)) / float(sr)
    lfo = np.sin(2 * np.pi * 0.55 * t) + (0.35 * np.sin(2 * np.pi * 3.6 * t))
    max_delay = int(max(1, sr * 0.0025 * amount))
    delayed_index = np.arange(len(audio)) + np.clip((lfo * max_delay).astype(int), -max_delay, max_delay)
    delayed_index = np.clip(delayed_index, 0, len(audio) - 1)
    out = np.zeros_like(audio)
    out[:, 0] = audio[delayed_index, 0]
    out[:, 1] = audio[np.flip(delayed_index), 1]
    return sanitize(out)


def run_vintage_color(project_json: str | Path, params: dict | None = None) -> dict:
    params = params or {}
    project_load = load_project(project_json, save_status=True)
    project = project_load["project"]
    paths = project["paths"]
    logger = ProjectLogger(paths["project_log"])
    logger.info("Task 034 Vintage / Color started.")

    audio, sr, _ = read_working_wav(paths["working_wav"])
    analysis_before = analyze_audio(audio, sr)
    mode = str(params.get("mode", "warm")).lower()
    noise_level = float(params.get("noise_level", 0.0))
    wow_flutter = max(0.0, float(params.get("wow_flutter", 0.0)))

    chain = []
    output = sanitize(audio)
    if mode in {"tape", "tube", "transformer"}:
        output, tape_report = process_vintage_tape(output, sr, {**params, "analog_mode": mode})
        chain.append(tape_report)
    elif mode == "eq":
        output, eq_report = process_vintage_eq(output, sr, params)
        chain.append(eq_report)
    elif mode == "limiter":
        output, limiter_report = process_vintage_limiter(output, sr, params)
        chain.append(limiter_report)
    else:
        output, tape_report = process_vintage_tape(output, sr, params)
        output, eq_report = process_vintage_eq(output, sr, params)
        output, limiter_report = process_vintage_limiter(output, sr, params)
        chain.extend([tape_report, eq_report, limiter_report])

    output = _apply_wow_flutter(output, sr, wow_flutter)
    output = _apply_noise(output, noise_level)
    output = sanitize(normalize_peak(output, -1.0))

    analysis_after = analyze_audio(output, sr)
    root = Path(paths["root"])
    analysis_dir = Path(paths["analysis_dir"])
    output_dir = root / "vintage_color"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_wav = output_dir / f"vintage_{mode}.wav"
    save_wav(output_wav, output, sr)

    report = {
        "task": "Task 034 - Vintage / Color",
        "status": "success",
        "mode": mode,
        "output_wav": str(output_wav),
        "noise_level": noise_level,
        "wow_flutter": wow_flutter,
        "processing_chain": chain,
        "oversampling_factor": max([int(step.get("oversampling_factor", 1)) for step in chain] or [1]),
        "dynamic_bias_enabled": any(bool(step.get("dynamic_bias_enabled", False)) for step in chain),
        "analysis_before": analysis_before,
        "analysis_after": analysis_after,
        "harmonic_delta": float(np.mean(np.abs(output - audio))),
        "brightness_delta": analysis_after["spectrum"]["brightness_index"] - analysis_before["spectrum"]["brightness_index"],
        "low_end_delta": analysis_after["spectrum"]["low_end_index"] - analysis_before["spectrum"]["low_end_index"],
        "spectral_centroid_delta_hz": analysis_after["spectrum"]["spectral_centroid_hz"] - analysis_before["spectrum"]["spectral_centroid_hz"],
    }
    write_json(analysis_dir / "vintage_color_report.json", report)
    write_json(
        analysis_dir / "vintage_color_curve.json",
        {
            "mode": mode,
            "brightness_delta": report["brightness_delta"],
            "low_end_delta": report["low_end_delta"],
            "spectral_centroid_delta_hz": report["spectral_centroid_delta_hz"],
            "harmonic_delta": report["harmonic_delta"],
        },
    )
    (analysis_dir / "vintage_color_report.md").write_text(
        "\n".join(
            [
                "# Vintage / Color Report",
                "",
                f"- Mode: `{mode}`",
                f"- Output: `{output_wav}`",
                f"- Harmonic delta: `{report['harmonic_delta']}`",
                f"- Brightness delta: `{report['brightness_delta']}`",
                f"- Spectral centroid delta (Hz): `{report['spectral_centroid_delta_hz']}`",
            ]
        ),
        encoding="utf-8",
    )
    logger.info(f"Task 034 Vintage / Color completed. mode={mode}")
    return {
        "report": report,
        "analysis_after": analysis_after,
    }



