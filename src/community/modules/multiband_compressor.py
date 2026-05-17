from __future__ import annotations
import numpy as np

from .dsp_utils import sanitize, bandpass, db20, undb20, normalize_peak
from community.analysis.dynamics import analyze_dynamics
from community.analysis.spectrum import analyze_spectrum

DEFAULT_BANDS = [
    {"id": 1, "name": "sub_bass", "low_hz": 20, "high_hz": 120, "threshold_db": -24, "ratio": 1.8, "attack_ms": 25, "release_ms": 180, "knee_db": 4, "makeup_db": 0.0, "mix": 0.75, "enabled": True},
    {"id": 2, "name": "low_mid", "low_hz": 120, "high_hz": 500, "threshold_db": -25, "ratio": 1.7, "attack_ms": 18, "release_ms": 160, "knee_db": 4, "makeup_db": 0.0, "mix": 0.70, "enabled": True},
    {"id": 3, "name": "mid", "low_hz": 500, "high_hz": 2500, "threshold_db": -24, "ratio": 1.5, "attack_ms": 12, "release_ms": 130, "knee_db": 3, "makeup_db": 0.0, "mix": 0.65, "enabled": True},
    {"id": 4, "name": "presence", "low_hz": 2500, "high_hz": 8000, "threshold_db": -27, "ratio": 1.8, "attack_ms": 6, "release_ms": 100, "knee_db": 3, "makeup_db": 0.0, "mix": 0.65, "enabled": True},
    {"id": 5, "name": "air", "low_hz": 8000, "high_hz": 18000, "threshold_db": -30, "ratio": 1.4, "attack_ms": 4, "release_ms": 90, "knee_db": 3, "makeup_db": 0.0, "mix": 0.55, "enabled": True},
]

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def smooth_envelope(raw: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    attack = np.exp(-1.0 / max(1.0, (attack_ms/1000.0)*sr))
    release = np.exp(-1.0 / max(1.0, (release_ms/1000.0)*sr))
    out = np.zeros_like(raw)
    prev = 0.0
    for i, x in enumerate(raw):
        coeff = attack if x > prev else release
        prev = coeff * prev + (1.0 - coeff) * x
        out[i] = prev
    return out

def normalize_band(raw: dict, idx: int) -> dict:
    base = dict(DEFAULT_BANDS[min(idx, len(DEFAULT_BANDS)-1)])
    base.update(raw or {})
    low = clamp(base.get("low_hz", 20), 10, 19000)
    high = clamp(base.get("high_hz", 120), low + 20, 20000)
    return {
        "id": int(base.get("id", idx + 1)),
        "name": str(base.get("name", f"band_{idx+1}")),
        "low_hz": round(low, 4),
        "high_hz": round(high, 4),
        "threshold_db": round(clamp(base.get("threshold_db", -24), -80, 0), 4),
        "ratio": round(clamp(base.get("ratio", 1.5), 1.0, 20.0), 4),
        "attack_ms": round(clamp(base.get("attack_ms", 10), 0.1, 500), 4),
        "release_ms": round(clamp(base.get("release_ms", 120), 1, 3000), 4),
        "knee_db": round(clamp(base.get("knee_db", 3), 0, 24), 4),
        "makeup_db": round(clamp(base.get("makeup_db", 0), -12, 12), 4),
        "mix": round(clamp(base.get("mix", 0.65), 0, 1), 4),
        "enabled": bool(base.get("enabled", True)),
    }

def build_bands(params: dict, full_analysis: dict | None):
    amount = float(params.get("amount", 0.35))
    amount = clamp(amount, 0.0, 1.0)
    raw_bands = params.get("band_settings") or params.get("bands")
    if isinstance(raw_bands, list) and raw_bands and isinstance(raw_bands[0], dict):
        bands = [normalize_band(b, i) for i, b in enumerate(raw_bands[:5])]
        while len(bands) < 5:
            bands.append(normalize_band({}, len(bands)))
    else:
        bands = [normalize_band({}, i) for i in range(5)]

    spectrum = full_analysis.get("spectrum", {}) if full_analysis else {}
    dynamics = full_analysis.get("dynamics", {}) if full_analysis else {}
    over = float(dynamics.get("overcompression_score", 0.0))
    mud = float(spectrum.get("mud_index", 0.2))
    harsh = float(spectrum.get("harshness_index", 0.1))
    low = float(spectrum.get("low_end_index", 0.25))

    # Amount scales compression gently. Avoid crushing already overcompressed material.
    safety_scale = 0.55 if over > 0.65 else 1.0
    for b in bands:
        b["ratio"] = round(1.0 + (b["ratio"] - 1.0) * (0.65 + amount) * safety_scale, 4)
        b["mix"] = round(clamp(b["mix"] * (0.55 + amount), 0.15, 0.95), 4)
    if low > 0.55:
        bands[0]["threshold_db"] -= 2.0
        bands[0]["ratio"] = round(min(4.0, bands[0]["ratio"] + 0.5), 4)
    if mud > 0.28:
        bands[1]["threshold_db"] -= 2.0
        bands[1]["ratio"] = round(min(4.0, bands[1]["ratio"] + 0.45), 4)
    if harsh > 0.18:
        bands[3]["threshold_db"] -= 2.5
        bands[3]["ratio"] = round(min(5.0, bands[3]["ratio"] + 0.7), 4)
    return bands, {"amount": amount, "overcompression_safety_scale": safety_scale, "source": "assistant/manual + analysis refinement"}

def rms_detector_db(band_audio: np.ndarray, sr: int, window_ms: float = 10.0):
    mono = np.mean(band_audio, axis=1)
    # Lightweight running RMS via convolution
    win = max(8, int(sr * window_ms / 1000.0))
    kernel = np.ones(win) / win
    power = np.convolve(mono * mono, kernel, mode="same")
    return db20(np.sqrt(np.maximum(power, 1e-12)))

def soft_knee_gain_reduction(level_db, threshold_db, ratio, knee_db):
    x = level_db - threshold_db
    if knee_db <= 0:
        over = np.maximum(x, 0.0)
    else:
        # soft knee from -knee/2 to +knee/2
        over = np.zeros_like(x)
        lower = -knee_db / 2.0
        upper = knee_db / 2.0
        below = x <= lower
        above = x >= upper
        middle = (~below) & (~above)
        over[above] = x[above]
        over[middle] = ((x[middle] - lower) ** 2) / (2.0 * knee_db)
    gr = over * (1.0 - 1.0 / max(1.0, ratio))
    return gr

def process_band(audio, sr, band):
    split = bandpass(audio, sr, band["low_hz"], band["high_hz"])
    if not band["enabled"]:
        return split, {
            "band": band,
            "enabled": False,
            "max_gain_reduction_db": 0.0,
            "mean_gain_reduction_db": 0.0,
            "before_rms_db": round(float(db20(np.sqrt(np.mean(split**2)) + 1e-12)), 6),
            "after_rms_db": round(float(db20(np.sqrt(np.mean(split**2)) + 1e-12)), 6),
        }

    level = rms_detector_db(split, sr)
    raw_gr = soft_knee_gain_reduction(level, band["threshold_db"], band["ratio"], band["knee_db"])
    env_gr = smooth_envelope(raw_gr, sr, band["attack_ms"], band["release_ms"])
    gain = undb20(-env_gr)[:, None]
    compressed = split * gain
    if abs(band["makeup_db"]) > 0.0001:
        compressed = compressed * undb20(band["makeup_db"])
    out = split * (1.0 - band["mix"]) + compressed * band["mix"]

    before_rms = float(db20(np.sqrt(np.mean(split**2)) + 1e-12))
    after_rms = float(db20(np.sqrt(np.mean(out**2)) + 1e-12))
    active_ratio = float(np.mean(env_gr > 0.05)) if len(env_gr) else 0.0
    return out, {
        "band": band,
        "enabled": True,
        "max_gain_reduction_db": round(float(np.max(env_gr)) if len(env_gr) else 0.0, 6),
        "mean_gain_reduction_db": round(float(np.mean(env_gr)) if len(env_gr) else 0.0, 6),
        "active_ratio": round(active_ratio, 6),
        "before_rms_db": round(before_rms, 6),
        "after_rms_db": round(after_rms, 6),
        "rms_delta_db": round(after_rms - before_rms, 6),
    }

def process_multiband_compressor_advanced(audio, sr, params: dict, full_analysis: dict | None = None):
    before_dynamics = analyze_dynamics(audio, sr)
    before_spectrum = analyze_spectrum(audio, sr)
    bands, build_report = build_bands(params, full_analysis)

    band_outputs = []
    band_reports = []
    for band in bands:
        out, rep = process_band(audio, sr, band)
        band_outputs.append(out)
        band_reports.append(rep)

    # Reconstruct from processed bands + residual outside band range
    summed_original_bands = np.zeros_like(audio)
    for band in bands:
        summed_original_bands += bandpass(audio, sr, band["low_hz"], band["high_hz"])
    residual = audio - summed_original_bands
    y = residual + sum(band_outputs)
    y = normalize_peak(sanitize(y), -1.0)

    after_dynamics = analyze_dynamics(y, sr)
    after_spectrum = analyze_spectrum(y, sr)

    report = {
        "task": "Task 025 - Multiband Compressor",
        "status": "success",
        "bands": bands,
        "build_report": build_report,
        "band_reports": band_reports,
        "before_dynamics": before_dynamics,
        "after_dynamics": after_dynamics,
        "before_spectrum": before_spectrum,
        "after_spectrum": after_spectrum,
        "max_gain_reduction_db": round(max([b["max_gain_reduction_db"] for b in band_reports] or [0.0]), 6),
        "mean_gain_reduction_db": round(float(np.mean([b["mean_gain_reduction_db"] for b in band_reports])) if band_reports else 0.0, 6),
        "dynamic_range_before": before_dynamics.get("dynamic_range_approx"),
        "dynamic_range_after": after_dynamics.get("dynamic_range_approx"),
        "crest_factor_before": before_dynamics.get("crest_factor_db"),
        "crest_factor_after": after_dynamics.get("crest_factor_db"),
        "band_solo_audio": band_outputs,
    }
    return sanitize(y), report



