from __future__ import annotations
import numpy as np

from .dsp_utils import sanitize, bandpass, db20, normalize_peak, undb20
from community.analysis.spectrum import analyze_spectrum
from community.analysis.basic_audio import analyze_basic_audio

MODE_PROFILES = {
    "tape": {
        "drive": 1.6,
        "odd_even_balance": 0.35,
        "soft_clip": 0.20,
        "tilt": {"low": 0.70, "mid": 0.95, "high": 0.65},
        "description": "rounded tape-like saturation with restrained top-end"
    },
    "tube": {
        "drive": 1.9,
        "odd_even_balance": 0.18,
        "soft_clip": 0.18,
        "tilt": {"low": 0.85, "mid": 1.10, "high": 0.90},
        "description": "even-harmonic forward tube-style warmth"
    },
    "warm": {
        "drive": 1.45,
        "odd_even_balance": 0.25,
        "soft_clip": 0.12,
        "tilt": {"low": 1.00, "mid": 0.90, "high": 0.55},
        "description": "low/mid warmth with reduced brightness"
    },
    "bright": {
        "drive": 1.75,
        "odd_even_balance": 0.28,
        "soft_clip": 0.16,
        "tilt": {"low": 0.35, "mid": 0.75, "high": 1.35},
        "description": "upper harmonic lift and air enhancement"
    },
    "modern": {
        "drive": 1.8,
        "odd_even_balance": 0.30,
        "soft_clip": 0.18,
        "tilt": {"low": 0.50, "mid": 0.85, "high": 1.10},
        "description": "clean modern multiband saturation"
    }
}

BANDS = {
    "low": (40, 180),
    "mid": (180, 4000),
    "high": (4000, 18000),
}

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def parse_amount(value, default=0.12):
    try:
        v = float(value)
    except Exception:
        v = default
    if v > 1.0:
        v /= 100.0
    return clamp(v, 0.0, 1.0)

def harmonic_signal(x: np.ndarray, drive: float, odd_even_balance: float, soft_clip: float) -> np.ndarray:
    # Odd harmonic component: symmetric tanh saturation.
    odd = np.tanh(x * drive) / (np.tanh(drive) + 1e-12)
    # Even-ish component: asymmetry term. Kept subtle to avoid DC/level drift.
    asym = np.tanh((x + 0.12) * drive) - np.tanh(0.12 * drive)
    asym = asym / (np.max(np.abs(asym)) + 1e-12)
    sat = odd * odd_even_balance + asym * (1.0 - odd_even_balance)
    # Soft clip blend.
    clipped = np.clip(sat, -1.0 + soft_clip * 0.15, 1.0 - soft_clip * 0.15)
    return sat * (1.0 - soft_clip) + clipped * soft_clip

def band_energy(audio, sr, low, high):
    b = bandpass(audio, sr, low, high)
    return float(np.mean(b*b) + 1e-12)

def harmonic_meter(audio, sr):
    spectrum = analyze_spectrum(audio, sr)
    low_e = band_energy(audio, sr, 40, 180)
    mid_e = band_energy(audio, sr, 180, 4000)
    high_e = band_energy(audio, sr, 4000, 18000)
    return {
        "spectrum": spectrum,
        "band_energy": {
            "low_40_180": low_e,
            "mid_180_4000": mid_e,
            "high_4000_18000": high_e,
        },
        "brightness_index": spectrum.get("brightness_index"),
        "air_index": spectrum.get("air_index"),
        "harshness_index": spectrum.get("harshness_index"),
    }

def infer_exciter_config(params: dict, full_analysis: dict | None):
    mode = str(params.get("mode", "modern")).lower()
    if mode not in MODE_PROFILES:
        mode = "modern"
    base_amount = parse_amount(params.get("amount", 0.12), 0.12)
    bands_param = params.get("bands", {}) or {}

    full_spectrum = full_analysis.get("spectrum", {}) if full_analysis else {}
    flags = full_analysis.get("diagnosis_flags", {}) if full_analysis else {}
    air_weak = bool(flags.get("weak_air", False))
    harsh = bool(flags.get("harsh_presence", False))

    profile = MODE_PROFILES[mode]
    low_amount = parse_amount(bands_param.get("low", base_amount * profile["tilt"]["low"]), base_amount * profile["tilt"]["low"])
    mid_amount = parse_amount(bands_param.get("mid", base_amount * profile["tilt"]["mid"]), base_amount * profile["tilt"]["mid"])
    high_amount = parse_amount(bands_param.get("high", base_amount * profile["tilt"]["high"]), base_amount * profile["tilt"]["high"])

    if air_weak:
        high_amount = max(high_amount, min(0.38, base_amount + 0.12))
    if harsh:
        high_amount = min(high_amount, 0.10)
        mid_amount = min(mid_amount, 0.14)

    return {
        "mode": mode,
        "profile": profile,
        "amount": round(base_amount, 6),
        "band_amounts": {
            "low": round(clamp(low_amount, 0.0, 0.6), 6),
            "mid": round(clamp(mid_amount, 0.0, 0.6), 6),
            "high": round(clamp(high_amount, 0.0, 0.6), 6),
        },
        "drive": float(params.get("drive", profile["drive"])),
        "odd_even_balance": float(params.get("odd_even_balance", profile["odd_even_balance"])),
        "soft_clip": float(params.get("soft_clip", profile["soft_clip"])),
        "oversafety_peak_db": float(params.get("oversafety_peak_db", -1.0)),
        "air_weak_auto_boost": air_weak,
        "harsh_auto_limit": harsh,
        "source_air_index": full_spectrum.get("air_index"),
        "source_harshness_index": full_spectrum.get("harshness_index"),
    }

def process_exciter_advanced(audio: np.ndarray, sr: int, params: dict, full_analysis: dict | None = None):
    cfg = infer_exciter_config(params, full_analysis)
    before = harmonic_meter(audio, sr)
    before_basic = analyze_basic_audio(audio, sr)

    wet_total = np.zeros_like(audio)
    band_reports = []
    for name, (low, high) in BANDS.items():
        band = bandpass(audio, sr, low, high)
        amount = cfg["band_amounts"][name]
        if amount <= 0:
            processed = band
            harmonic_delta = 0.0
        else:
            saturated = harmonic_signal(band, cfg["drive"], cfg["odd_even_balance"], cfg["soft_clip"])
            harmonic = saturated - band
            processed = band + harmonic * amount
            harmonic_delta = float(np.mean((processed-band)**2))
        wet_total += processed
        band_reports.append({
            "band": name,
            "range_hz": [low, high],
            "amount": amount,
            "harmonic_delta_energy": harmonic_delta,
            "input_energy": float(np.mean(band*band) + 1e-12),
            "output_energy": float(np.mean(processed*processed) + 1e-12),
        })

    # Residual outside split range kept dry.
    dry_split = sum([bandpass(audio, sr, low, high) for low, high in BANDS.values()])
    residual = audio - dry_split
    y = residual + wet_total
    y = normalize_peak(sanitize(y), cfg["oversafety_peak_db"])

    after = harmonic_meter(y, sr)
    after_basic = analyze_basic_audio(y, sr)

    # Safety pass: if saturation creates too much harshness, blend back high band.
    safety_blend_applied = False
    if after["harshness_index"] > before["harshness_index"] + 0.08:
        high_original = bandpass(audio, sr, 4000, 18000)
        high_processed = bandpass(y, sr, 4000, 18000)
        y = y - high_processed + high_original * 0.45 + high_processed * 0.55
        y = normalize_peak(sanitize(y), cfg["oversafety_peak_db"])
        after = harmonic_meter(y, sr)
        after_basic = analyze_basic_audio(y, sr)
        safety_blend_applied = True

    high_before = before["band_energy"]["high_4000_18000"]
    high_after = after["band_energy"]["high_4000_18000"]
    air_before = before["air_index"]
    air_after = after["air_index"]

    clipping_warning = after_basic["clipping_samples"] > 0 or after_basic["peak_dbfs"] > -0.1
    harshness_warning = after["harshness_index"] > before["harshness_index"] + 0.08

    report = {
        "task": "Task 027 - Exciter / Saturation",
        "status": "success",
        "config": {
            k: v for k, v in cfg.items() if k != "profile"
        } | {"profile_description": cfg["profile"]["description"]},
        "band_reports": band_reports,
        "before_meter": before,
        "after_meter": after,
        "before_basic": before_basic,
        "after_basic": after_basic,
        "high_band_energy_before": high_before,
        "high_band_energy_after": high_after,
        "high_band_energy_delta": round(high_after - high_before, 12),
        "air_index_before": air_before,
        "air_index_after": air_after,
        "air_index_delta": round(float(air_after - air_before), 8),
        "brightness_index_before": before["brightness_index"],
        "brightness_index_after": after["brightness_index"],
        "harshness_index_before": before["harshness_index"],
        "harshness_index_after": after["harshness_index"],
        "clipping_warning": bool(clipping_warning),
        "harshness_warning": bool(harshness_warning),
        "peak_safety_applied": True,
        "harshness_safety_blend_applied": bool(safety_blend_applied),
    }
    return sanitize(y), report



