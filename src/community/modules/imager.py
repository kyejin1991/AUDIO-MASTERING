from __future__ import annotations
import numpy as np

from .dsp_utils import sanitize, bandpass, lowpass, mid_side, from_mid_side, normalize_peak
from community.analysis.stereo import analyze_stereo
from community.analysis.spectrum import analyze_spectrum

DEFAULT_BANDS = [
    {"id": 1, "name": "low", "low_hz": 20, "high_hz": 120, "width": 0.0, "enabled": True},
    {"id": 2, "name": "low_mid", "low_hz": 120, "high_hz": 500, "width": 0.85, "enabled": True},
    {"id": 3, "name": "mid", "low_hz": 500, "high_hz": 4000, "width": 1.0, "enabled": True},
    {"id": 4, "name": "high", "low_hz": 4000, "high_hz": 18000, "width": 1.12, "enabled": True},
]

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def normalize_band(raw: dict, idx: int):
    base = dict(DEFAULT_BANDS[min(idx, len(DEFAULT_BANDS)-1)])
    base.update(raw or {})
    low = clamp(base.get("low_hz", 20), 10, 19000)
    high = clamp(base.get("high_hz", 120), low+20, 20000)
    return {
        "id": int(base.get("id", idx+1)),
        "name": str(base.get("name", f"band_{idx+1}")),
        "low_hz": round(low, 4),
        "high_hz": round(high, 4),
        "width": round(clamp(base.get("width", 1.0), 0.0, 2.0), 4),
        "side_gain_db": round(clamp(base.get("side_gain_db", 0.0), -18.0, 18.0), 4),
        "enabled": bool(base.get("enabled", True)),
    }

def db_to_gain(db):
    return 10.0 ** (float(db) / 20.0)

def build_bands(params: dict, full_analysis: dict | None):
    raw = params.get("bands")
    if isinstance(raw, list) and raw:
        bands = [normalize_band(b, i) for i, b in enumerate(raw[:4])]
        while len(bands) < 4:
            bands.append(normalize_band({}, len(bands)))
    else:
        bands = [
            normalize_band({"width": float(params.get("low_width", 0.0)), "name": "low", "low_hz": 20, "high_hz": 120}, 0),
            normalize_band({"width": float(params.get("low_mid_width", 0.85)), "name": "low_mid", "low_hz": 120, "high_hz": 500}, 1),
            normalize_band({"width": float(params.get("mid_width", 1.0)), "name": "mid", "low_hz": 500, "high_hz": 4000}, 2),
            normalize_band({"width": float(params.get("high_width", 1.12)), "name": "high", "low_hz": 4000, "high_hz": 18000}, 3),
        ]

    stereo = full_analysis.get("stereo", {}) if full_analysis else {}
    flags = full_analysis.get("diagnosis_flags", {}) if full_analysis else {}
    leakage = float(stereo.get("low_end_stereo_leakage", 0.0))
    phase = float(stereo.get("phase_correlation", 1.0))

    mono_below_hz = float(params.get("mono_below_hz", 120))
    if leakage > 0.15 or flags.get("wide_low_end", False):
        mono_below_hz = max(mono_below_hz, 150.0)
        bands[0]["width"] = 0.0

    # If phase is risky, avoid widening and narrow sides slightly.
    phase_safety_scale = 1.0
    if phase < 0.25 or flags.get("phase_risk", False):
        phase_safety_scale = 0.75
        for b in bands:
            if b["width"] > 1.0:
                b["width"] = round(1.0 + (b["width"] - 1.0) * 0.35, 4)
            else:
                b["width"] = round(b["width"] * 0.9, 4)

    return bands, {
        "mono_below_hz": round(clamp(mono_below_hz, 40, 250), 4),
        "phase_safety_scale": phase_safety_scale,
        "source_low_end_stereo_leakage": leakage,
        "source_phase_correlation": phase,
    }

def apply_band_width(audio, sr, band):
    split = bandpass(audio, sr, band["low_hz"], band["high_hz"])
    if not band["enabled"]:
        return split, {
            "band": band,
            "processed": False,
            "mid_energy": float(np.mean(((split[:,0]+split[:,1])*0.5)**2)),
            "side_energy": float(np.mean(((split[:,0]-split[:,1])*0.5)**2)),
        }
    mid, side = mid_side(split)
    side_before = float(np.mean(side*side))
    mid_energy = float(np.mean(mid*mid))
    side = side * band["width"] * db_to_gain(band.get("side_gain_db", 0.0))
    out = from_mid_side(mid, side)
    side_after = float(np.mean(side*side))
    return out, {
        "band": band,
        "processed": True,
        "mid_energy": mid_energy,
        "side_energy_before": side_before,
        "side_energy_after": side_after,
        "side_energy_ratio": float(side_after / (side_before + 1e-12)),
    }

def low_mono(audio, sr, cutoff_hz):
    low = lowpass(audio, sr, cutoff_hz)
    high = audio - low
    mono_low = np.repeat(np.mean(low, axis=1, keepdims=True), 2, axis=1)
    return mono_low + high

def stereo_quality(stereo_report):
    phase = stereo_report.get("phase_correlation", 1.0)
    mono_loss = stereo_report.get("mono_collapse_loss_db", 0.0)
    leakage = stereo_report.get("low_end_stereo_leakage", 0.0)
    # Higher is better. Penalize negative correlation, mono loss, leakage.
    return float(phase - max(0.0, -mono_loss - 3.0) * 0.08 - leakage * 0.35)

def process_imager_advanced(audio, sr, params: dict, full_analysis: dict | None = None):
    before_stereo = analyze_stereo(audio, sr)
    before_spectrum = analyze_spectrum(audio, sr)
    before_quality = stereo_quality(before_stereo)

    bands, build_report = build_bands(params, full_analysis)

    # Band reconstruction
    band_outputs = []
    band_reports = []
    original_sum = np.zeros_like(audio)
    for b in bands:
        original_band = bandpass(audio, sr, b["low_hz"], b["high_hz"])
        original_sum += original_band
        out, rep = apply_band_width(audio, sr, b)
        band_outputs.append(out)
        band_reports.append(rep)

    residual = audio - original_sum
    y = residual + sum(band_outputs)
    y = low_mono(y, sr, build_report["mono_below_hz"])
    y = normalize_peak(sanitize(y), -1.0)

    after_stereo = analyze_stereo(y, sr)
    after_spectrum = analyze_spectrum(y, sr)
    after_quality = stereo_quality(after_stereo)

    rollback_used = False
    rollback_reason = None
    # Protection: do not allow phase/mono compatibility to get materially worse.
    if (
        after_stereo.get("phase_correlation", 1.0) < -0.05
        or after_stereo.get("mono_collapse_loss_db", 0.0) < before_stereo.get("mono_collapse_loss_db", 0.0) - 2.0
        or after_quality < before_quality - 0.18
    ):
        rollback_used = True
        rollback_reason = "phase_or_mono_compatibility_protection"
        # Apply a conservative version: low mono only + half side changes.
        conservative = []
        original_sum = np.zeros_like(audio)
        for b in bands:
            bb = dict(b)
            bb["width"] = round(1.0 + (bb["width"] - 1.0) * 0.35, 4)
            if bb["name"] == "low":
                bb["width"] = 0.0
            original_band = bandpass(audio, sr, bb["low_hz"], bb["high_hz"])
            original_sum += original_band
            out, _ = apply_band_width(audio, sr, bb)
            conservative.append(out)
        y = audio - original_sum + sum(conservative)
        y = low_mono(y, sr, build_report["mono_below_hz"])
        y = normalize_peak(sanitize(y), -1.0)
        after_stereo = analyze_stereo(y, sr)
        after_spectrum = analyze_spectrum(y, sr)
        after_quality = stereo_quality(after_stereo)

    report = {
        "task": "Task 028 - Imager",
        "status": "success",
        "bands": bands,
        "build_report": build_report,
        "band_reports": band_reports,
        "before_stereo": before_stereo,
        "after_stereo": after_stereo,
        "before_spectrum": before_spectrum,
        "after_spectrum": after_spectrum,
        "stereo_width_before": before_stereo.get("stereo_width"),
        "stereo_width_after": after_stereo.get("stereo_width"),
        "phase_correlation_before": before_stereo.get("phase_correlation"),
        "phase_correlation_after": after_stereo.get("phase_correlation"),
        "mono_collapse_loss_before": before_stereo.get("mono_collapse_loss_db"),
        "mono_collapse_loss_after": after_stereo.get("mono_collapse_loss_db"),
        "low_end_stereo_leakage_before": before_stereo.get("low_end_stereo_leakage"),
        "low_end_stereo_leakage_after": after_stereo.get("low_end_stereo_leakage"),
        "stereo_quality_before": round(before_quality, 6),
        "stereo_quality_after": round(after_quality, 6),
        "rollback_used": rollback_used,
        "rollback_reason": rollback_reason,
        "phase_protection_passed": bool(after_stereo.get("phase_correlation", 1.0) >= -0.05),
        "mono_compatibility_passed": bool(after_stereo.get("mono_collapse_loss_db", 0.0) >= -6.0),
    }
    return sanitize(y), report



