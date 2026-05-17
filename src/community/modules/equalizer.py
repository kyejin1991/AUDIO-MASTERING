from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy import signal

from .dsp_utils import sanitize, highpass, lowpass, peaking_eq, highshelf, lowshelf, mid_side, from_mid_side, db20
from community.analysis.spectrum import analyze_spectrum

SUPPORTED_FILTER_TYPES = {"highpass", "lowpass", "bell", "low_shelf", "high_shelf", "notch"}
SUPPORTED_MODES = {"stereo", "mid", "side", "left", "right"}

DEFAULT_8_BAND_EQ = [
    {"id": 1, "type": "highpass", "freq_hz": 24, "gain_db": 0.0, "q": 0.707, "mode": "stereo", "enabled": True},
    {"id": 2, "type": "low_shelf", "freq_hz": 90, "gain_db": 0.0, "q": 0.707, "mode": "stereo", "enabled": True},
    {"id": 3, "type": "bell", "freq_hz": 180, "gain_db": 0.0, "q": 0.9, "mode": "mid", "enabled": True},
    {"id": 4, "type": "bell", "freq_hz": 280, "gain_db": -1.0, "q": 0.9, "mode": "mid", "enabled": True},
    {"id": 5, "type": "bell", "freq_hz": 1200, "gain_db": 0.0, "q": 1.0, "mode": "stereo", "enabled": True},
    {"id": 6, "type": "bell", "freq_hz": 4200, "gain_db": 0.0, "q": 1.1, "mode": "mid", "enabled": True},
    {"id": 7, "type": "high_shelf", "freq_hz": 10500, "gain_db": 0.5, "q": 0.707, "mode": "side", "enabled": True},
    {"id": 8, "type": "lowpass", "freq_hz": 19500, "gain_db": 0.0, "q": 0.707, "mode": "stereo", "enabled": False},
]

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))

def normalize_band(raw: dict, band_id: int) -> dict:
    typ = str(raw.get("type", "bell")).lower()
    mode = str(raw.get("mode", "stereo")).lower()
    if typ not in SUPPORTED_FILTER_TYPES:
        typ = "bell"
    if mode not in SUPPORTED_MODES:
        mode = "stereo"
    freq = clamp(raw.get("freq_hz", 1000), 10, 22000)
    gain = clamp(raw.get("gain_db", 0.0), -18.0, 18.0)
    q = clamp(raw.get("q", 0.707), 0.1, 12.0)
    enabled = bool(raw.get("enabled", True))
    return {
        "id": int(raw.get("id", band_id)),
        "type": typ,
        "freq_hz": round(freq, 4),
        "gain_db": round(gain, 4),
        "q": round(q, 4),
        "mode": mode,
        "enabled": enabled,
    }

def expand_to_8_bands(input_bands: list[dict] | None) -> list[dict]:
    """
    Assistant媛 4媛?band留?以섎룄 8-band EQ rack?쇰줈 ?뺤옣?쒕떎.
    """
    result = [dict(b) for b in DEFAULT_8_BAND_EQ]
    if not input_bands:
        return [normalize_band(b, i+1) for i, b in enumerate(result)]
    for i, b in enumerate(input_bands[:8]):
        merged = dict(result[i])
        merged.update(b)
        merged["id"] = i + 1
        merged["enabled"] = b.get("enabled", True)
        result[i] = merged
    return [normalize_band(b, i+1) for i, b in enumerate(result)]

def auto_enhance_bands_from_analysis(bands: list[dict], full_analysis: dict | None) -> tuple[list[dict], dict]:
    """
    full_analysis瑜??ъ슜??8-band EQ瑜??먮룞 蹂닿컯?쒕떎.
    assistant bands媛 ?대? ?ㅼ뼱????꾨씫?????쓣 ?곌뎄?⑹쑝濡?蹂댁젙?쒕떎.
    """
    if not full_analysis:
        return bands, {"auto_enhance_used": False, "reason": "full_analysis_missing"}

    spectrum = full_analysis.get("spectrum", {})
    flags = full_analysis.get("diagnosis_flags", {})
    auto = {"auto_enhance_used": True, "adjustments": []}
    out = [dict(b) for b in bands]

    low_end = float(spectrum.get("low_end_index", 0.0))
    mud = float(spectrum.get("mud_index", 0.0))
    harsh = float(spectrum.get("harshness_index", 0.0))
    air = float(spectrum.get("air_index", 0.0))

    # Band 2: low shelf
    if low_end > 0.55:
        out[1]["gain_db"] = min(out[1]["gain_db"], -1.5)
        auto["adjustments"].append("low_end_excess_low_shelf_cut")
    elif low_end < 0.14:
        out[1]["gain_db"] = max(out[1]["gain_db"], 1.0)
        auto["adjustments"].append("thin_low_end_low_shelf_boost")

    # Band 4: mud cut
    if mud > 0.25:
        cut = -1.2 - min(3.0, (mud - 0.25) * 12.0)
        out[3]["type"] = "bell"
        out[3]["freq_hz"] = 280
        out[3]["gain_db"] = min(out[3]["gain_db"], cut)
        out[3]["q"] = 0.9
        out[3]["mode"] = "mid"
        out[3]["enabled"] = True
        auto["adjustments"].append("mud_cut_280hz_mid")

    # Band 6: harshness cut
    if harsh > 0.16 or flags.get("harsh_presence", False):
        cut = -1.0 - min(3.5, max(0.0, harsh - 0.16) * 10.0)
        out[5]["type"] = "bell"
        out[5]["freq_hz"] = 4200
        out[5]["gain_db"] = min(out[5]["gain_db"], cut)
        out[5]["q"] = 1.2
        out[5]["mode"] = "mid"
        out[5]["enabled"] = True
        auto["adjustments"].append("harshness_cut_4200hz_mid")

    # Band 7: air boost
    if air < 0.05 or flags.get("weak_air", False):
        boost = 0.8 + min(2.5, max(0.0, 0.05 - air) * 30.0)
        out[6]["type"] = "high_shelf"
        out[6]["freq_hz"] = 10500
        out[6]["gain_db"] = max(out[6]["gain_db"], boost)
        out[6]["q"] = 0.707
        out[6]["mode"] = "side"
        out[6]["enabled"] = True
        auto["adjustments"].append("air_boost_10500hz_side")

    return [normalize_band(b, i+1) for i, b in enumerate(out)], auto

def apply_filter_to_array(arr: np.ndarray, sr: int, band: dict) -> np.ndarray:
    typ = band["type"]
    freq = band["freq_hz"]
    gain = band["gain_db"]
    q = band["q"]
    if not band["enabled"]:
        return arr
    if typ == "highpass":
        return highpass(arr, sr, freq)
    if typ == "lowpass":
        return lowpass(arr, sr, freq)
    if typ == "bell":
        return peaking_eq(arr, sr, freq, gain, q)
    if typ == "low_shelf":
        return lowshelf(arr, sr, freq, gain)
    if typ == "high_shelf":
        return highshelf(arr, sr, freq, gain)
    if typ == "notch":
        w0 = max(20.0, min(freq, sr/2 - 200)) / (sr / 2)
        b, a = signal.iirnotch(w0, max(q, 0.2))
        return signal.filtfilt(b, a, arr, axis=0)
    return arr

def apply_band(audio: np.ndarray, sr: int, band: dict) -> np.ndarray:
    mode = band["mode"]
    y = audio.copy()
    if not band["enabled"]:
        return y

    if mode == "stereo":
        return apply_filter_to_array(y, sr, band)

    if mode in {"mid", "side"}:
        mid, side = mid_side(y)
        arr = mid[:, None] if mode == "mid" else side[:, None]
        arr2 = apply_filter_to_array(arr, sr, band)[:, 0]
        return from_mid_side(arr2, side) if mode == "mid" else from_mid_side(mid, arr2)

    if mode == "left":
        y[:, 0:1] = apply_filter_to_array(y[:, 0:1], sr, band)
        return y

    if mode == "right":
        y[:, 1:2] = apply_filter_to_array(y[:, 1:2], sr, band)
        return y

    return y

def band_diff(before: dict, after: dict) -> dict:
    b = before.get("bands", {})
    a = after.get("bands", {})
    keys = sorted(set(b) | set(a))
    return {k: round(float(a.get(k, 0.0) - b.get(k, 0.0)), 6) for k in keys}

def process_equalizer_advanced(audio: np.ndarray, sr: int, params: dict, full_analysis: dict | None = None) -> tuple[np.ndarray, dict]:
    """
    Task 020 ?ㅼ젣 EQ 泥섎━.
    8-band rack + Mid/Side/Left/Right/Stereo + auto enhancement.
    """
    before_spectrum = analyze_spectrum(audio, sr)
    bands = expand_to_8_bands(params.get("bands"))
    bands, auto_report = auto_enhance_bands_from_analysis(bands, full_analysis)

    y = sanitize(audio.copy())
    applied = []
    for band in bands:
        before_peak = float(np.max(np.abs(y))) if y.size else 0.0
        y = apply_band(y, sr, band)
        after_peak = float(np.max(np.abs(y))) if y.size else 0.0
        applied.append({
            **band,
            "peak_before": round(before_peak, 8),
            "peak_after": round(after_peak, 8),
            "actually_processed": bool(band["enabled"]),
        })

    after_spectrum = analyze_spectrum(y, sr)

    report = {
        "task": "Task 020 - Equalizer",
        "status": "success",
        "eq_type": "8_band_parametric_mid_side_lr",
        "supported_filter_types": sorted(SUPPORTED_FILTER_TYPES),
        "supported_modes": sorted(SUPPORTED_MODES),
        "bands": bands,
        "applied_bands": applied,
        "auto_report": auto_report,
        "before_spectrum": before_spectrum,
        "after_spectrum": after_spectrum,
        "spectrum_band_delta": band_diff(before_spectrum, after_spectrum),
        "mud_index_before": before_spectrum.get("mud_index"),
        "mud_index_after": after_spectrum.get("mud_index"),
        "harshness_index_before": before_spectrum.get("harshness_index"),
        "harshness_index_after": after_spectrum.get("harshness_index"),
        "air_index_before": before_spectrum.get("air_index"),
        "air_index_after": after_spectrum.get("air_index"),
    }
    return sanitize(y), report



