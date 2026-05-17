from __future__ import annotations
import numpy as np
from scipy import signal

from .dsp_utils import sanitize, db20, undb20, bandpass, lowpass, highpass, mid_side, from_mid_side
from community.analysis.spectrum import analyze_spectrum

SUPPORTED_DYNAMIC_EQ_TYPES = {"bell", "low_shelf", "high_shelf"}
SUPPORTED_DYNAMIC_EQ_MODES = {"stereo", "mid", "side"}

def clamp(value, lo, hi):
    return max(lo, min(hi, float(value)))

def smooth_envelope(raw: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    attack = np.exp(-1.0 / max(1.0, (attack_ms / 1000.0) * sr))
    release = np.exp(-1.0 / max(1.0, (release_ms / 1000.0) * sr))
    env = np.zeros_like(raw)
    prev = 0.0
    for i, x in enumerate(raw):
        coeff = attack if x > prev else release
        prev = coeff * prev + (1.0 - coeff) * x
        env[i] = prev
    return env

def normalize_band(raw: dict, idx: int) -> dict:
    typ = str(raw.get("type", "bell")).lower()
    if typ not in SUPPORTED_DYNAMIC_EQ_TYPES:
        typ = "bell"
    mode = str(raw.get("mode", "stereo")).lower()
    if mode not in SUPPORTED_DYNAMIC_EQ_MODES:
        mode = "stereo"
    freq = clamp(raw.get("freq_hz", 1000), 20, 18000)
    q = clamp(raw.get("q", 1.0), 0.15, 8.0)
    threshold = clamp(raw.get("threshold_db", -26), -80, 0)
    ratio = clamp(raw.get("ratio", 2.0), 1.0, 20.0)
    range_db = clamp(raw.get("range_db", -3.0), -18.0, 18.0)
    attack_ms = clamp(raw.get("attack_ms", 15.0), 0.1, 500.0)
    release_ms = clamp(raw.get("release_ms", 120.0), 1.0, 3000.0)
    direction = str(raw.get("direction", "downward")).lower()
    if direction not in {"downward", "upward"}:
        direction = "downward"
    enabled = bool(raw.get("enabled", True))
    return {
        "id": int(raw.get("id", idx)),
        "type": typ,
        "freq_hz": round(freq, 4),
        "q": round(q, 4),
        "threshold_db": round(threshold, 4),
        "ratio": round(ratio, 4),
        "range_db": round(range_db, 4),
        "attack_ms": round(attack_ms, 4),
        "release_ms": round(release_ms, 4),
        "direction": direction,
        "mode": mode,
        "enabled": enabled,
    }

def auto_bands_from_analysis(full_analysis: dict | None) -> list[dict]:
    if not full_analysis:
        return []
    spectrum = full_analysis.get("spectrum", {})
    flags = full_analysis.get("diagnosis_flags", {})
    bands = []
    mud = float(spectrum.get("mud_index", 0.0))
    harsh = float(spectrum.get("harshness_index", 0.0))
    air = float(spectrum.get("air_index", 0.0))
    low = float(spectrum.get("low_end_index", 0.0))

    if mud > 0.22 or flags.get("muddy_low_mid", False):
        bands.append({
            "id": 1, "type": "bell", "freq_hz": 260, "q": 0.9,
            "threshold_db": -30, "ratio": 2.0 + min(2.0, mud * 4),
            "range_db": -2.5 - min(2.0, mud * 4),
            "attack_ms": 25, "release_ms": 180, "direction": "downward", "mode": "mid", "enabled": True,
            "source": "auto_mud_control"
        })
    if harsh > 0.16 or flags.get("harsh_presence", False):
        bands.append({
            "id": 2, "type": "bell", "freq_hz": 4200, "q": 1.2,
            "threshold_db": -34, "ratio": 2.4 + min(3.0, harsh * 5),
            "range_db": -3.0 - min(3.0, harsh * 4),
            "attack_ms": 8, "release_ms": 120, "direction": "downward", "mode": "mid", "enabled": True,
            "source": "auto_harshness_control"
        })
    if air < 0.04 or flags.get("weak_air", False):
        bands.append({
            "id": 3, "type": "high_shelf", "freq_hz": 9500, "q": 0.7,
            "threshold_db": -45, "ratio": 1.5, "range_db": 1.5,
            "attack_ms": 80, "release_ms": 450, "direction": "upward", "mode": "side", "enabled": True,
            "source": "auto_air_lift"
        })
    if low > 0.55 or flags.get("too_much_low_end", False):
        bands.append({
            "id": 4, "type": "low_shelf", "freq_hz": 100, "q": 0.7,
            "threshold_db": -28, "ratio": 2.2, "range_db": -2.5,
            "attack_ms": 30, "release_ms": 220, "direction": "downward", "mode": "stereo", "enabled": True,
            "source": "auto_low_end_control"
        })
    return bands

def merge_bands(params: dict, full_analysis: dict | None) -> tuple[list[dict], dict]:
    raw_bands = params.get("bands", [])
    auto = auto_bands_from_analysis(full_analysis)
    combined = []
    # explicit bands first
    for b in raw_bands:
        bb = dict(b)
        if "type" not in bb:
            bb["type"] = "bell"
        if "direction" not in bb:
            bb["direction"] = "downward"
        if "attack_ms" not in bb:
            bb["attack_ms"] = 12 if float(bb.get("freq_hz", 1000)) > 2000 else 25
        if "release_ms" not in bb:
            bb["release_ms"] = 140
        if "enabled" not in bb:
            bb["enabled"] = True
        combined.append(bb)
    # add auto bands if not near same freq
    for ab in auto:
        freq = float(ab["freq_hz"])
        if not any(abs(float(b.get("freq_hz", 0)) - freq) < freq * 0.25 for b in combined):
            combined.append(ab)
    normalized = [normalize_band(b, i+1) | {"source": b.get("source", "assistant_or_manual")} for i, b in enumerate(combined[:8])]
    return normalized, {"auto_bands": auto, "explicit_band_count": len(raw_bands), "final_band_count": len(normalized)}

def split_mode(audio: np.ndarray, mode: str):
    if mode == "mid":
        mid, side = mid_side(audio)
        return mid[:, None], side
    if mode == "side":
        mid, side = mid_side(audio)
        return side[:, None], mid
    return audio, None

def join_mode(processed: np.ndarray, carrier, mode: str):
    if mode == "mid":
        side = carrier
        return from_mid_side(processed[:, 0], side)
    if mode == "side":
        mid = carrier
        return from_mid_side(mid, processed[:, 0])
    return processed

def extract_dynamic_band(target: np.ndarray, sr: int, band: dict) -> np.ndarray:
    freq = band["freq_hz"]
    typ = band["type"]
    if typ == "low_shelf":
        return lowpass(target, sr, freq * 1.8)
    if typ == "high_shelf":
        return highpass(target, sr, freq * 0.75)
    # bell
    low = max(20.0, freq / (1.0 + band["q"] * 0.6))
    high = min(sr/2 - 100, freq * (1.0 + band["q"] * 0.6))
    return bandpass(target, sr, low, high)

def apply_dynamic_eq_band(audio: np.ndarray, sr: int, band: dict) -> tuple[np.ndarray, dict]:
    if not band["enabled"]:
        return audio, {"band": band, "bypassed": True}

    target, carrier = split_mode(audio, band["mode"])
    dyn_band = extract_dynamic_band(target, sr, band)
    rest = target - dyn_band

    detector = np.sqrt(np.mean(dyn_band ** 2, axis=1) + 1e-12)
    detector_db = db20(detector)
    if band["direction"] == "downward":
        over = np.maximum(detector_db - band["threshold_db"], 0.0)
        raw_gr = over * (1.0 - 1.0 / max(1.0, band["ratio"]))
        raw_gr = np.minimum(raw_gr, abs(band["range_db"]))
        env_gr = smooth_envelope(raw_gr, sr, band["attack_ms"], band["release_ms"])
        gain = undb20(-env_gr)
        processed_band = dyn_band * gain[:, None]
        reduction_report = env_gr
        effective_range_db = -float(np.max(env_gr)) if len(env_gr) else 0.0
    else:
        under = np.maximum(band["threshold_db"] - detector_db, 0.0)
        raw_boost = under * (1.0 - 1.0 / max(1.0, band["ratio"]))
        raw_boost = np.minimum(raw_boost, abs(band["range_db"]))
        env_boost = smooth_envelope(raw_boost, sr, band["attack_ms"], band["release_ms"])
        gain = undb20(env_boost)
        processed_band = dyn_band * gain[:, None]
        reduction_report = -env_boost
        effective_range_db = float(np.max(env_boost)) if len(env_boost) else 0.0

    target_out = rest + processed_band
    out = join_mode(target_out, carrier, band["mode"])
    report = {
        "band": band,
        "detector_db": {
            "min": round(float(np.min(detector_db)), 4),
            "mean": round(float(np.mean(detector_db)), 4),
            "max": round(float(np.max(detector_db)), 4),
        },
        "gain_change_db": {
            "max_abs": round(float(np.max(np.abs(reduction_report))) if len(reduction_report) else 0.0, 4),
            "mean_abs": round(float(np.mean(np.abs(reduction_report))) if len(reduction_report) else 0.0, 4),
            "effective_range_db": round(effective_range_db, 4),
        },
        "active_ratio": round(float(np.mean(np.abs(reduction_report) > 0.05)) if len(reduction_report) else 0.0, 6),
    }
    return sanitize(out), report

def process_dynamic_eq_advanced(audio: np.ndarray, sr: int, params: dict, full_analysis: dict | None = None) -> tuple[np.ndarray, dict]:
    before_spectrum = analyze_spectrum(audio, sr)
    bands, merge_report = merge_bands(params, full_analysis)
    y = sanitize(audio.copy())
    band_reports = []
    for band in bands:
        y, rep = apply_dynamic_eq_band(y, sr, band)
        band_reports.append(rep)
    after_spectrum = analyze_spectrum(y, sr)
    report = {
        "task": "Task 021 - Dynamic EQ",
        "status": "success",
        "dynamic_eq_type": "multiband_detector_envelope",
        "bands": bands,
        "merge_report": merge_report,
        "band_reports": band_reports,
        "before_spectrum": before_spectrum,
        "after_spectrum": after_spectrum,
        "mud_index_before": before_spectrum.get("mud_index"),
        "mud_index_after": after_spectrum.get("mud_index"),
        "harshness_index_before": before_spectrum.get("harshness_index"),
        "harshness_index_after": after_spectrum.get("harshness_index"),
        "air_index_before": before_spectrum.get("air_index"),
        "air_index_after": after_spectrum.get("air_index"),
        "max_gain_change_db": round(max([r.get("gain_change_db", {}).get("max_abs", 0.0) for r in band_reports] or [0.0]), 4),
    }
    return sanitize(y), report



