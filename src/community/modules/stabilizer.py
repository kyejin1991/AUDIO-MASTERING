from __future__ import annotations
import numpy as np

from .dsp_utils import sanitize, lowshelf, highshelf, peaking_eq
from community.analysis.spectrum import analyze_spectrum

TARGET_CURVES = {
    "hiphop": {
        "sub_20_60": 0.18, "bass_60_150": 0.24, "low_mid_150_500": 0.18,
        "mid_500_2500": 0.18, "presence_2500_8000": 0.14, "air_8000_18000": 0.08,
    },
    "trap": {
        "sub_20_60": 0.22, "bass_60_150": 0.24, "low_mid_150_500": 0.16,
        "mid_500_2500": 0.16, "presence_2500_8000": 0.14, "air_8000_18000": 0.08,
    },
    "drill": {
        "sub_20_60": 0.20, "bass_60_150": 0.22, "low_mid_150_500": 0.16,
        "mid_500_2500": 0.18, "presence_2500_8000": 0.16, "air_8000_18000": 0.08,
    },
    "rnb": {
        "sub_20_60": 0.12, "bass_60_150": 0.20, "low_mid_150_500": 0.22,
        "mid_500_2500": 0.22, "presence_2500_8000": 0.15, "air_8000_18000": 0.09,
    },
    "trapsoul": {
        "sub_20_60": 0.16, "bass_60_150": 0.22, "low_mid_150_500": 0.20,
        "mid_500_2500": 0.20, "presence_2500_8000": 0.14, "air_8000_18000": 0.08,
    },
    "edm": {
        "sub_20_60": 0.18, "bass_60_150": 0.22, "low_mid_150_500": 0.14,
        "mid_500_2500": 0.18, "presence_2500_8000": 0.17, "air_8000_18000": 0.11,
    },
    "house": {
        "sub_20_60": 0.14, "bass_60_150": 0.25, "low_mid_150_500": 0.16,
        "mid_500_2500": 0.18, "presence_2500_8000": 0.16, "air_8000_18000": 0.11,
    },
    "pop": {
        "sub_20_60": 0.10, "bass_60_150": 0.18, "low_mid_150_500": 0.20,
        "mid_500_2500": 0.24, "presence_2500_8000": 0.18, "air_8000_18000": 0.10,
    },
    "ballad": {
        "sub_20_60": 0.08, "bass_60_150": 0.16, "low_mid_150_500": 0.22,
        "mid_500_2500": 0.28, "presence_2500_8000": 0.17, "air_8000_18000": 0.09,
    },
    "rock": {
        "sub_20_60": 0.09, "bass_60_150": 0.18, "low_mid_150_500": 0.22,
        "mid_500_2500": 0.25, "presence_2500_8000": 0.18, "air_8000_18000": 0.08,
    },
    "lofi": {
        "sub_20_60": 0.12, "bass_60_150": 0.20, "low_mid_150_500": 0.26,
        "mid_500_2500": 0.24, "presence_2500_8000": 0.12, "air_8000_18000": 0.06,
    },
    "cinematic": {
        "sub_20_60": 0.18, "bass_60_150": 0.20, "low_mid_150_500": 0.18,
        "mid_500_2500": 0.20, "presence_2500_8000": 0.14, "air_8000_18000": 0.10,
    },
}

BAND_TO_FILTER = {
    "sub_20_60": ("low_shelf", 45, 0.7),
    "bass_60_150": ("low_shelf", 105, 0.7),
    "low_mid_150_500": ("bell", 280, 0.85),
    "mid_500_2500": ("bell", 1200, 1.0),
    "presence_2500_8000": ("bell", 4200, 1.15),
    "air_8000_18000": ("high_shelf", 10500, 0.7),
}

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def infer_genre(params: dict, full_analysis: dict | None) -> str:
    genre = params.get("genre")
    if genre:
        return str(genre).lower()
    if full_analysis:
        # project-level assistant info may not be in full analysis, fallback pop
        return str(full_analysis.get("project", {}).get("genre", "pop")).lower()
    return "pop"

def distance_to_target(bands: dict, target: dict) -> float:
    keys = target.keys()
    return float(sum(abs(float(bands.get(k, 0.0)) - float(target[k])) for k in keys))

def build_correction_curve(current_bands: dict, target: dict, amount: float, max_gain_db: float) -> list[dict]:
    corrections = []
    for band_name, target_value in target.items():
        current_value = float(current_bands.get(band_name, 0.0))
        diff = float(target_value) - current_value
        # Convert normalized energy difference to a conservative dB move.
        raw_gain = diff * 18.0 * amount
        gain_db = clamp(raw_gain, -max_gain_db, max_gain_db)
        ftype, freq, q = BAND_TO_FILTER[band_name]
        # Avoid tiny meaningless filters.
        enabled = abs(gain_db) >= 0.15
        corrections.append({
            "band": band_name,
            "filter_type": ftype,
            "freq_hz": freq,
            "q": q,
            "current": round(current_value, 6),
            "target": round(float(target_value), 6),
            "diff": round(diff, 6),
            "gain_db": round(gain_db, 4),
            "enabled": enabled,
        })
    return corrections

def apply_corrections(audio, sr, corrections):
    y = audio.copy()
    applied = []
    # order: cuts/low first then highs
    for c in corrections:
        if not c["enabled"]:
            applied.append({**c, "processed": False})
            continue
        typ = c["filter_type"]
        gain = c["gain_db"]
        freq = c["freq_hz"]
        q = c["q"]
        if typ == "low_shelf":
            y = lowshelf(y, sr, freq, gain)
        elif typ == "high_shelf":
            y = highshelf(y, sr, freq, gain)
        elif typ == "bell":
            y = peaking_eq(y, sr, freq, gain, q)
        applied.append({**c, "processed": True})
    return sanitize(y), applied

def process_stabilizer_advanced(audio, sr, params: dict, full_analysis: dict | None = None):
    amount = params.get("amount", 0.35)
    # accept 0~100 or 0~1
    amount = float(amount)
    if amount > 1.0:
        amount = amount / 100.0
    amount = clamp(amount, 0.0, 1.0)
    max_gain_db = clamp(params.get("max_gain_db", 3.0), 0.5, 8.0)
    genre = infer_genre(params, full_analysis)
    target = TARGET_CURVES.get(genre, TARGET_CURVES["pop"])

    before = analyze_spectrum(audio, sr)
    current_bands = before["bands"]
    before_distance = distance_to_target(current_bands, target)

    corrections = build_correction_curve(current_bands, target, amount, max_gain_db)
    y, applied = apply_corrections(audio, sr, corrections)

    after = analyze_spectrum(y, sr)
    after_distance = distance_to_target(after["bands"], target)

    # If target distance got worse, retry at half amount.
    rollback_used = False
    if after_distance > before_distance and amount > 0.05:
        corrections2 = build_correction_curve(current_bands, target, amount * 0.5, max_gain_db)
        y2, applied2 = apply_corrections(audio, sr, corrections2)
        after2 = analyze_spectrum(y2, sr)
        after2_distance = distance_to_target(after2["bands"], target)
        if after2_distance <= after_distance:
            y, applied, after, after_distance = y2, applied2, after2, after2_distance
            rollback_used = True

    report = {
        "task": "Task 022 - Stabilizer",
        "status": "success",
        "genre": genre,
        "amount": round(amount, 4),
        "max_gain_db": max_gain_db,
        "target_curve": target,
        "correction_curve": corrections,
        "applied_corrections": applied,
        "before_spectrum": before,
        "after_spectrum": after,
        "target_distance_before": round(before_distance, 6),
        "target_distance_after": round(after_distance, 6),
        "target_distance_improved": bool(after_distance <= before_distance),
        "rollback_used": rollback_used,
    }
    return sanitize(y), report



