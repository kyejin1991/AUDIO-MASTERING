from __future__ import annotations
import numpy as np

from .dsp_utils import sanitize, highpass, db20, undb20, normalize_peak
from community.analysis.dynamics import analyze_dynamics
from community.analysis.loudness import analyze_loudness

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

def detector_level_db(audio: np.ndarray, sr: int, mode: str, sidechain_highpass_hz: float):
    detector_audio = audio
    if sidechain_highpass_hz and sidechain_highpass_hz > 20:
        detector_audio = highpass(audio, sr, sidechain_highpass_hz, order=1)
    mono = np.mean(detector_audio, axis=1)

    if mode == "peak":
        level = np.abs(mono) + 1e-12
        return db20(level)

    # RMS detector via moving average
    win = max(8, int(sr * 0.012))
    kernel = np.ones(win) / win
    power = np.convolve(mono * mono, kernel, mode="same")
    return db20(np.sqrt(np.maximum(power, 1e-12)))

def soft_knee_gain_reduction(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float) -> np.ndarray:
    x = level_db - threshold_db
    if knee_db <= 0:
        over = np.maximum(x, 0.0)
    else:
        lower = -knee_db / 2.0
        upper = knee_db / 2.0
        over = np.zeros_like(x)
        above = x >= upper
        mid = (x > lower) & (x < upper)
        over[above] = x[above]
        over[mid] = ((x[mid] - lower) ** 2) / (2.0 * knee_db)
    return over * (1.0 - 1.0 / max(ratio, 1.0))

def infer_params(params: dict, full_analysis: dict | None):
    dynamics = full_analysis.get("dynamics", {}) if full_analysis else {}
    flags = full_analysis.get("diagnosis_flags", {}) if full_analysis else {}
    over = float(dynamics.get("overcompression_score", 0.0))
    crest = float(dynamics.get("crest_factor_db", 12.0))
    dr = float(dynamics.get("dynamic_range_approx", 8.0))

    detector_mode = str(params.get("detector_mode", "rms")).lower()
    if detector_mode not in {"rms", "peak"}:
        detector_mode = "rms"

    threshold_db = float(params.get("threshold_db", -18.0))
    ratio = float(params.get("ratio", 1.5))
    attack_ms = float(params.get("attack_ms", 25.0))
    release_ms = float(params.get("release_ms", 140.0))
    knee_db = float(params.get("knee_db", 4.0))
    makeup_gain_db = float(params.get("makeup_gain_db", params.get("makeup_db", 0.0)))
    auto_makeup = bool(params.get("auto_makeup", True))
    auto_threshold = bool(params.get("auto_threshold", True))
    min_gain_reduction_db = float(params.get("min_gain_reduction_db", 0.35))
    mix = float(params.get("mix", 0.45))
    sidechain_highpass_hz = float(params.get("sidechain_highpass_hz", 90.0))

    # Safety: if already overcompressed, keep compressor very light.
    safety_scale = 1.0
    if over > 0.72 or flags.get("overcompressed", False):
        safety_scale = 0.35
    elif over > 0.55:
        safety_scale = 0.55

    # If very dynamic, allow more glue.
    if dr > 12.0 and over < 0.5:
        ratio += 0.25
        mix += 0.08

    # If crest already low, use slower/lighter settings.
    if crest < 8.0:
        attack_ms = max(attack_ms, 35.0)
        ratio = min(ratio, 1.35)
        mix = min(mix, 0.30)

    ratio = 1.0 + (ratio - 1.0) * safety_scale
    mix = mix * safety_scale

    return {
        "detector_mode": detector_mode,
        "threshold_db": round(clamp(threshold_db, -80, 0), 4),
        "ratio": round(clamp(ratio, 1.0, 20.0), 4),
        "attack_ms": round(clamp(attack_ms, 0.1, 500), 4),
        "release_ms": round(clamp(release_ms, 1, 3000), 4),
        "knee_db": round(clamp(knee_db, 0, 24), 4),
        "makeup_gain_db": round(clamp(makeup_gain_db, -12, 12), 4),
        "auto_makeup": auto_makeup,
        "auto_threshold": auto_threshold,
        "min_gain_reduction_db": round(clamp(min_gain_reduction_db, 0.0, 3.0), 4),
        "mix": round(clamp(mix, 0, 1), 4),
        "sidechain_highpass_hz": round(clamp(sidechain_highpass_hz, 0, 500), 4),
        "overcompression_safety_scale": round(safety_scale, 4),
    }

def process_compressor_advanced(audio: np.ndarray, sr: int, params: dict, full_analysis: dict | None = None):
    cfg = infer_params(params, full_analysis)
    before_dynamics = analyze_dynamics(audio, sr)
    before_loudness = analyze_loudness(audio, sr)

    level_db = detector_level_db(audio, sr, cfg["detector_mode"], cfg["sidechain_highpass_hz"])
    original_threshold_db = cfg["threshold_db"]

    # Auto threshold: previous modules may reduce level below static threshold.
    # For a research compressor, threshold should still bite gently unless safety scale is extremely low.
    if cfg.get("auto_threshold", True):
        p90 = float(np.percentile(level_db, 90))
        p95 = float(np.percentile(level_db, 95))
        adaptive_threshold = p90 - cfg.get("min_gain_reduction_db", 0.35)
        # Never make threshold higher than user threshold; only lower it to catch actual program level.
        cfg["threshold_db"] = round(min(cfg["threshold_db"], adaptive_threshold), 4)
        cfg["auto_threshold_reference"] = {
            "original_threshold_db": round(original_threshold_db, 4),
            "detector_p90_db": round(p90, 4),
            "detector_p95_db": round(p95, 4),
            "adapted_threshold_db": cfg["threshold_db"],
        }

    raw_gr = soft_knee_gain_reduction(level_db, cfg["threshold_db"], cfg["ratio"], cfg["knee_db"])
    env_gr = smooth_envelope(raw_gr, sr, cfg["attack_ms"], cfg["release_ms"])

    gain = undb20(-env_gr)[:, None]
    compressed = audio * gain

    auto_makeup_db = 0.0
    if cfg["auto_makeup"]:
        # Conservative makeup based on mean GR, not max GR.
        auto_makeup_db = min(3.0, float(np.mean(env_gr)) * 0.6)
    total_makeup = cfg["makeup_gain_db"] + auto_makeup_db
    if abs(total_makeup) > 0.0001:
        compressed = compressed * undb20(total_makeup)

    y = audio * (1.0 - cfg["mix"]) + compressed * cfg["mix"]
    y = normalize_peak(sanitize(y), -1.0)

    after_dynamics = analyze_dynamics(y, sr)
    after_loudness = analyze_loudness(y, sr)

    report = {
        "task": "Task 026 - Compressor",
        "status": "success",
        "config": cfg,
        "auto_makeup_db": round(auto_makeup_db, 6),
        "total_makeup_db": round(total_makeup, 6),
        "gain_reduction": {
            "max_gain_reduction_db": round(float(np.max(env_gr)) if len(env_gr) else 0.0, 6),
            "mean_gain_reduction_db": round(float(np.mean(env_gr)) if len(env_gr) else 0.0, 6),
            "p95_gain_reduction_db": round(float(np.percentile(env_gr, 95)) if len(env_gr) else 0.0, 6),
            "active_ratio": round(float(np.mean(env_gr > 0.05)) if len(env_gr) else 0.0, 6),
        },
        "before_dynamics": before_dynamics,
        "after_dynamics": after_dynamics,
        "before_loudness": before_loudness,
        "after_loudness": after_loudness,
        "dynamic_range_before": before_dynamics.get("dynamic_range_approx"),
        "dynamic_range_after": after_dynamics.get("dynamic_range_approx"),
        "crest_factor_before": before_dynamics.get("crest_factor_db"),
        "crest_factor_after": after_dynamics.get("crest_factor_db"),
        "punch_score_before": before_dynamics.get("punch_score"),
        "punch_score_after": after_dynamics.get("punch_score"),
        "lufs_before": before_loudness.get("integrated_lufs"),
        "lufs_after": after_loudness.get("integrated_lufs"),
    }
    return sanitize(y), report



