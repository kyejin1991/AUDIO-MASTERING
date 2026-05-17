from __future__ import annotations
import numpy as np

from .dsp_utils import sanitize, db20, undb20, normalize_peak
from community.analysis.dynamics import analyze_dynamics
from community.analysis.basic_audio import analyze_basic_audio
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

def moving_average(x: np.ndarray, n: int):
    n = max(3, int(n))
    kernel = np.ones(n) / n
    return np.convolve(x, kernel, mode="same")

def infer_config(params: dict, full_analysis: dict | None):
    dynamics = full_analysis.get("dynamics", {}) if full_analysis else {}
    flags = full_analysis.get("diagnosis_flags", {}) if full_analysis else {}
    over = float(dynamics.get("overcompression_score", 0.0))
    needs = bool(dynamics.get("needs_unlimiter", False) or flags.get("needs_unlimiter", False))

    amount = params.get("amount", 0.35 + max(0.0, over - 0.45) * 0.9)
    amount = float(amount)
    if amount > 1.0:
        amount /= 100.0
    amount = clamp(amount, 0.0, 1.0)

    sensitivity = float(params.get("sensitivity", 0.65))
    recovery_depth = float(params.get("recovery_depth", 0.45 + max(0.0, over - 0.45) * 0.5))
    transient_boost_db = float(params.get("transient_boost_db", 1.2 + max(0.0, over - 0.45) * 3.0))
    expansion_ratio = float(params.get("expansion_ratio", 1.08 + max(0.0, over - 0.45) * 0.22))

    # If not overcompressed, keep subtle.
    if not needs and over < 0.45:
        amount *= 0.35
        recovery_depth *= 0.35
        transient_boost_db *= 0.45
        expansion_ratio = min(expansion_ratio, 1.08)

    return {
        "amount": round(clamp(amount, 0.0, 1.0), 6),
        "sensitivity": round(clamp(sensitivity, 0.05, 1.0), 6),
        "recovery_depth": round(clamp(recovery_depth, 0.0, 1.0), 6),
        "transient_boost_db": round(clamp(transient_boost_db, 0.0, 6.0), 6),
        "expansion_ratio": round(clamp(expansion_ratio, 1.0, 1.6), 6),
        "attack_ms": round(clamp(params.get("attack_ms", 2.5), 0.1, 60.0), 6),
        "release_ms": round(clamp(params.get("release_ms", 85.0), 5.0, 1000.0), 6),
        "ceiling_db": round(clamp(params.get("ceiling_db", -1.0), -6.0, -0.1), 6),
        "overcompression_score": round(over, 6),
        "needs_unlimiter": needs,
    }

def flattened_peak_mask(audio: np.ndarray, sr: int, sensitivity: float):
    mono = np.mean(audio, axis=1)
    absx = np.abs(mono)
    peak_ref = np.percentile(absx, 98.5) + 1e-12
    near_peak = absx > peak_ref * (0.82 - sensitivity * 0.12)
    slope = np.abs(np.diff(mono, prepend=mono[0]))
    slope_ref = np.percentile(slope, 80) + 1e-12
    flat = slope < slope_ref * (0.45 + (1.0-sensitivity)*0.35)
    raw = (near_peak & flat).astype(float)
    env = smooth_envelope(raw, sr, 1.0, 45.0)
    env = env / (np.max(env) + 1e-12)
    return np.clip(env, 0, 1), {
        "peak_ref": float(peak_ref),
        "slope_ref": float(slope_ref),
        "raw_flattened_ratio": float(np.mean(raw)),
    }

def transient_edge_mask(audio: np.ndarray, sr: int, sensitivity: float):
    mono = np.mean(audio, axis=1)
    slope = np.abs(np.diff(mono, prepend=mono[0]))
    # Detect attack edges but avoid constant clipped plateaus.
    threshold = np.percentile(slope, 90 - sensitivity * 18)
    raw = (slope >= threshold).astype(float)
    env = smooth_envelope(raw, sr, 0.5, 18.0)
    env = env / (np.max(env) + 1e-12)
    return np.clip(env, 0, 1), {
        "edge_threshold": float(threshold),
        "raw_edge_ratio": float(np.mean(raw)),
    }

def micro_expansion_gain(audio: np.ndarray, sr: int, ratio: float, amount: float):
    mono = np.mean(audio, axis=1)
    absx = np.abs(mono) + 1e-12
    short = moving_average(absx, max(8, int(sr * 0.006)))
    long = moving_average(absx, max(32, int(sr * 0.080))) + 1e-12
    contrast = short / long
    contrast = np.clip(contrast, 0, 3.0)
    # Expand events above local average.
    raw_gain_db = np.maximum(contrast - 1.0, 0.0) * (ratio - 1.0) * 8.0 * amount
    raw_gain_db = np.clip(raw_gain_db, 0.0, 3.0)
    gain_db = smooth_envelope(raw_gain_db, sr, 1.5, 70.0)
    return undb20(gain_db), {
        "contrast_mean": float(np.mean(contrast)),
        "contrast_p95": float(np.percentile(contrast, 95)),
        "max_expansion_gain_db": float(np.max(gain_db)),
        "mean_expansion_gain_db": float(np.mean(gain_db)),
    }

def attack_edge_restore(audio: np.ndarray, sr: int, transient_mask: np.ndarray, boost_db: float, amount: float):
    # High-pass-ish attack residue via short prediction error.
    mono = np.mean(audio, axis=1)
    smooth = moving_average(mono, max(5, int(sr * 0.002)))
    edge = mono - smooth
    edge_stereo = np.repeat(edge[:, None], 2, axis=1)
    gain = (undb20(boost_db) - 1.0) * amount
    return audio + edge_stereo * transient_mask[:, None] * gain

def sustain_deflatten(audio: np.ndarray, sr: int, flat_mask: np.ndarray, depth: float):
    # Subtle amplitude contouring on flattened regions: reduce sustained plateaus, keep attacks.
    mono = np.mean(audio, axis=1)
    env = moving_average(np.abs(mono), max(16, int(sr * 0.025)))
    env_norm = env / (np.percentile(env, 95) + 1e-12)
    plateau = np.clip(env_norm - 0.75, 0, 1.0)
    cut_db = -plateau * flat_mask * depth * 1.4
    gain = undb20(cut_db)
    return audio * gain[:, None], {
        "max_sustain_cut_db": float(np.min(cut_db)) if len(cut_db) else 0.0,
        "mean_sustain_cut_db": float(np.mean(cut_db)) if len(cut_db) else 0.0,
    }

def process_unlimiter_advanced(audio: np.ndarray, sr: int, params: dict, full_analysis: dict | None = None):
    cfg = infer_config(params, full_analysis)
    before_dyn = analyze_dynamics(audio, sr)
    before_basic = analyze_basic_audio(audio, sr)
    before_loud = analyze_loudness(audio, sr)

    y = sanitize(audio.copy())

    flat_mask, flat_report = flattened_peak_mask(y, sr, cfg["sensitivity"])
    edge_mask, edge_report = transient_edge_mask(y, sr, cfg["sensitivity"])
    recovery_mask = np.clip(flat_mask * 0.65 + edge_mask * 0.85, 0, 1)
    recovery_mask = smooth_envelope(recovery_mask, sr, cfg["attack_ms"], cfg["release_ms"])
    recovery_mask = np.clip(recovery_mask / (np.max(recovery_mask) + 1e-12), 0, 1)

    exp_gain, exp_report = micro_expansion_gain(y, sr, cfg["expansion_ratio"], cfg["amount"])
    y = y * (1.0 + (exp_gain[:, None] - 1.0) * recovery_mask[:, None] * cfg["recovery_depth"])

    y = attack_edge_restore(y, sr, edge_mask, cfg["transient_boost_db"], cfg["amount"])
    y, sustain_report = sustain_deflatten(y, sr, flat_mask, cfg["recovery_depth"])

    # Safety normalize
    y = normalize_peak(sanitize(y), cfg["ceiling_db"])

    after_dyn = analyze_dynamics(y, sr)
    after_basic = analyze_basic_audio(y, sr)
    after_loud = analyze_loudness(y, sr)

    # Safety rollback/blend if clipping appears or crest explodes too much.
    rollback_used = False
    rollback_reason = None
    crest_before = before_dyn.get("crest_factor_db", 0.0)
    crest_after = after_dyn.get("crest_factor_db", 0.0)
    if after_basic.get("clipping_samples", 0) > 0 or crest_after > crest_before + 6.0:
        rollback_used = True
        rollback_reason = "clipping_or_over_recovery"
        y = audio * 0.55 + y * 0.45
        y = normalize_peak(sanitize(y), cfg["ceiling_db"])
        after_dyn = analyze_dynamics(y, sr)
        after_basic = analyze_basic_audio(y, sr)
        after_loud = analyze_loudness(y, sr)

    report = {
        "task": "Task 030 - Unlimiter",
        "status": "success",
        "config": cfg,
        "before_dynamics": before_dyn,
        "after_dynamics": after_dyn,
        "before_basic": before_basic,
        "after_basic": after_basic,
        "before_loudness": before_loud,
        "after_loudness": after_loud,
        "flattened_peak_report": flat_report,
        "transient_edge_report": edge_report,
        "micro_expansion_report": exp_report,
        "sustain_deflatten_report": sustain_report,
        "mask_stats": {
            "flat_mask_mean": round(float(np.mean(flat_mask)), 8),
            "edge_mask_mean": round(float(np.mean(edge_mask)), 8),
            "recovery_mask_mean": round(float(np.mean(recovery_mask)), 8),
            "recovery_mask_p95": round(float(np.percentile(recovery_mask, 95)), 8),
        },
        "crest_factor_before": before_dyn.get("crest_factor_db"),
        "crest_factor_after": after_dyn.get("crest_factor_db"),
        "crest_factor_delta": round(after_dyn.get("crest_factor_db", 0.0) - before_dyn.get("crest_factor_db", 0.0), 6),
        "dynamic_range_before": before_dyn.get("dynamic_range_approx"),
        "dynamic_range_after": after_dyn.get("dynamic_range_approx"),
        "punch_score_before": before_dyn.get("punch_score"),
        "punch_score_after": after_dyn.get("punch_score"),
        "clipping_samples_after": after_basic.get("clipping_samples", 0),
        "rollback_used": rollback_used,
        "rollback_reason": rollback_reason,
        "recovery_applied": bool(cfg["amount"] > 0 and np.mean(recovery_mask) > 0),
    }
    return sanitize(y), report



