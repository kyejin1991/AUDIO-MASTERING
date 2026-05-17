from __future__ import annotations
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d

from .dsp_utils import sanitize, db20, undb20
from community.analysis.loudness import analyze_loudness, integrated_lufs, true_peak_dbtp
from community.analysis.basic_audio import analyze_basic_audio
from community.analysis.dynamics import analyze_dynamics

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def smooth_release(gain_target: np.ndarray, sr: int, release_ms: float | np.ndarray):
    release_arr = np.asarray(release_ms, dtype=float)
    if release_arr.ndim == 0:
        release_arr = np.full(len(gain_target), float(release_arr), dtype=float)
    elif len(release_arr) != len(gain_target):
        xp = np.linspace(0.0, 1.0, len(release_arr))
        x = np.linspace(0.0, 1.0, len(gain_target))
        release_arr = np.interp(x, xp, release_arr)
    release_arr = np.clip(release_arr, 5.0, 2500.0)
    out = np.ones_like(gain_target)
    prev = 1.0
    for i, g in enumerate(gain_target):
        # clamp down instantly; recover with release
        if g < prev:
            prev = g
        else:
            release = np.exp(-1.0 / max(1.0, (float(release_arr[i]) / 1000.0) * sr))
            prev = release * prev + (1.0 - release) * g
        out[i] = prev
    return out

def estimate_tempo_bpm(audio: np.ndarray, sr: int) -> float | None:
    mono = np.mean(audio, axis=1)
    if len(mono) < sr * 2:
        return None
    diff = np.abs(np.diff(mono, prepend=mono[0]))
    env_win = max(64, int(sr * 0.02))
    env = np.convolve(diff, np.ones(env_win, dtype=float) / env_win, mode="same")
    env = env - np.mean(env)
    if not np.any(np.abs(env) > 1e-12):
        return None
    min_bpm = 70.0
    max_bpm = 180.0
    min_lag = int(sr * 60.0 / max_bpm)
    max_lag = int(sr * 60.0 / min_bpm)
    if max_lag >= len(env):
        max_lag = len(env) - 1
    if min_lag >= max_lag:
        return None
    corr = signal.correlate(env, env, mode="full")
    corr = corr[len(corr) // 2:]
    search = corr[min_lag:max_lag]
    if search.size == 0:
        return None
    lag = min_lag + int(np.argmax(search))
    bpm = 60.0 * sr / max(lag, 1)
    # Normalize half/double tempo toward a stable club/pop range.
    while bpm < 80.0:
        bpm *= 2.0
    while bpm > 175.0:
        bpm /= 2.0
    if not np.isfinite(bpm):
        return None
    return float(np.clip(bpm, 70.0, 180.0))

def build_adaptive_release_profile(
    audio: np.ndarray,
    sr: int,
    base_release_ms: float,
    mode: str,
    full_analysis: dict | None = None,
):
    dynamics = (full_analysis or {}).get("dynamics", {}) if isinstance(full_analysis, dict) else {}
    transient_density = float(dynamics.get("transient_density", 0.0) or 0.0)
    overcompression_score = float(dynamics.get("overcompression_score", 0.0) or 0.0)
    tempo_bpm = estimate_tempo_bpm(audio, sr)
    beat_release_ms = None
    if tempo_bpm:
        beat_ms = 60000.0 / tempo_bpm
        beat_ratio = {"gentle": 0.48, "balanced": 0.34, "aggressive": 0.24}.get(mode, 0.34)
        beat_release_ms = beat_ms * beat_ratio
    target_release_ms = beat_release_ms if beat_release_ms is not None else base_release_ms
    blended_base = float(np.clip(base_release_ms * 0.55 + target_release_ms * 0.45, 18.0, 900.0))

    mono = np.mean(audio, axis=1)
    diff = np.abs(np.diff(mono, prepend=mono[0]))
    flux_win = max(64, int(sr * 0.015))
    flux = np.convolve(diff, np.ones(flux_win, dtype=float) / flux_win, mode="same")
    flux_norm = flux / (np.percentile(flux, 95) + 1e-12)
    flux_norm = np.clip(flux_norm, 0.0, 1.5)

    rms_win = max(256, int(sr * 0.05))
    power = mono * mono
    energy = np.sqrt(np.convolve(power, np.ones(rms_win, dtype=float) / rms_win, mode="same") + 1e-12)
    energy_norm = energy / (np.percentile(energy, 95) + 1e-12)
    energy_norm = np.clip(energy_norm, 0.0, 1.25)

    # Faster recovery around dense transients, slower on sustained energetic passages.
    transient_push = np.clip(0.55 + transient_density * 0.6, 0.55, 1.2)
    sustain_pull = np.clip(0.35 + overcompression_score * 0.55, 0.35, 0.95)
    release_profile = blended_base * (1.08 - transient_push * flux_norm + sustain_pull * np.maximum(energy_norm - 0.55, 0.0))
    release_profile = signal.savgol_filter(release_profile, max(9, (sr // 1000) | 1), 2, mode="interp") if len(release_profile) >= 9 else release_profile
    release_profile = np.clip(release_profile, 15.0, 1200.0)

    return release_profile.astype(float), {
        "enabled": True,
        "estimated_tempo_bpm": round(float(tempo_bpm), 4) if tempo_bpm is not None else None,
        "base_release_ms": round(float(base_release_ms), 4),
        "tempo_synced_release_ms": round(float(target_release_ms), 4) if beat_release_ms is not None else None,
        "release_min_ms": round(float(np.min(release_profile)), 4),
        "release_mean_ms": round(float(np.mean(release_profile)), 4),
        "release_max_ms": round(float(np.max(release_profile)), 4),
        "transient_density": round(transient_density, 6),
        "overcompression_score": round(overcompression_score, 6),
    }

def lookahead_limiter(
    audio: np.ndarray,
    sr: int,
    ceiling_db: float,
    lookahead_ms: float,
    release_ms: float | np.ndarray,
):
    ceiling = undb20(ceiling_db)
    lookahead = max(1, int(sr * lookahead_ms / 1000.0))
    padded = np.pad(audio, ((0, lookahead), (0,0)), mode="constant")
    absmax = np.max(np.abs(padded), axis=1)
    # future peak envelope
    kernel = np.ones(lookahead)
    # max filter via sliding maximum approximation using scipy maximum_filter1d if available
    future_peak = maximum_filter1d(absmax, size=lookahead, mode="nearest")
    future_peak = future_peak[:len(audio)]
    target_gain = np.minimum(1.0, ceiling / (future_peak + 1e-12))
    gain = smooth_release(target_gain, sr, release_ms)
    y = audio * gain[:, None]
    reduction_db = -db20(np.maximum(gain, 1e-12))
    return sanitize(y), reduction_db

def true_peak_normalize(audio: np.ndarray, sr: int, ceiling_db: float, oversample: int = 4):
    tp = true_peak_dbtp(audio, sr, oversample=oversample)
    if tp > ceiling_db:
        audio = audio * undb20(ceiling_db - tp - 0.02)
    return sanitize(audio)

def infer_config(params: dict, full_analysis: dict | None):
    target_lufs = float(params.get("target_lufs", -14.0))
    ceiling = float(params.get("ceiling_dbtp", -1.0))
    codec_safe = bool(params.get("codec_safe", True))
    if codec_safe:
        ceiling = min(ceiling, -1.05)
    mode = str(params.get("mode", "balanced")).lower()
    if mode not in {"gentle", "balanced", "aggressive"}:
        mode = "balanced"
    release_ms = float(params.get("release_ms", 80 if mode == "aggressive" else (160 if mode == "gentle" else 120)))
    lookahead_ms = float(params.get("lookahead_ms", 3.0 if mode == "aggressive" else 5.0))
    max_input_gain_db = float(params.get("max_input_gain_db", 18.0))
    max_iterations = int(params.get("max_iterations", 8))
    tolerance_lufs = float(params.get("tolerance_lufs", 0.35))
    oversampling = int(params.get("oversampling", 4))
    adaptive_release = bool(params.get("adaptive_release", True))
    return {
        "target_lufs": round(clamp(target_lufs, -24.0, -5.0), 4),
        "ceiling_dbtp": round(clamp(ceiling, -6.0, -0.1), 4),
        "codec_safe": codec_safe,
        "mode": mode,
        "release_ms": round(clamp(release_ms, 5, 2000), 4),
        "lookahead_ms": round(clamp(lookahead_ms, 0.5, 20), 4),
        "max_input_gain_db": round(clamp(max_input_gain_db, 0, 36), 4),
        "max_iterations": max(1, min(16, max_iterations)),
        "tolerance_lufs": round(clamp(tolerance_lufs, 0.05, 2.0), 4),
        "oversampling": oversampling if oversampling in {2,4,8} else 4,
        "true_peak": bool(params.get("true_peak", True)),
        "adaptive_release": adaptive_release,
    }

def process_maximizer_advanced(audio: np.ndarray, sr: int, params: dict, full_analysis: dict | None = None):
    cfg = infer_config(params, full_analysis)
    before_loudness = analyze_loudness(audio, sr)
    before_basic = analyze_basic_audio(audio, sr)
    before_dynamics = analyze_dynamics(audio, sr)
    adaptive_release_report = {
        "enabled": False,
        "estimated_tempo_bpm": None,
        "base_release_ms": cfg["release_ms"],
        "tempo_synced_release_ms": None,
        "release_min_ms": cfg["release_ms"],
        "release_mean_ms": cfg["release_ms"],
        "release_max_ms": cfg["release_ms"],
    }
    release_profile = cfg["release_ms"]
    if cfg["adaptive_release"]:
        release_profile, adaptive_release_report = build_adaptive_release_profile(
            audio,
            sr,
            cfg["release_ms"],
            cfg["mode"],
            full_analysis=full_analysis,
        )

    y = sanitize(audio.copy())
    history = []
    total_input_gain = 0.0
    all_reduction = []
    target_lufs_live = cfg["target_lufs"]

    # iterative loudness matching + limiter safety
    for i in range(cfg["max_iterations"]):
        cur_lufs = integrated_lufs(y, sr)
        diff = target_lufs_live - cur_lufs
        step_gain = float(np.clip(diff, -4.0, 4.0))
        # Do not exceed total input gain safety.
        if total_input_gain + step_gain > cfg["max_input_gain_db"]:
            step_gain = cfg["max_input_gain_db"] - total_input_gain
        y = y * undb20(step_gain)
        total_input_gain += step_gain

        pre_limit_tp = true_peak_dbtp(y, sr, oversample=cfg["oversampling"])
        y, reduction = lookahead_limiter(y, sr, cfg["ceiling_dbtp"], cfg["lookahead_ms"], release_profile)
        y = true_peak_normalize(y, sr, cfg["ceiling_dbtp"], oversample=cfg["oversampling"])
        all_reduction.append(reduction)

        post_lufs = integrated_lufs(y, sr)
        post_tp = true_peak_dbtp(y, sr, oversample=cfg["oversampling"])
        reduction_p95 = float(np.percentile(reduction, 95)) if len(reduction) else 0.0
        history.append({
            "iteration": i + 1,
            "lufs_before_gain": round(cur_lufs, 6),
            "target_lufs_live": round(target_lufs_live, 6),
            "gain_step_db": round(step_gain, 6),
            "total_input_gain_db": round(total_input_gain, 6),
            "pre_limit_true_peak_dbtp": round(pre_limit_tp, 6),
            "post_lufs": round(post_lufs, 6),
            "post_true_peak_dbtp": round(post_tp, 6),
            "max_limiter_reduction_db": round(float(np.max(reduction)) if len(reduction) else 0.0, 6),
            "mean_limiter_reduction_db": round(float(np.mean(reduction)) if len(reduction) else 0.0, 6),
            "p95_limiter_reduction_db": round(reduction_p95, 6),
        })
        if abs(target_lufs_live - post_lufs) <= cfg["tolerance_lufs"] and post_tp <= cfg["ceiling_dbtp"] + 0.02:
            break
        # If limiter is fighting too hard, stop pushing upward.
        if len(reduction) and reduction_p95 > 5.0 and diff > 0:
            target_lufs_live -= min(0.75, 0.18 + 0.08 * i)
            break
        if len(reduction) and reduction_p95 > 3.5 and diff > 0:
            target_lufs_live -= 0.2

    y = true_peak_normalize(y, sr, cfg["ceiling_dbtp"], oversample=cfg["oversampling"])
    after_loudness = analyze_loudness(y, sr)
    after_basic = analyze_basic_audio(y, sr)
    after_dynamics = analyze_dynamics(y, sr)

    if all_reduction:
        red = np.concatenate(all_reduction)
    else:
        red = np.zeros(len(y))

    clipping_samples = after_basic.get("clipping_samples", 0)
    ceiling_passed = after_loudness.get("true_peak_dbtp", 0.0) <= cfg["ceiling_dbtp"] + 0.05
    target_hit = abs(after_loudness.get("integrated_lufs", cfg["target_lufs"]) - cfg["target_lufs"]) <= 0.75

    report = {
        "task": "Task 029 - Maximizer",
        "status": "success",
        "config": cfg,
        "history": history,
        "adaptive_release": adaptive_release_report,
        "before_loudness": before_loudness,
        "after_loudness": after_loudness,
        "before_basic": before_basic,
        "after_basic": after_basic,
        "before_dynamics": before_dynamics,
        "after_dynamics": after_dynamics,
        "limiter_reduction": {
            "max_reduction_db": round(float(np.max(red)) if len(red) else 0.0, 6),
            "mean_reduction_db": round(float(np.mean(red)) if len(red) else 0.0, 6),
            "p95_reduction_db": round(float(np.percentile(red, 95)) if len(red) else 0.0, 6),
            "active_ratio": round(float(np.mean(red > 0.05)) if len(red) else 0.0, 6),
        },
        "target_lufs": cfg["target_lufs"],
        "effective_target_lufs": round(target_lufs_live, 6),
        "final_lufs": after_loudness.get("integrated_lufs"),
        "lufs_error": round(after_loudness.get("integrated_lufs") - cfg["target_lufs"], 6),
        "ceiling_dbtp": cfg["ceiling_dbtp"],
        "final_true_peak_dbtp": after_loudness.get("true_peak_dbtp"),
        "ceiling_passed": bool(ceiling_passed),
        "target_hit": bool(target_hit),
        "clipping_samples": clipping_samples,
        "clipping_passed": bool(clipping_samples == 0),
        "total_input_gain_db": round(total_input_gain, 6),
        "isp_oversampling_factor": cfg["oversampling"],
    }
    return sanitize(y), report



