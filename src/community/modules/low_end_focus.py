from __future__ import annotations
import numpy as np

from .dsp_utils import sanitize, bandpass, peaking_eq, db20, undb20, normalize_peak
from community.analysis.spectrum import analyze_spectrum

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def low_band(audio, sr):
    return bandpass(audio, sr, 40, 140)

def sustain_band(audio, sr):
    return bandpass(audio, sr, 80, 220)

def low_band_metrics(audio, sr):
    lb = low_band(audio, sr)
    mono = np.mean(lb, axis=1)
    peak = float(np.max(np.abs(mono)) + 1e-12)
    rms = float(np.sqrt(np.mean(mono * mono)) + 1e-12)
    crest = float(db20(peak / rms))
    # transient proxy
    diff = np.abs(np.diff(mono, prepend=mono[0]))
    threshold = np.percentile(diff, 92) if len(diff) else 0.0
    transient_density = float(np.mean(diff >= threshold)) if threshold > 0 else 0.0
    punch_score = float(np.clip((crest / 18.0) * 0.65 + transient_density * 5.0 * 0.35, 0, 1))
    spectrum = analyze_spectrum(audio, sr)
    return {
        "low_band_peak_db": round(float(db20(peak)), 6),
        "low_band_rms_db": round(float(db20(rms)), 6),
        "low_band_crest_factor_db": round(crest, 6),
        "transient_density": round(transient_density, 6),
        "punch_score": round(punch_score, 6),
        "low_end_index": spectrum.get("low_end_index"),
        "mud_index": spectrum.get("mud_index"),
    }

def smooth_envelope(raw, sr, attack_ms=8.0, release_ms=120.0):
    attack = np.exp(-1.0 / max(1.0, (attack_ms/1000.0)*sr))
    release = np.exp(-1.0 / max(1.0, (release_ms/1000.0)*sr))
    out = np.zeros_like(raw)
    prev = 0.0
    for i, x in enumerate(raw):
        coeff = attack if x > prev else release
        prev = coeff * prev + (1 - coeff) * x
        out[i] = prev
    return out

def transient_mask(audio, sr):
    lb = low_band(audio, sr)
    mono = np.mean(lb, axis=1)
    diff = np.abs(np.diff(mono, prepend=mono[0]))
    threshold = np.percentile(diff, 92) if len(diff) else 0.0
    raw = (diff >= threshold).astype(float) if threshold > 0 else np.zeros_like(diff)
    # short smooth pulse
    win = max(8, int(sr * 0.007))
    kernel = np.hanning(win)
    kernel = kernel / (np.sum(kernel) + 1e-12)
    env = np.convolve(raw, kernel, mode="same")
    env = env / (np.max(env) + 1e-12)
    return np.clip(env, 0, 1), threshold

def sustain_envelope(audio, sr):
    sb = sustain_band(audio, sr)
    mono = np.mean(sb, axis=1)
    power = np.sqrt(np.maximum(mono * mono, 1e-12))
    env = smooth_envelope(power, sr, attack_ms=35, release_ms=260)
    env = env / (np.percentile(env, 95) + 1e-12)
    return np.clip(env, 0, 2)

def process_punchy(audio, sr, amount, sustain_tame):
    lb = low_band(audio, sr)
    rest = audio - lb
    tmask, threshold = transient_mask(audio, sr)
    sustain = sustain_envelope(audio, sr)

    transient_gain = 1.0 + tmask[:, None] * amount * 0.35
    sustain_cut = undb20(-sustain_tame * np.clip(sustain - 0.55, 0, 1.2) * 2.0)[:, None]
    focused_low = lb * transient_gain * sustain_cut

    y = rest + focused_low
    return y, {
        "mode": "punchy",
        "amount": amount,
        "sustain_tame": sustain_tame,
        "transient_threshold": float(threshold),
        "transient_env_mean": float(np.mean(tmask)),
        "sustain_env_mean": float(np.mean(sustain)),
        "max_transient_gain_db": float(db20(np.max(transient_gain))),
        "max_sustain_cut_db": float(db20(np.min(sustain_cut))),
    }

def process_smooth(audio, sr, amount, sustain_tame):
    lb = low_band(audio, sr)
    rest = audio - lb
    sustain = sustain_envelope(audio, sr)
    # Smooth = less pokey transient, cleaner sustain/boom
    tmask, threshold = transient_mask(audio, sr)
    transient_soften = undb20(-amount * tmask * 1.6)[:, None]
    sustain_balance = undb20(-sustain_tame * np.clip(sustain - 0.75, 0, 1.2) * 1.4)[:, None]
    focused_low = lb * transient_soften * sustain_balance
    # compensate gentle low body
    focused_low = peaking_eq(focused_low, sr, 85, amount * 0.6, q=0.7)
    y = rest + focused_low
    return y, {
        "mode": "smooth",
        "amount": amount,
        "sustain_tame": sustain_tame,
        "transient_threshold": float(threshold),
        "transient_env_mean": float(np.mean(tmask)),
        "sustain_env_mean": float(np.mean(sustain)),
        "max_transient_soften_db": float(db20(np.min(transient_soften))),
        "max_sustain_cut_db": float(db20(np.min(sustain_balance))),
    }

def infer_mode_and_amount(params, full_analysis):
    mode = str(params.get("mode", "auto")).lower()
    amount = params.get("amount", params.get("contrast_amount", 0.45))
    amount = float(amount)
    if amount > 1:
        amount /= 100.0
    amount = clamp(amount, 0, 1)
    sustain_tame = float(params.get("sustain_tame", 0.35))
    sustain_tame = clamp(sustain_tame, 0, 1)

    if mode == "auto":
        dynamics = full_analysis.get("dynamics", {}) if full_analysis else {}
        spectrum = full_analysis.get("spectrum", {}) if full_analysis else {}
        punch_score = float(dynamics.get("punch_score", 0.5))
        mud = float(spectrum.get("mud_index", 0.2))
        if punch_score < 0.45:
            mode = "punchy"
        elif mud > 0.28:
            mode = "smooth"
        else:
            mode = "punchy"
    if mode not in {"punchy", "smooth"}:
        mode = "punchy"
    return mode, amount, sustain_tame

def process_low_end_focus_advanced(audio, sr, params: dict, full_analysis: dict | None = None):
    before_metrics = low_band_metrics(audio, sr)
    mode, amount, sustain_tame = infer_mode_and_amount(params, full_analysis)

    if mode == "smooth":
        y, mode_report = process_smooth(audio, sr, amount, sustain_tame)
    else:
        y, mode_report = process_punchy(audio, sr, amount, sustain_tame)

    y = normalize_peak(sanitize(y), -1.0)
    after_metrics = low_band_metrics(y, sr)

    report = {
        "task": "Task 024 - Low End Focus",
        "status": "success",
        "mode": mode,
        "amount": round(amount, 6),
        "sustain_tame": round(sustain_tame, 6),
        "before_metrics": before_metrics,
        "after_metrics": after_metrics,
        "mode_report": mode_report,
        "low_band_crest_factor_before": before_metrics["low_band_crest_factor_db"],
        "low_band_crest_factor_after": after_metrics["low_band_crest_factor_db"],
        "punch_score_before": before_metrics["punch_score"],
        "punch_score_after": after_metrics["punch_score"],
        "crest_factor_delta": round(after_metrics["low_band_crest_factor_db"] - before_metrics["low_band_crest_factor_db"], 6),
        "punch_score_delta": round(after_metrics["punch_score"] - before_metrics["punch_score"], 6),
    }
    return sanitize(y), report



