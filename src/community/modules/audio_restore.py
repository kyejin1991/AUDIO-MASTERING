from __future__ import annotations

import numpy as np
from scipy import signal

from .dsp_utils import sanitize, normalize_peak
from community.analysis.basic_audio import analyze_basic_audio
from community.analysis.dynamics import analyze_dynamics
from community.analysis.loudness import analyze_loudness
from community.analysis.spectrum import analyze_spectrum


def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def _stft(audio: np.ndarray, n_fft: int, hop_length: int):
    left = signal.stft(audio[:, 0], nperseg=n_fft, noverlap=n_fft - hop_length, boundary="zeros", padded=True)
    right = signal.stft(audio[:, 1], nperseg=n_fft, noverlap=n_fft - hop_length, boundary="zeros", padded=True)
    return left, right


def _istft(stft_left, stft_right, sr: int, n_fft: int, hop_length: int, length: int):
    _, l = signal.istft(stft_left, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=True)
    _, r = signal.istft(stft_right, fs=sr, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=True)
    out = np.stack([l[:length], r[:length]], axis=1)
    if len(out) < length:
        pad = np.zeros((length - len(out), 2), dtype=out.dtype)
        out = np.concatenate([out, pad], axis=0)
    return sanitize(out[:length])


def _frame_rms(audio: np.ndarray, sr: int, frame_sec: float = 0.05):
    mono = np.mean(audio, axis=1)
    frame = max(256, int(sr * frame_sec))
    hop = frame
    vals = []
    for start in range(0, max(1, len(mono) - frame + 1), hop):
        seg = mono[start:start + frame]
        if len(seg) < frame:
            break
        vals.append(np.sqrt(np.mean(seg ** 2) + 1e-12))
    return np.array(vals, dtype=float) if vals else np.array([np.sqrt(np.mean(mono ** 2) + 1e-12)], dtype=float)


def estimate_noise_floor_db(audio: np.ndarray, sr: int) -> float:
    rms = _frame_rms(audio, sr)
    floor = np.percentile(rms, 15)
    return float(20.0 * np.log10(max(floor, 1e-12)))


def infer_config(params: dict, full_analysis: dict | None):
    spectrum = (full_analysis or {}).get("spectrum", {}) if isinstance(full_analysis, dict) else {}
    basic = (full_analysis or {}).get("basic_audio", {}) if isinstance(full_analysis, dict) else {}
    flags = (full_analysis or {}).get("diagnosis_flags", {}) if isinstance(full_analysis, dict) else {}

    air_index = float(spectrum.get("air_index", 0.05) or 0.05)
    brightness = float(spectrum.get("brightness_index", 0.12) or 0.12)
    silence_ratio = float(basic.get("silence_ratio", 0.0) or 0.0)
    source_noise_floor = float(basic.get("noise_floor_dbfs", -70.0) or -70.0)

    restore_amount = float(params.get("amount", 0.35 + max(0.0, (0.055 - air_index) * 8.0)))
    restore_amount = clamp(restore_amount, 0.0, 1.0)
    noise_suppress = float(params.get("noise_suppress", 0.18 + max(0.0, silence_ratio - 0.04) * 1.8 + max(0.0, source_noise_floor + 58.0) * 0.01))
    noise_suppress = clamp(noise_suppress, 0.0, 0.95)
    high_synth = float(params.get("high_synth", 0.25 + max(0.0, (0.05 - air_index) * 10.0) + max(0.0, 0.10 - brightness) * 1.4))
    high_synth = clamp(high_synth, 0.0, 1.0)
    transient_recovery = float(params.get("transient_recovery", 0.18 + max(0.0, (0.11 - brightness) * 1.6)))
    transient_recovery = clamp(transient_recovery, 0.0, 1.0)
    enabled = bool(params.get("enabled", flags.get("needs_audio_restore", False) or air_index < 0.03))

    return {
        "enabled": enabled,
        "amount": round(restore_amount, 6),
        "noise_suppress": round(noise_suppress, 6),
        "high_synth": round(high_synth, 6),
        "transient_recovery": round(transient_recovery, 6),
        "synth_start_hz": int(params.get("synth_start_hz", 8000)),
        "n_fft": int(params.get("n_fft", 2048)),
        "hop_length": int(params.get("hop_length", 512)),
        "ceiling_db": round(clamp(params.get("ceiling_db", -1.0), -6.0, -0.1), 6),
    }


def spectral_noise_suppress(Zxx: np.ndarray, strength: float):
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    quiet_profile = np.percentile(mag, 18, axis=1, keepdims=True)
    threshold = quiet_profile * (0.96 + strength * 0.55)
    mask = np.clip((mag - threshold) / (mag + quiet_profile + 1e-12), 0.0, 1.0)
    mask = mask ** (0.8 + 0.35 * strength)
    mask = np.clip(0.22 + mask * (0.78 - 0.20 * strength), 0.0, 1.0)
    out = mag * mask * np.exp(1j * phase)
    return out, {
        "quiet_profile_mean": float(np.mean(quiet_profile)),
        "mask_mean": float(np.mean(mask)),
        "mask_p95": float(np.percentile(mask, 95)),
    }


def spectral_inpaint(Zxx: np.ndarray, freqs: np.ndarray, amount: float, start_hz: float):
    if amount <= 0:
        return Zxx, {"applied_bins": 0, "mean_gain_ratio": 1.0}
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)
    synth_mask = freqs >= start_hz
    applied = 0
    gain_ratios = []
    out_mag = mag.copy()
    for idx in np.where(synth_mask)[0]:
        source_idx = max(1, idx // 2)
        source_idx_2 = max(1, idx // 3)
        harmonic = (0.72 * mag[source_idx] + 0.28 * mag[source_idx_2]) * (0.55 + 0.45 * amount)
        existing = mag[idx]
        target = np.maximum(existing, harmonic)
        blended = existing * (1.0 - amount) + target * amount
        out_mag[idx] = blended
        applied += 1
        gain_ratios.append(float(np.mean(blended / (existing + 1e-12))))
    out = out_mag * np.exp(1j * phase)
    return out, {
        "applied_bins": applied,
        "mean_gain_ratio": float(np.mean(gain_ratios)) if gain_ratios else 1.0,
    }


def transient_recover(audio: np.ndarray, sr: int, amount: float):
    if amount <= 0:
        return audio, {"edge_mask_mean": 0.0, "boost_db_equiv": 0.0}
    mono = np.mean(audio, axis=1)
    nine = max(9, ((sr // 2000) * 2 + 1))
    smooth = signal.savgol_filter(mono, nine, 2, mode="interp") if len(mono) >= nine else mono
    edge = mono - smooth
    slope = np.abs(np.diff(mono, prepend=mono[0]))
    thresh = np.percentile(slope, 92)
    mask = (slope >= thresh).astype(float)
    win = max(16, int(sr * 0.008))
    kernel = np.hanning(win)
    kernel = kernel / (np.sum(kernel) + 1e-12)
    edge_env = np.convolve(mask, kernel, mode="same")
    edge_env = np.clip(edge_env / (np.max(edge_env) + 1e-12), 0.0, 1.0)
    boost = 0.22 * amount
    recovered = audio + np.repeat(edge[:, None], 2, axis=1) * edge_env[:, None] * boost
    return sanitize(recovered), {
        "edge_mask_mean": float(np.mean(edge_env)),
        "boost_db_equiv": float(20.0 * np.log10(max(1.0 + boost, 1e-12))),
    }


def process_audio_restore_advanced(audio: np.ndarray, sr: int, params: dict, full_analysis: dict | None = None):
    cfg = infer_config(params, full_analysis)
    before_basic = analyze_basic_audio(audio, sr)
    before_loudness = analyze_loudness(audio, sr)
    before_spectrum = analyze_spectrum(audio, sr)
    before_dynamics = analyze_dynamics(audio, sr)
    before_noise_floor = estimate_noise_floor_db(audio, sr)

    if not cfg["enabled"] or cfg["amount"] <= 0:
        report = {
            "task": "Task 002 - Audio Restore",
            "status": "bypassed",
            "config": cfg,
            "before_basic": before_basic,
            "after_basic": before_basic,
            "before_loudness": before_loudness,
            "after_loudness": before_loudness,
            "before_spectrum": before_spectrum,
            "after_spectrum": before_spectrum,
            "before_dynamics": before_dynamics,
            "after_dynamics": before_dynamics,
            "noise_floor_before_dbfs": round(before_noise_floor, 6),
            "noise_floor_after_dbfs": round(before_noise_floor, 6),
            "noise_floor_delta_db": 0.0,
        }
        return sanitize(audio.copy()), report

    n_fft = min(cfg["n_fft"], max(512, len(audio) // 2))
    hop_length = min(cfg["hop_length"], max(128, n_fft // 4))
    (_, _, Zl), (_, _, Zr) = _stft(audio, n_fft=n_fft, hop_length=hop_length)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    Zl, gate_l = spectral_noise_suppress(Zl, cfg["noise_suppress"] * cfg["amount"])
    Zr, gate_r = spectral_noise_suppress(Zr, cfg["noise_suppress"] * cfg["amount"])

    Zl, synth_l = spectral_inpaint(Zl, freqs, cfg["high_synth"] * cfg["amount"], cfg["synth_start_hz"])
    Zr, synth_r = spectral_inpaint(Zr, freqs, cfg["high_synth"] * cfg["amount"], cfg["synth_start_hz"])

    restored = _istft(Zl, Zr, sr=sr, n_fft=n_fft, hop_length=hop_length, length=len(audio))
    restored, transient_report = transient_recover(restored, sr, cfg["transient_recovery"] * cfg["amount"])
    restored = normalize_peak(restored, cfg["ceiling_db"])

    after_basic = analyze_basic_audio(restored, sr)
    after_loudness = analyze_loudness(restored, sr)
    after_spectrum = analyze_spectrum(restored, sr)
    after_dynamics = analyze_dynamics(restored, sr)
    after_noise_floor = estimate_noise_floor_db(restored, sr)

    report = {
        "task": "Task 002 - Audio Restore",
        "status": "success",
        "config": cfg,
        "before_basic": before_basic,
        "after_basic": after_basic,
        "before_loudness": before_loudness,
        "after_loudness": after_loudness,
        "before_spectrum": before_spectrum,
        "after_spectrum": after_spectrum,
        "before_dynamics": before_dynamics,
        "after_dynamics": after_dynamics,
        "noise_floor_before_dbfs": round(before_noise_floor, 6),
        "noise_floor_after_dbfs": round(after_noise_floor, 6),
        "noise_floor_delta_db": round(after_noise_floor - before_noise_floor, 6),
        "air_index_before": before_spectrum["air_index"],
        "air_index_after": after_spectrum["air_index"],
        "brightness_index_before": before_spectrum["brightness_index"],
        "brightness_index_after": after_spectrum["brightness_index"],
        "crest_factor_before": before_dynamics["crest_factor_db"],
        "crest_factor_after": after_dynamics["crest_factor_db"],
        "spectral_gate": {
            "left": gate_l,
            "right": gate_r,
        },
        "spectral_inpaint": {
            "left": synth_l,
            "right": synth_r,
            "start_hz": cfg["synth_start_hz"],
        },
        "transient_recovery_report": transient_report,
    }
    return sanitize(restored), report



