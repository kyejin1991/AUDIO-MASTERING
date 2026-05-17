from __future__ import annotations
import numpy as np

from .dsp_utils import sanitize, lowpass, bandpass, lowshelf, peaking_eq, mid_side, from_mid_side, db20, undb20, normalize_peak
from community.analysis.spectrum import analyze_spectrum
from community.analysis.stereo import analyze_stereo

def clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))

def low_band_meter(audio, sr):
    spectrum = analyze_spectrum(audio, sr)
    stereo = analyze_stereo(audio, sr)
    sub = spectrum["bands"].get("sub_20_60", 0.0)
    bass = spectrum["bands"].get("bass_60_150", 0.0)
    low_mid = spectrum["bands"].get("low_mid_150_500", 0.0)
    return {
        "sub_20_60": sub,
        "bass_60_150": bass,
        "low_mid_150_500": low_mid,
        "low_end_index": spectrum.get("low_end_index", sub + bass),
        "mud_index": spectrum.get("mud_index", low_mid),
        "low_end_stereo_leakage": stereo.get("low_end_stereo_leakage", 0.0),
        "stereo_width": stereo.get("stereo_width", 0.0),
        "phase_correlation": stereo.get("phase_correlation", 1.0),
    }

def infer_controls(params: dict, full_analysis: dict | None):
    spectrum = full_analysis.get("spectrum", {}) if full_analysis else {}
    stereo = full_analysis.get("stereo", {}) if full_analysis else {}
    flags = full_analysis.get("diagnosis_flags", {}) if full_analysis else {}

    low_end = float(spectrum.get("low_end_index", 0.25))
    mud = float(spectrum.get("mud_index", 0.20))
    leakage = float(stereo.get("low_end_stereo_leakage", 0.0))

    sub_gain = float(params.get("sub_gain_db", 0.0))
    bass_gain = float(params.get("bass_gain_db", 0.0))
    low_mono_hz = float(params.get("low_mono_hz", 120))
    punch_amount = float(params.get("punch_amount", 0.25))

    # Auto refine
    if low_end > 0.55 or flags.get("too_much_low_end", False):
        sub_gain = min(sub_gain, -1.8)
        bass_gain = min(bass_gain, -1.2)
    elif low_end < 0.14 or flags.get("thin_low_end", False):
        sub_gain = max(sub_gain, 1.2)
        bass_gain = max(bass_gain, 0.8)
        punch_amount = max(punch_amount, 0.35)

    boom_cut_db = 0.0
    if mud > 0.28 or flags.get("muddy_low_mid", False):
        boom_cut_db = -1.0 - min(3.0, max(0.0, mud - 0.22) * 12.0)

    if leakage > 0.15 or flags.get("wide_low_end", False):
        low_mono_hz = max(low_mono_hz, 150)

    harmonic_amount = float(params.get("harmonic_amount", 0.0))
    if low_end < 0.16:
        harmonic_amount = max(harmonic_amount, 0.18)

    return {
        "sub_gain_db": clamp(sub_gain, -6.0, 6.0),
        "bass_gain_db": clamp(bass_gain, -6.0, 6.0),
        "boom_cut_db": clamp(boom_cut_db, -6.0, 0.0),
        "low_mono_hz": clamp(low_mono_hz, 80.0, 220.0),
        "punch_amount": clamp(punch_amount, 0.0, 1.0),
        "harmonic_amount": clamp(harmonic_amount, 0.0, 0.5),
    }

def mono_lock_low_end(audio, sr, cutoff_hz):
    low = lowpass(audio, sr, cutoff_hz)
    high = audio - low
    low_mono = np.repeat(np.mean(low, axis=1, keepdims=True), 2, axis=1)
    return low_mono + high

def enhance_low_punch(audio, sr, amount):
    if amount <= 0:
        return audio, {"punch_events": 0, "amount": amount}
    low = bandpass(audio, sr, 40, 140)
    rest = audio - low
    x = np.mean(low, axis=1)
    transient = np.abs(np.diff(x, prepend=x[0]))
    threshold = np.percentile(transient, 92)
    mask = (transient >= threshold).astype(float)
    # smooth attack bump
    win = max(8, int(sr * 0.006))
    kernel = np.hanning(win)
    kernel = kernel / (np.sum(kernel) + 1e-12)
    env = np.convolve(mask, kernel, mode="same")
    env = np.clip(env / (np.max(env) + 1e-12), 0, 1)
    low_out = low * (1.0 + env[:, None] * amount * 0.22)
    return rest + low_out, {"punch_events": int(np.sum(mask)), "amount": amount, "env_mean": float(np.mean(env))}

def add_bass_harmonics(audio, sr, amount):
    if amount <= 0:
        return audio, {"amount": amount}
    bass = bandpass(audio, sr, 50, 140)
    harmonic = np.tanh(bass * 4.0)
    # move harmonic into audible low-mid area gently
    harmonic = peaking_eq(harmonic, sr, 180, 2.0, 0.8)
    y = audio + (harmonic - bass) * amount * 0.25
    return y, {"amount": amount}

def process_bass_control_advanced(audio, sr, params: dict, full_analysis: dict | None = None):
    before_meter = low_band_meter(audio, sr)
    controls = infer_controls(params, full_analysis)

    y = sanitize(audio.copy())

    # Sub/Bass tone control
    y = lowshelf(y, sr, 55, controls["sub_gain_db"])
    y = lowshelf(y, sr, 110, controls["bass_gain_db"])

    # Boom reduction around 160~220Hz
    if controls["boom_cut_db"] < -0.01:
        y = peaking_eq(y, sr, 180, controls["boom_cut_db"], q=0.9)

    # Low mono lock
    y = mono_lock_low_end(y, sr, controls["low_mono_hz"])

    # Punch transient enhancement
    y, punch_report = enhance_low_punch(y, sr, controls["punch_amount"])

    # Harmonic enhancer for weak bass
    y, harmonic_report = add_bass_harmonics(y, sr, controls["harmonic_amount"])

    y = normalize_peak(sanitize(y), -1.0)
    after_meter = low_band_meter(y, sr)

    report = {
        "task": "Task 023 - Bass Control",
        "status": "success",
        "controls": controls,
        "before_meter": before_meter,
        "after_meter": after_meter,
        "punch_report": punch_report,
        "harmonic_report": harmonic_report,
        "low_end_stereo_leakage_before": before_meter["low_end_stereo_leakage"],
        "low_end_stereo_leakage_after": after_meter["low_end_stereo_leakage"],
        "low_end_index_before": before_meter["low_end_index"],
        "low_end_index_after": after_meter["low_end_index"],
        "mud_index_before": before_meter["mud_index"],
        "mud_index_after": after_meter["mud_index"],
        "leakage_reduced": bool(after_meter["low_end_stereo_leakage"] <= before_meter["low_end_stereo_leakage"]),
    }
    return sanitize(y), report



