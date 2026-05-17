from __future__ import annotations
import numpy as np
from scipy import signal
from .dsp_utils import (
    sanitize, clip_audio, db20, undb20, highpass, lowpass, bandpass,
    peaking_eq, highshelf, lowshelf, mid_side, from_mid_side,
    static_compress, soft_limiter, normalize_peak, apply_gain_db
)
from community.analysis.loudness import integrated_lufs, true_peak_dbtp
from community.modules.equalizer import process_equalizer_advanced
from community.modules.dynamic_eq import process_dynamic_eq_advanced
from community.modules.stabilizer import process_stabilizer_advanced
from community.modules.bass_control import process_bass_control_advanced
from community.modules.low_end_focus import process_low_end_focus_advanced
from community.modules.multiband_compressor import process_multiband_compressor_advanced
from community.modules.compressor import process_compressor_advanced
from community.modules.exciter import process_exciter_advanced
from community.modules.imager import process_imager_advanced
from community.modules.maximizer import process_maximizer_advanced
from community.modules.unlimiter import process_unlimiter_advanced
from community.modules.audio_restore import process_audio_restore_advanced

def process_trim_silence(audio, sr, params):
    threshold_db = float(params.get("threshold_db", -60))
    min_silence_ms = int(params.get("min_silence_ms", 500))
    frame = max(256, int(sr * 0.02))
    x = np.mean(audio, axis=1)
    rms = []
    for start in range(0, len(x), frame):
        seg = x[start:start+frame]
        if len(seg) == 0: 
            continue
        rms.append(float(db20(np.sqrt(np.mean(seg*seg)) + 1e-12)))
    if not rms:
        return audio, {"trimmed_start_samples": 0, "trimmed_end_samples": 0}
    active = [i for i, v in enumerate(rms) if v > threshold_db]
    if not active:
        return audio, {"trimmed_start_samples": 0, "trimmed_end_samples": 0, "warning": "all_silence_detected"}
    start_frame = max(0, active[0] - int((min_silence_ms/1000) / 0.02))
    end_frame = min(len(rms), active[-1] + 1 + int((min_silence_ms/1000) / 0.02))
    start = start_frame * frame
    end = min(len(audio), end_frame * frame)
    return audio[start:end], {"trimmed_start_samples": int(start), "trimmed_end_samples": int(len(audio)-end)}

def process_dc_offset_remove(audio, sr, params):
    before = [float(np.mean(audio[:,0])), float(np.mean(audio[:,1]))]
    y = highpass(audio, sr, 10, order=1)
    after = [float(np.mean(y[:,0])), float(np.mean(y[:,1]))]
    return sanitize(y), {"dc_before": before, "dc_after": after}

def process_audio_restore(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_audio_restore_advanced(audio, sr, params, full_analysis=full_analysis)

def process_unlimiter(audio, sr, params):
    amount = float(params.get("amount", 0.35))
    transient_boost_db = float(params.get("transient_boost_db", 1.0))
    if amount <= 0:
        return audio, {"amount": 0, "transient_boost_db": 0}
    x = np.mean(audio, axis=1)
    diff = np.abs(np.diff(x, prepend=x[0]))
    threshold = np.percentile(diff, 92)
    mask = (diff >= threshold).astype(float)
    # Smooth transient mask
    win = max(8, int(sr * 0.008))
    kernel = np.hanning(win)
    kernel = kernel / (np.sum(kernel) + 1e-12)
    env = np.convolve(mask, kernel, mode="same")
    env = np.clip(env / (np.max(env) + 1e-12), 0, 1)
    gain = 1.0 + (undb20(transient_boost_db) - 1.0) * env[:, None] * amount
    y = audio * gain
    # micro expansion away from zero
    y = np.sign(y) * (np.abs(y) ** (1.0 - 0.08 * amount))
    y = normalize_peak(y, -1.0)
    return sanitize(y), {"amount": amount, "transient_boost_db": transient_boost_db, "transient_mask_mean": float(np.mean(env))}


def process_equalizer(audio, sr, params):
    # Task 020 advanced equalizer. full_analysis can be injected into params by render engine.
    full_analysis = params.get("_full_analysis")
    return process_equalizer_advanced(audio, sr, params, full_analysis=full_analysis)


def process_stabilizer(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_stabilizer_advanced(audio, sr, params, full_analysis=full_analysis)


def process_bass_control(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_bass_control_advanced(audio, sr, params, full_analysis=full_analysis)

def process_low_end_focus(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_low_end_focus_advanced(audio, sr, params, full_analysis=full_analysis)

def _dynamic_band_reduce(audio, sr, freq, threshold_db, ratio, range_db):
    low = max(20, freq / 1.6)
    high = min(sr/2 - 100, freq * 1.6)
    band = bandpass(audio, sr, low, high)
    rest = audio - band
    mag = np.sqrt(np.mean(band**2, axis=1)) + 1e-12
    level = db20(mag)
    over = np.maximum(level - threshold_db, 0)
    gr = over * (1 - 1/max(ratio, 1.0))
    gr = np.minimum(gr, abs(range_db))
    gain = undb20(-gr)[:, None]
    return rest + band * gain, float(np.max(gr)) if len(gr) else 0.0


def process_dynamic_eq(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_dynamic_eq_advanced(audio, sr, params, full_analysis=full_analysis)

def process_deesser(audio, sr, params):
    lo, hi = params.get("freq_range_hz", [5500, 9000])
    threshold = float(params.get("threshold_db", -28))
    max_red = float(params.get("max_reduction_db", 3))
    band = bandpass(audio, sr, lo, hi)
    rest = audio - band
    mag = np.sqrt(np.mean(band**2, axis=1)) + 1e-12
    level = db20(mag)
    over = np.maximum(level - threshold, 0)
    gr = np.minimum(over * 0.5, max_red)
    y = rest + band * undb20(-gr)[:, None]
    return sanitize(y), {"freq_range_hz": [lo, hi], "max_gain_reduction_db": float(np.max(gr)) if len(gr) else 0.0}


def process_compressor(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_compressor_advanced(audio, sr, params, full_analysis=full_analysis)

def process_multiband_compressor(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_multiband_compressor_advanced(audio, sr, params, full_analysis=full_analysis)


def process_exciter(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_exciter_advanced(audio, sr, params, full_analysis=full_analysis)


def process_imager(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_imager_advanced(audio, sr, params, full_analysis=full_analysis)


def process_unlimiter(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_unlimiter_advanced(audio, sr, params, full_analysis=full_analysis)

def process_maximizer(audio, sr, params):
    full_analysis = params.get("_full_analysis")
    return process_maximizer_advanced(audio, sr, params, full_analysis=full_analysis)

PROCESSORS = {
    "TrimSilence": process_trim_silence,
    "DCOffsetRemove": process_dc_offset_remove,
    "AudioRestore": process_audio_restore,
    "Unlimiter": process_unlimiter,
    "Equalizer": process_equalizer,
    "Stabilizer": process_stabilizer,
    "BassControl": process_bass_control,
    "LowEndFocus": process_low_end_focus,
    "DynamicEQ": process_dynamic_eq,
    "DeEsser": process_deesser,
    "Compressor": process_compressor,
    "MultibandCompressor": process_multiband_compressor,
    "Exciter": process_exciter,
    "Imager": process_imager,
    "Maximizer": process_maximizer,
}




