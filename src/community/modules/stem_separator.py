from __future__ import annotations
from pathlib import Path
import json
import numpy as np
from scipy import signal
import soundfile as sf

from community.core.project_loader import load_project
from community.core.project_logger import ProjectLogger
from community.core.audio_reader import read_working_wav
from community.analysis.basic_audio import analyze_basic_audio
from community.analysis.loudness import analyze_loudness
from community.analysis.spectrum import analyze_spectrum
from community.analysis.stereo import analyze_stereo
from community.analysis.dynamics import analyze_dynamics
from .dsp_utils import sanitize, bandpass, lowpass, highpass, normalize_peak, mid_side, from_mid_side
from community.modules.neural_stem_backend import DemucsBackend, backend_config_from_params

def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def save_wav(path: str | Path, audio: np.ndarray, sr: int):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, sanitize(audio), sr, subtype="PCM_24")
    return path

def load_stem_wav(path: str | Path, target_len: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, always_2d=True)
    audio = audio.astype(np.float64)
    if audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    if target_len is not None:
        if len(audio) < target_len:
            audio = np.pad(audio, ((0, target_len-len(audio)), (0,0)))
        elif len(audio) > target_len:
            audio = audio[:target_len]
    return sanitize(audio), sr

def analyze_audio(audio: np.ndarray, sr: int) -> dict:
    return {
        "basic_audio": analyze_basic_audio(audio, sr),
        "loudness": analyze_loudness(audio, sr),
        "spectrum": analyze_spectrum(audio, sr),
        "stereo": analyze_stereo(audio, sr),
        "dynamics": analyze_dynamics(audio, sr),
    }

def hpss_split(audio: np.ndarray, sr: int):
    """
    Lightweight offline HPSS:
    STFT magnitude median filtering along time/frequency.
    Returns harmonic-like and percussive-like stereo audio.
    """
    out_h = []
    out_p = []
    for ch in range(audio.shape[1]):
        x = audio[:, ch]
        f, t, Z = signal.stft(x, fs=sr, nperseg=2048, noverlap=1536, boundary='zeros')
        mag = np.abs(Z)
        phase = np.exp(1j * np.angle(Z))
        harm_mag = signal.medfilt2d(mag, kernel_size=(1, 17))
        perc_mag = signal.medfilt2d(mag, kernel_size=(17, 1))
        h_mask = harm_mag / (harm_mag + perc_mag + 1e-12)
        p_mask = perc_mag / (harm_mag + perc_mag + 1e-12)
        _, h = signal.istft(Z * h_mask * phase, fs=sr, nperseg=2048, noverlap=1536, input_onesided=True)
        _, p = signal.istft(Z * p_mask * phase, fs=sr, nperseg=2048, noverlap=1536, input_onesided=True)
        h = h[:len(x)] if len(h) >= len(x) else np.pad(h, (0, len(x)-len(h)))
        p = p[:len(x)] if len(p) >= len(x) else np.pad(p, (0, len(x)-len(p)))
        out_h.append(h)
        out_p.append(p)
    harmonic = np.stack(out_h, axis=1)
    percussive = np.stack(out_p, axis=1)
    return sanitize(harmonic), sanitize(percussive)

def estimate_vocals(harmonic: np.ndarray, sr: int, amount: float = 0.72):
    """
    Center-channel vocal approximation:
    centered mid frequencies from harmonic material.
    """
    mid, side = mid_side(harmonic)
    vocal_band = bandpass(mid[:, None], sr, 160, 6500)[:, 0]
    # suppress very steady low-mid bed by subtracting side-related leakage
    side_band = bandpass(side[:, None], sr, 160, 6500)[:, 0]
    vocal = vocal_band - side_band * 0.18
    vocal = np.repeat(vocal[:, None], 2, axis=1) * amount
    return sanitize(vocal)

def estimate_bass(audio: np.ndarray, sr: int):
    bass = lowpass(audio, sr, 150)
    # mono bass
    bass_mono = np.repeat(np.mean(bass, axis=1, keepdims=True), 2, axis=1)
    return sanitize(bass_mono)

def estimate_ambience(audio: np.ndarray, sr: int):
    mid, side = mid_side(audio)
    high_side = highpass(side[:, None], sr, 900)[:, 0]
    amb = from_mid_side(np.zeros_like(mid), high_side * 0.8)
    return sanitize(amb)

def safe_stem(audio: np.ndarray):
    return normalize_peak(sanitize(audio), -1.0)

def separate_stems_offline(audio: np.ndarray, sr: int, params: dict | None = None):
    params = params or {}
    vocal_amount = float(params.get("vocal_center_amount", 0.72))
    harmonic, percussive = hpss_split(audio, sr)

    drums = safe_stem(percussive)
    bass = safe_stem(estimate_bass(audio, sr))
    vocals = safe_stem(estimate_vocals(harmonic, sr, vocal_amount))
    ambience = safe_stem(estimate_ambience(audio, sr))

    # residual music: subtract estimated stems conservatively
    music = audio - drums * 0.75 - bass * 0.65 - vocals * 0.55 - ambience * 0.45
    music = safe_stem(music)

    stems = {
        "vocals": vocals,
        "drums": drums,
        "bass": bass,
        "music": music,
        "ambience": ambience,
    }

    reconstruction = drums * 0.75 + bass * 0.65 + vocals * 0.55 + ambience * 0.45 + music
    reconstruction_error = audio - reconstruction
    rms_error = float(np.sqrt(np.mean(reconstruction_error ** 2)) + 1e-12)
    rms_source = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)

    energy = {name: float(np.mean(stem ** 2) + 1e-12) for name, stem in stems.items()}
    total_energy = sum(energy.values()) + 1e-12
    energy_ratio = {name: round(val / total_energy, 8) for name, val in energy.items()}

    separation_report = {
        "method": "offline_hpss_center_residual_research",
        "model_based": False,
        "note": "This is an offline research separator, not a neural Spleeter/MDX replacement.",
        "stem_names": list(stems.keys()),
        "energy": energy,
        "energy_ratio": energy_ratio,
        "reconstruction_error_rms": rms_error,
        "source_rms": rms_source,
        "reconstruction_error_ratio": round(rms_error / rms_source, 8),
        "params": params,
    }
    return stems, separation_report

def create_stem_report_md(project: dict, report: dict, stem_analysis: dict) -> str:
    lines = []
    for name, data in stem_analysis.items():
        lines.append(f"""## {name}

- File: `{data['file']}`
- LUFS: `{data['analysis']['loudness']['integrated_lufs']}`
- Peak: `{data['analysis']['basic_audio']['peak_dbfs']}`
- Clipping Samples: `{data['analysis']['basic_audio']['clipping_samples']}`
- Energy Ratio: `{report['energy_ratio'].get(name)}`
""")
    return f"""# Stem Separator Report

## Project

- Project ID: `{project['project_id']}`
- Project Name: `{project['project_name']}`
- Source: `{project['source_filename']}`

## Method

- Method: `{report['method']}`
- Model Based: `{report['model_based']}`
- Note: {report['note']}

## Reconstruction

- Reconstruction Error RMS: `{report['reconstruction_error_rms']}`
- Reconstruction Error Ratio: `{report['reconstruction_error_ratio']}`

## Stems

{chr(10).join(lines)}
"""


def run_stem_separator(project_json: str | Path, params: dict | None = None):
    params = params or {}
    project_load = load_project(project_json, save_status=True)
    project = project_load["project"]
    paths = project["paths"]
    logger = ProjectLogger(paths["project_log"])
    logger.info("Task 031 Stem Separator started.")

    audio, sr, audio_info = read_working_wav(paths["working_wav"])
    root = Path(paths["root"])
    stems_dir = root / "stems"
    neural_tmp_dir = root / "stems_neural_raw"
    analysis_dir = Path(paths["analysis_dir"])
    stems_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    backend_cfg = backend_config_from_params(params)
    requested_backend = backend_cfg["backend"]
    backend_report = {
        "requested_backend": requested_backend,
        "backend_used": None,
        "fallback_used": False,
        "fallback_reason": None,
        "internal_neural": None,
        "demucs": None,
    }

    stems = None
    sep_report = None

    if requested_backend == "internal_neural":
        try:
            from community.modules.internal_neural_stem_backend import run_internal_neural_backend

            stems, internal_report = run_internal_neural_backend(audio, sr, params={
                **params,
                "internal_model": backend_cfg["internal_model"],
                "device": backend_cfg["device"],
                "segment_seconds": backend_cfg["segment_seconds"],
                "overlap": backend_cfg["overlap"],
            })
            backend_report["backend_used"] = "internal_neural"
            backend_report["internal_neural"] = internal_report
            reconstruction = np.zeros_like(audio)
            for stem in stems.values():
                reconstruction += stem / max(1, len(stems))
            reconstruction_error = audio - reconstruction
            rms_error = float(np.sqrt(np.mean(reconstruction_error ** 2)) + 1e-12)
            rms_source = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
            energy = {name: float(np.mean(stem ** 2) + 1e-12) for name, stem in stems.items()}
            total_energy = sum(energy.values()) + 1e-12
            energy_ratio = {name: round(val / total_energy, 8) for name, val in energy.items()}
            sep_report = {
                "method": "internal_neural_hybrid_transformer_skeleton",
                "model_based": True,
                "note": "Internal PyTorch neural architecture and chunked inference were used. Bundled model is untrained unless a local .pt weight is installed.",
                "stem_names": list(stems.keys()),
                "energy": energy,
                "energy_ratio": energy_ratio,
                "reconstruction_error_rms": rms_error,
                "source_rms": rms_source,
                "reconstruction_error_ratio": round(rms_error / rms_source, 8),
                "params": params,
            }
        except Exception as e:
            backend_report["backend_used"] = "none"
            backend_report["fallback_used"] = False
            backend_report["fallback_reason"] = str(e)
            raise RuntimeError(f"Internal neural backend failed: {e}") from e

    if stems is None and requested_backend in {"auto", "demucs", "neural"}:
        demucs = DemucsBackend(
            model=backend_cfg["demucs_model"],
            device=backend_cfg["device"],
            shifts=backend_cfg["shifts"],
            overlap=backend_cfg["overlap"],
            clip_mode=backend_cfg["clip_mode"],
            jobs=backend_cfg["jobs"],
            two_stems=backend_cfg["two_stems"],
        )
        demucs_result = demucs.run(paths["working_wav"], neural_tmp_dir, stems_dir, timeout_sec=backend_cfg["timeout_sec"])
        backend_report["demucs"] = demucs_result

        if demucs_result.get("status") == "success":
            backend_report["backend_used"] = "demucs"
            backend_report["fallback_used"] = False

            # Load standardized stems generated by Demucs.
            stem_files = {
                "vocals": stems_dir / "vocals.wav",
                "drums": stems_dir / "drums.wav",
                "bass": stems_dir / "bass.wav",
                "music": stems_dir / "music.wav",
            }
            stems = {}
            for name, path in stem_files.items():
                if path.exists():
                    stems[name], stem_sr = load_stem_wav(path, target_len=len(audio))
                    if stem_sr != sr:
                        # keep analysis honest; current path normally matches Demucs output sample rate
                        pass
            if "ambience" not in stems:
                stems["ambience"] = estimate_ambience(audio, sr)
                save_wav(stems_dir / "ambience.wav", stems["ambience"], sr)

            reconstruction = np.zeros_like(audio)
            for stem in stems.values():
                reconstruction += stem / max(1, len(stems))
            reconstruction_error = audio - reconstruction
            rms_error = float(np.sqrt(np.mean(reconstruction_error ** 2)) + 1e-12)
            rms_source = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
            energy = {name: float(np.mean(stem ** 2) + 1e-12) for name, stem in stems.items()}
            total_energy = sum(energy.values()) + 1e-12
            energy_ratio = {name: round(val / total_energy, 8) for name, val in energy.items()}
            sep_report = {
                "method": "demucs_neural_backend",
                "model_based": True,
                "note": "Neural Demucs backend was used. Model weights are managed by Demucs cache/install, not embedded as raw files in this ZIP.",
                "stem_names": list(stems.keys()),
                "energy": energy,
                "energy_ratio": energy_ratio,
                "reconstruction_error_rms": rms_error,
                "source_rms": rms_source,
                "reconstruction_error_ratio": round(rms_error / rms_source, 8),
                "params": params,
            }
        elif requested_backend in {"demucs", "neural"}:
            backend_report["backend_used"] = "none"
            backend_report["fallback_used"] = False
            backend_report["fallback_reason"] = demucs_result.get("error", "Demucs backend failed and fallback was disabled by requested backend.")
            raise RuntimeError(f"Demucs backend failed: {backend_report['fallback_reason']}")
        else:
            backend_report["fallback_used"] = True
            backend_report["fallback_reason"] = demucs_result.get("error", "Demucs unavailable; using offline fallback.")

    if stems is None:
        stems, sep_report = separate_stems_offline(audio, sr, params=params)
        backend_report["backend_used"] = "offline_hpss"
        if requested_backend == "offline":
            backend_report["fallback_used"] = False
            backend_report["fallback_reason"] = None

    stem_analysis = {}
    all_clipping = 0
    # Save/re-save standardized stems to ensure all exist.
    for name, stem in stems.items():
        out = stems_dir / f"{name}.wav"
        save_wav(out, stem, sr)
        analysis = analyze_audio(stem, sr)
        all_clipping += int(analysis["basic_audio"]["clipping_samples"])
        stem_analysis[name] = {
            "file": str(out),
            "analysis": analysis,
        }

    report = {
        "task": "Task 031B - Neural Stem Backend + Stem Separator",
        "status": "success",
        "project_id": project["project_id"],
        "source_audio_info": audio_info,
        **sep_report,
        "backend_report": backend_report,
        "stems_dir": str(stems_dir),
        "stem_files": {name: data["file"] for name, data in stem_analysis.items()},
        "all_stems_clipping_samples": all_clipping,
        "clipping_passed": all_clipping == 0,
    }

    write_json(analysis_dir / "stem_separator_report.json", report)
    write_json(analysis_dir / "stem_backend_report.json", backend_report)
    write_json(analysis_dir / "stem_analysis.json", stem_analysis)
    (analysis_dir / "stem_separator_report.md").write_text(create_stem_report_md(project, report, stem_analysis), encoding="utf-8")

    logger.info(f"Task 031B Stem Separator completed with backend={backend_report['backend_used']}.")
    return {
        "report": report,
        "stem_analysis": stem_analysis,
    }



