from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import soundfile as sf
from scipy import signal

from community.analysis.dynamics import analyze_dynamics
from community.analysis.loudness import analyze_loudness, integrated_lufs
from community.analysis.spectrum import BANDS, analyze_spectrum
from community.analysis.stereo import analyze_stereo
from community.core.project_loader import load_project, read_json
from community.core.project_logger import ProjectLogger
from community.modules.dsp_utils import apply_gain_db, normalize_peak, sanitize


def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def save_wav(path: str | Path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, sanitize(audio), sr, subtype="PCM_24")
    return path


def _ensure_stereo(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        audio = audio[:, None]
    if audio.shape[1] == 1:
        return np.repeat(audio, 2, axis=1)
    if audio.shape[1] > 2:
        return audio[:, :2]
    return audio


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    if sr == target_sr:
        return audio
    factor = math.gcd(sr, target_sr)
    up = target_sr // factor
    down = sr // factor
    return signal.resample_poly(audio, up, down, axis=0)


def _load_audio(path: str | Path, target_sr: int | None = None) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)
    audio = _ensure_stereo(sanitize(audio))
    if target_sr is not None and sr != target_sr:
        audio = _resample(audio, sr, target_sr)
        sr = target_sr
    return audio, sr


def _analysis(audio: np.ndarray, sr: int) -> dict:
    return {
        "loudness": analyze_loudness(audio, sr),
        "spectrum": analyze_spectrum(audio, sr),
        "stereo": analyze_stereo(audio, sr),
        "dynamics": analyze_dynamics(audio, sr),
    }


def _default_compare_paths(project: dict, reference_path: str | Path | None = None) -> dict[str, str]:
    root = Path(project["paths"]["root"])
    analysis_dir = Path(project["paths"]["analysis_dir"])
    paths: dict[str, str] = {
        "original": project["paths"]["working_wav"],
    }

    quick_suite_path = root / "quick_master" / "quick_master_suite_report.json"
    if quick_suite_path.exists():
        suite = read_json(quick_suite_path)
        style_map = [("master_a", "clean"), ("master_b", "punch"), ("master_c", "loud")]
        for label, style in style_map:
            output = suite.get("styles", {}).get(style, {}).get("output_wav")
            if output:
                paths[label] = output

    render_report_path = analysis_dir / "render_report.json"
    if "master_a" not in paths and render_report_path.exists():
        render_report = read_json(render_report_path)
        paths["master_a"] = render_report["final_output_wav"]

    if reference_path:
        paths["reference"] = str(Path(reference_path).resolve())
    return paths


def _preview_slice(audio: np.ndarray, sr: int, start_sec: float, duration_sec: float) -> np.ndarray:
    start = max(0, int(start_sec * sr))
    length = max(1, int(duration_sec * sr))
    end = min(len(audio), start + length)
    clip = audio[start:end]
    if len(clip) < length:
        clip = np.pad(clip, ((0, length - len(clip)), (0, 0)))
    return sanitize(clip)


def _target_lufs_from_analyses(analyses: dict[str, dict], reference_label: str | None = None) -> float:
    if reference_label and reference_label in analyses:
        return float(analyses[reference_label]["loudness"]["integrated_lufs"])
    values = [float(item["loudness"]["integrated_lufs"]) for item in analyses.values()]
    return float(sum(values) / max(1, len(values)))


def run_loudness_match_compare(
    project_json: str | Path,
    compare_paths: dict[str, str],
    start_sec: float = 0.0,
    duration_sec: float = 6.0,
) -> dict:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    compare_root = Path(project["paths"]["root"]) / "compare" / "gain_matched_preview"
    logger = ProjectLogger(project["paths"]["project_log"])
    logger.info("Task 051 Loudness Match Compare started.")

    base_sr = None
    analyses: dict[str, dict] = {}
    raw_audio: dict[str, np.ndarray] = {}
    for label, path in compare_paths.items():
        audio, sr = _load_audio(path, target_sr=base_sr)
        if base_sr is None:
            base_sr = sr
        elif sr != base_sr:
            audio, sr = _load_audio(path, target_sr=base_sr)
        clip = _preview_slice(audio, base_sr, start_sec, duration_sec)
        raw_audio[label] = clip
        analyses[label] = _analysis(clip, base_sr)

    target_lufs = _target_lufs_from_analyses(analyses, reference_label="reference" if "reference" in analyses else None)
    previews = {}
    for label, clip in raw_audio.items():
        current_lufs = float(analyses[label]["loudness"]["integrated_lufs"])
        gain_db = float(target_lufs - current_lufs)
        matched = apply_gain_db(clip, gain_db)
        matched = sanitize(normalize_peak(matched, -1.0))
        output_path = compare_root / f"{label}_preview.wav"
        save_wav(output_path, matched, base_sr)
        matched_analysis = _analysis(matched, base_sr)
        previews[label] = {
            "source_path": compare_paths[label],
            "preview_wav": str(output_path),
            "source_lufs": current_lufs,
            "applied_gain_db": gain_db,
            "matched_lufs": matched_analysis["loudness"]["integrated_lufs"],
            "true_peak_after": matched_analysis["loudness"]["true_peak_dbtp"],
        }

    report = {
        "task": "Task 051 - Loudness Match Compare",
        "status": "success",
        "target_lufs": target_lufs,
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "previews": previews,
    }
    write_json(analysis_dir / "gain_matched_preview.json", report)
    logger.info("Task 051 Loudness Match Compare completed.")
    return report


def run_ab_player(
    project_json: str | Path,
    reference_path: str | Path | None = None,
    gain_match: bool = True,
    start_sec: float = 0.0,
    duration_sec: float = 6.0,
) -> dict:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    logger = ProjectLogger(project["paths"]["project_log"])
    logger.info("Task 050 A/B Player started.")

    compare_paths = _default_compare_paths(project, reference_path=reference_path)
    available = {}
    for label, path in compare_paths.items():
        exists = Path(path).exists()
        available[label] = {"path": path, "exists": exists}
    if not available.get("original", {}).get("exists"):
        raise FileNotFoundError("Original source for A/B player is missing.")

    matched_report = None
    if gain_match:
        matched_report = run_loudness_match_compare(project_json, {k: v["path"] for k, v in available.items() if v["exists"]}, start_sec=start_sec, duration_sec=duration_sec)

    manifest = {
        "task": "Task 050 - A/B Player",
        "status": "success",
        "start_sec": start_sec,
        "duration_sec": duration_sec,
        "gain_match_enabled": gain_match,
        "sources": available,
        "gain_matched_preview": matched_report["previews"] if matched_report else None,
    }
    write_json(analysis_dir / "ab_player.json", manifest)
    logger.info("Task 050 A/B Player completed.")
    return manifest


def _band_diff_db(source_report: dict, target_report: dict) -> dict[str, float]:
    out = {}
    for name in BANDS:
        src = float(source_report["raw_band_energy"][name]) + 1e-20
        tgt = float(target_report["raw_band_energy"][name]) + 1e-20
        out[name] = round(float(10.0 * np.log10(tgt / src)), 6)
    return out


def _save_spectrum_plot(output_path: Path, pairs: list[dict]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        output_path.write_bytes(b"")
        return

    labels = list(BANDS.keys())
    x = np.arange(len(labels))
    plt.figure(figsize=(10, 4))
    for pair in pairs:
        values = [pair["band_diff_db"][name] for name in labels]
        plt.plot(x, values, marker="o", label=pair["pair"])
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.ylabel("Delta dB")
    plt.title("Spectrum Compare")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_spectrum_compare(project_json: str | Path, reference_path: str | Path | None = None) -> dict:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    logger = ProjectLogger(project["paths"]["project_log"])
    logger.info("Task 052 Spectrum Compare started.")

    compare_paths = _default_compare_paths(project, reference_path=reference_path)
    analyses = {label: _analysis(*_load_audio(path)) for label, path in compare_paths.items() if Path(path).exists()}
    pairs = []
    pair_defs = [("original", "master_a")]
    if "reference" in analyses and "master_a" in analyses:
        pair_defs.append(("master_a", "reference"))

    for left, right in pair_defs:
        if left in analyses and right in analyses:
            band_diff = _band_diff_db(analyses[left]["spectrum"], analyses[right]["spectrum"])
            pairs.append({
                "pair": f"{left}_vs_{right}",
                "left": left,
                "right": right,
                "band_diff_db": band_diff,
                "correction_curve_db": {k: round(-v, 6) for k, v in band_diff.items()},
            })

    png_path = Path(project["paths"]["root"]) / "compare" / "spectrum_compare.png"
    _save_spectrum_plot(png_path, pairs)

    report = {
        "task": "Task 052 - Spectrum Compare",
        "status": "success",
        "pairs": pairs,
        "png_graph": str(png_path),
    }
    write_json(analysis_dir / "spectrum_compare.json", report)
    logger.info("Task 052 Spectrum Compare completed.")
    return report


def run_stereo_compare(project_json: str | Path, reference_path: str | Path | None = None) -> dict:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    logger = ProjectLogger(project["paths"]["project_log"])
    logger.info("Task 053 Stereo Compare started.")

    compare_paths = _default_compare_paths(project, reference_path=reference_path)
    analyses = {label: _analysis(*_load_audio(path)) for label, path in compare_paths.items() if Path(path).exists()}
    table = {
        label: {
            "stereo_width": data["stereo"]["stereo_width"],
            "mid_energy": data["stereo"]["mid_energy"],
            "side_energy": data["stereo"]["side_energy"],
            "phase_correlation": data["stereo"]["phase_correlation"],
            "mono_collapse_loss_db": data["stereo"]["mono_collapse_loss_db"],
            "low_end_stereo_leakage": data["stereo"]["low_end_stereo_leakage"],
            "warnings": data["stereo"]["warnings"],
        }
        for label, data in analyses.items()
    }

    comparisons = []
    pair_defs = [("original", "master_a")]
    if "reference" in analyses and "master_a" in analyses:
        pair_defs.append(("master_a", "reference"))
    for left, right in pair_defs:
        if left in analyses and right in analyses:
            left_data = analyses[left]["stereo"]
            right_data = analyses[right]["stereo"]
            comparisons.append({
                "pair": f"{left}_vs_{right}",
                "width_delta": round(float(right_data["stereo_width"] - left_data["stereo_width"]), 6),
                "phase_delta": round(float(right_data["phase_correlation"] - left_data["phase_correlation"]), 6),
                "mono_loss_delta_db": round(float(right_data["mono_collapse_loss_db"] - left_data["mono_collapse_loss_db"]), 6),
                "low_end_leakage_delta": round(float(right_data["low_end_stereo_leakage"] - left_data["low_end_stereo_leakage"]), 6),
            })

    report = {
        "task": "Task 053 - Stereo Compare",
        "status": "success",
        "table": table,
        "comparisons": comparisons,
    }
    write_json(analysis_dir / "stereo_compare.json", report)
    logger.info("Task 053 Stereo Compare completed.")
    return report


def _psr_like(metrics: dict) -> float:
    return round(float(metrics["loudness"]["sample_peak_dbfs"] - metrics["loudness"]["short_term_lufs_max"]), 6)


def run_dynamics_compare(project_json: str | Path, reference_path: str | Path | None = None) -> dict:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    logger = ProjectLogger(project["paths"]["project_log"])
    logger.info("Task 054 Dynamics Compare started.")

    compare_paths = _default_compare_paths(project, reference_path=reference_path)
    analyses = {label: _analysis(*_load_audio(path)) for label, path in compare_paths.items() if Path(path).exists()}
    table = {
        label: {
            "crest_factor_db": data["dynamics"]["crest_factor_db"],
            "dynamic_range_approx": data["dynamics"]["dynamic_range_approx"],
            "loudness_range": data["loudness"]["loudness_range"],
            "transient_density": data["dynamics"]["transient_density"],
            "psr_like": _psr_like(data),
            "overcompression_score": data["dynamics"]["overcompression_score"],
        }
        for label, data in analyses.items()
    }

    warnings = []
    if "master_a" in table and "reference" in table:
        master = table["master_a"]
        ref = table["reference"]
        if master["dynamic_range_approx"] + 0.75 < ref["dynamic_range_approx"]:
            warnings.append("master_dynamic_range_below_reference")
        if master["crest_factor_db"] + 0.75 < ref["crest_factor_db"]:
            warnings.append("master_crest_factor_below_reference")
    if "master_a" in table and table["master_a"]["overcompression_score"] > 0.75:
        warnings.append("master_a_too_squashed")

    comparisons = []
    pair_defs = [("original", "master_a")]
    if "reference" in table and "master_a" in table:
        pair_defs.append(("master_a", "reference"))
    for left, right in pair_defs:
        if left in table and right in table:
            comparisons.append({
                "pair": f"{left}_vs_{right}",
                "crest_factor_delta_db": round(float(table[right]["crest_factor_db"] - table[left]["crest_factor_db"]), 6),
                "dynamic_range_delta": round(float(table[right]["dynamic_range_approx"] - table[left]["dynamic_range_approx"]), 6),
                "loudness_range_delta": round(float(table[right]["loudness_range"] - table[left]["loudness_range"]), 6),
                "transient_density_delta": round(float(table[right]["transient_density"] - table[left]["transient_density"]), 6),
                "psr_like_delta": round(float(table[right]["psr_like"] - table[left]["psr_like"]), 6),
            })

    report = {
        "task": "Task 054 - Dynamics Compare",
        "status": "success",
        "table": table,
        "comparisons": comparisons,
        "warnings": warnings,
    }
    write_json(analysis_dir / "dynamics_compare.json", report)
    logger.info("Task 054 Dynamics Compare completed.")
    return report


def run_compare_suite(
    project_json: str | Path,
    reference_path: str | Path | None = None,
    gain_match: bool = True,
    start_sec: float = 0.0,
    duration_sec: float = 6.0,
) -> dict:
    ab = run_ab_player(project_json, reference_path=reference_path, gain_match=gain_match, start_sec=start_sec, duration_sec=duration_sec)
    spectrum = run_spectrum_compare(project_json, reference_path=reference_path)
    stereo = run_stereo_compare(project_json, reference_path=reference_path)
    dynamics = run_dynamics_compare(project_json, reference_path=reference_path)

    project = load_project(project_json, save_status=True)["project"]
    summary = {
        "task": "Task 050~054 - Compare Suite",
        "status": "success",
        "ab_player_ready": True,
        "gain_match_enabled": gain_match,
        "available_sources": [label for label, meta in ab["sources"].items() if meta["exists"]],
        "spectrum_pairs": [pair["pair"] for pair in spectrum["pairs"]],
        "stereo_pairs": [pair["pair"] for pair in stereo["comparisons"]],
        "dynamics_pairs": [pair["pair"] for pair in dynamics["comparisons"]],
        "dynamics_warnings": dynamics["warnings"],
    }
    write_json(Path(project["paths"]["analysis_dir"]) / "compare_suite.json", summary)
    return {
        "ab_player": ab,
        "spectrum_compare": spectrum,
        "stereo_compare": stereo,
        "dynamics_compare": dynamics,
        "summary": summary,
    }



