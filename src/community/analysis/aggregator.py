from pathlib import Path
import json
from community.core.project_logger import ProjectLogger

def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def build_diagnosis_flags(basic: dict, loudness: dict, spectrum: dict, stereo: dict, dynamics: dict) -> dict:
    flags = {
        "too_quiet": loudness["integrated_lufs"] < -15.0,
        "too_loud_for_youtube_safe": loudness["integrated_lufs"] > -12.0,
        "true_peak_risk": loudness["true_peak_dbtp"] > -1.0,
        "clipping_detected": basic["clipping_samples"] > 0,
        "dc_offset_risk": abs(basic["dc_offset_l"]) > 0.01 or abs(basic["dc_offset_r"]) > 0.01,
        "too_much_silence": basic["silence_ratio"] > 0.25,
        "noisy_floor": basic.get("noise_floor_dbfs", -70.0) > -52.0,
        "muddy_low_mid": spectrum["mud_index"] > 0.28,
        "harsh_presence": spectrum["harshness_index"] > 0.22,
        "weak_air": spectrum["air_index"] < 0.035,
        "too_much_low_end": spectrum["low_end_index"] > 0.55,
        "thin_low_end": spectrum["low_end_index"] < 0.12,
        "wide_low_end": stereo["low_end_stereo_leakage"] > 0.15,
        "phase_risk": stereo["phase_correlation"] < 0.0 or stereo["mono_collapse_loss_db"] < -6.0,
        "too_narrow": stereo["stereo_width"] < 0.08,
        "too_wide": stereo["stereo_width"] > 0.85,
        "overcompressed": dynamics["overcompression_score"] > 0.58,
        "needs_unlimiter": dynamics["needs_unlimiter"],
    }

    flags["needs_bass_control"] = flags["too_much_low_end"] or flags["wide_low_end"] or flags["thin_low_end"]
    flags["needs_deesser"] = flags["harsh_presence"]
    flags["needs_imager"] = flags["too_narrow"] or flags["too_wide"] or flags["phase_risk"]
    flags["needs_maximizer"] = flags["too_quiet"] or loudness["integrated_lufs"] < -14.5
    flags["needs_eq"] = flags["muddy_low_mid"] or flags["harsh_presence"] or flags["weak_air"] or flags["too_much_low_end"] or flags["thin_low_end"]
    flags["needs_dynamics_control"] = flags["overcompressed"] or dynamics["dynamic_range_approx"] > 14.0
    flags["needs_audio_restore"] = (
        flags["noisy_floor"]
        or (flags["weak_air"] and spectrum["brightness_index"] < 0.075)
        or (flags["too_much_silence"] and spectrum["brightness_index"] < 0.09)
    )

    return {k: bool(v) for k, v in flags.items()}

def create_analysis_summary_md(project: dict, basic: dict, loudness: dict, spectrum: dict, genre_match: dict, stereo: dict, dynamics: dict, flags: dict) -> str:
    active_flags = [k for k, v in flags.items() if v]
    if not active_flags:
        active_flags = ["no_major_issue_detected"]

    return f"""# Analysis Summary

## Project

- Project ID: `{project["project_id"]}`
- Project Name: `{project["project_name"]}`
- Source: `{project["source_filename"]}`

## Basic

- Duration: {basic["duration_sec"]} sec
- Peak: {basic["peak_dbfs"]} dBFS
- RMS: {basic["rms_dbfs"]} dBFS
- Crest Factor: {basic["crest_factor_db"]} dB
- Clipping Samples: {basic["clipping_samples"]}
- Silence Ratio: {basic["silence_ratio"]}
- Noise Floor: {basic["noise_floor_dbfs"]} dBFS

## Loudness

- Integrated LUFS: {loudness["integrated_lufs"]}
- True Peak: {loudness["true_peak_dbtp"]} dBTP
- Loudness Range: {loudness["loudness_range"]}
- YouTube Target Difference: {loudness["youtube_lufs_diff"]} dB
- Recommended Gain: {loudness["recommended_gain_db"]} dB

## Spectrum

- Low-End Index: {spectrum["low_end_index"]}
- Mud Index: {spectrum["mud_index"]}
- Harshness Index: {spectrum["harshness_index"]}
- Air Index: {spectrum["air_index"]}
- Spectral Centroid: {spectrum["spectral_centroid_hz"]} Hz
- Spectral Rolloff: {spectrum["spectral_rolloff_hz"]} Hz

## Genre Match

- Inferred Genre: {genre_match["inferred_genre"]}
- Best Distance: {genre_match["best_distance"]}
- Confidence Margin: {genre_match["confidence_margin"]}
- Top Match Confidence: {genre_match["confidence"][genre_match["inferred_genre"]]}

## Stereo / Phase

- Stereo Width: {stereo["stereo_width"]}
- Phase Correlation: {stereo["phase_correlation"]}
- Mono Collapse Loss: {stereo["mono_collapse_loss_db"]} dB
- Low-End Stereo Leakage: {stereo["low_end_stereo_leakage"]}

## Dynamics

- Dynamic Range Approx: {dynamics["dynamic_range_approx"]}
- Transient Density: {dynamics["transient_density"]}
- Punch Score: {dynamics["punch_score"]}
- Overcompression Score: {dynamics["overcompression_score"]}
- Limiter Damage Risk: {dynamics["limiter_damage_risk"]}

## Diagnosis Flags

{chr(10).join(f"- {flag}" for flag in active_flags)}

## Next Step

This `full_analysis.json` is ready for Task 010 Master Assistant.
"""

def aggregate_full_analysis(project_load_result: dict, basic: dict, loudness: dict, spectrum: dict, genre_match: dict, stereo: dict, dynamics: dict) -> dict:
    project = project_load_result["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    logger = ProjectLogger(project["paths"]["project_log"])
    logger.info("Task 009 full analysis aggregation started.")

    flags = build_diagnosis_flags(basic, loudness, spectrum, stereo, dynamics)

    full = {
        "task": "Task 009 - Full Analysis Aggregator",
        "status": "success",
        "project": {
            "project_id": project["project_id"],
            "project_name": project["project_name"],
            "source_filename": project["source_filename"],
            "working_wav": project["paths"]["working_wav"],
        },
        "basic_audio": basic,
        "loudness": loudness,
        "spectrum": spectrum,
        "genre_match": genre_match,
        "stereo": stereo,
        "dynamics": dynamics,
        "diagnosis_flags": flags,
        "assistant_input_ready": True,
    }

    write_json(analysis_dir / "full_analysis.json", full)
    summary = create_analysis_summary_md(project, basic, loudness, spectrum, genre_match, stereo, dynamics, flags)
    (analysis_dir / "analysis_summary.md").write_text(summary, encoding="utf-8")
    logger.info("Task 009 full analysis aggregation completed.")
    return full



