from __future__ import annotations

from pathlib import Path
import json

from community.core.ffmpeg_tools import require_ffmpeg
from community.core.project_loader import load_project, read_json
from community.core.project_logger import ProjectLogger
from community.core.shell import run_command
from pro.qc.workflows import run_youtube_upload_qc


EXPORT_FORMATS = {
    "wav_48k_24": {"suffix": ".wav"},
    "wav_44k1_24": {"suffix": ".wav"},
    "mp3_320": {"suffix": ".mp3"},
    "aac_192": {"suffix": ".m4a"},
}


def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _load_rendered_master(project_json: str | Path) -> tuple[dict, dict, Path]:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    render_report_path = analysis_dir / "render_report.json"
    if not render_report_path.exists():
        raise FileNotFoundError(f"render_report.json missing: {render_report_path}")
    render_report = read_json(render_report_path)
    final_wav = Path(render_report["final_output_wav"])
    if not final_wav.exists():
        raise FileNotFoundError(f"Final master missing: {final_wav}")
    return project, render_report, final_wav


def _expand_requested_formats(requested_formats: list[str]) -> list[str]:
    expanded = []
    for name in requested_formats:
        if name == "youtube_preset":
            expanded.extend(["wav_48k_24", "mp3_320"])
        elif name == "archive_preset":
            expanded.extend(["wav_48k_24", "wav_44k1_24", "mp3_320", "aac_192"])
        else:
            expanded.append(name)
    out = []
    for name in expanded:
        if name in EXPORT_FORMATS and name not in out:
            out.append(name)
    return out


def _export_with_ffmpeg(source_wav: Path, out_path: Path, format_name: str) -> None:
    require_ffmpeg()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if format_name == "wav_48k_24":
        cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(source_wav), "-ar", "48000", "-ac", "2", "-c:a", "pcm_s24le", str(out_path)]
    elif format_name == "wav_44k1_24":
        cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(source_wav), "-ar", "44100", "-ac", "2", "-c:a", "pcm_s24le", str(out_path)]
    elif format_name == "mp3_320":
        cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(source_wav), "-codec:a", "libmp3lame", "-b:a", "320k", str(out_path)]
    elif format_name == "aac_192":
        cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(source_wav), "-c:a", "aac", "-b:a", "192k", str(out_path)]
    else:
        raise ValueError(f"Unsupported export format: {format_name}")
    run_command(cmd)


def export_project_bundle(project_json: str | Path, requested_formats: list[str]) -> dict:
    project, _, final_wav = _load_rendered_master(project_json)
    logger = ProjectLogger(project["paths"]["project_log"])
    logger.info("Task 084 Export Bundle started.")

    qc_report = run_youtube_upload_qc(project_json)
    export_dir = Path(project["paths"]["root"]) / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)

    selected_formats = _expand_requested_formats(requested_formats)
    export_files = []
    for format_name in selected_formats:
        suffix = EXPORT_FORMATS[format_name]["suffix"]
        out_path = export_dir / f"{project['project_name']}_{format_name}{suffix}"
        _export_with_ffmpeg(final_wav, out_path, format_name)
        export_files.append({"format": format_name, "file": str(out_path)})

    failures = [check for check in qc_report.get("checks", []) if check.get("status") == "fail"]
    report = {
        "task": "Task 084 - Export Bundle",
        "status": "success",
        "project_id": project["project_id"],
        "project_name": project["project_name"],
        "qc_report": qc_report,
        "qc_failures": failures,
        "requested_formats": requested_formats,
        "selected_formats": selected_formats,
        "export_files": export_files,
    }
    write_json(Path(project["paths"]["analysis_dir"]) / "export_bundle.json", report)
    logger.info(f"Task 084 Export Bundle completed. files={len(export_files)}")
    return report

