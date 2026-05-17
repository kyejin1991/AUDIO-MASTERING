from __future__ import annotations

from pathlib import Path
import html
import json

from community.core.project_loader import load_project, read_json


def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_markdown(path: str | Path, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def write_html(path: str | Path, html_text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html_text, encoding="utf-8")
    return path


def _optional_json(analysis_dir: Path, name: str) -> dict | None:
    path = analysis_dir / name
    if not path.exists():
        return None
    return read_json(path)


def _available_report_files(analysis_dir: Path) -> list[str]:
    return sorted(p.name for p in analysis_dir.glob("*.json"))


def _module_change_summary(step: dict) -> dict:
    before = step.get("before", {})
    after = step.get("after", {})
    return {
        "module": step.get("module"),
        "render_behavior": step.get("render_behavior"),
        "input_lufs": before.get("loudness", {}).get("integrated_lufs"),
        "output_lufs": after.get("loudness", {}).get("integrated_lufs"),
        "input_true_peak_dbtp": before.get("loudness", {}).get("true_peak_dbtp"),
        "output_true_peak_dbtp": after.get("loudness", {}).get("true_peak_dbtp"),
        "input_dynamic_range": before.get("dynamics", {}).get("dynamic_range_approx"),
        "output_dynamic_range": after.get("dynamics", {}).get("dynamic_range_approx"),
        "input_stereo_width": before.get("stereo", {}).get("stereo_width"),
        "output_stereo_width": after.get("stereo", {}).get("stereo_width"),
        "module_report": step.get("module_report", {}),
        "output_file": step.get("output_file"),
    }


def _build_summary(project: dict, analysis_bundle: dict) -> dict:
    full = analysis_bundle.get("full_analysis") or {}
    final_master = analysis_bundle.get("final_master_analysis") or full
    assistant = analysis_bundle.get("ai_assistant") or {}
    reference = analysis_bundle.get("reference_analysis") or {}
    youtube_qc = analysis_bundle.get("youtube_upload_qc") or analysis_bundle.get("youtube_qc") or {}
    audio_restore = analysis_bundle.get("audio_restore") or {}

    return {
        "project_name": project["project_name"],
        "source_filename": project["source_filename"],
        "source_lufs": full.get("loudness", {}).get("integrated_lufs"),
        "final_lufs": final_master.get("loudness", {}).get("integrated_lufs"),
        "source_true_peak_dbtp": full.get("loudness", {}).get("true_peak_dbtp"),
        "final_true_peak_dbtp": final_master.get("loudness", {}).get("true_peak_dbtp"),
        "source_dynamic_range": full.get("dynamics", {}).get("dynamic_range_approx"),
        "final_dynamic_range": final_master.get("dynamics", {}).get("dynamic_range_approx"),
        "source_stereo_width": full.get("stereo", {}).get("stereo_width"),
        "final_stereo_width": final_master.get("stereo", {}).get("stereo_width"),
        "target_lufs": assistant.get("target", {}).get("target_lufs"),
        "target_ceiling_dbtp": assistant.get("target", {}).get("ceiling_dbtp"),
        "youtube_qc_overall": youtube_qc.get("overall"),
        "reference_ready": bool(reference),
        "audio_restore_used": bool(audio_restore) and audio_restore.get("status") == "success",
        "source_noise_floor_dbfs": full.get("basic_audio", {}).get("noise_floor_dbfs"),
        "restored_noise_floor_dbfs": audio_restore.get("noise_floor_after_dbfs"),
    }


def _analysis_snapshot(analysis_bundle: dict) -> dict:
    keys = [
        "full_analysis",
        "master_assistant",
        "recommended_chain",
        "module_parameter_draft",
        "processing_chain",
        "render_report",
        "final_master_analysis",
        "reference_analysis",
        "reference_match_summary",
        "audio_restore",
        "compare_suite",
        "youtube_upload_qc",
        "codec_preview_project",
        "auto_ceiling_repair",
        "export_bundle",
    ]
    return {key: analysis_bundle.get(key) for key in keys if analysis_bundle.get(key) is not None}


def _load_analysis_bundle(analysis_dir: Path) -> dict:
    mapping = {
        "full_analysis": "full_analysis.json",
        "ai_assistant": "master_assistant.json",
        "recommended_chain": "recommended_chain.json",
        "module_parameter_draft": "module_parameter_draft.json",
        "processing_chain": "processing_chain.json",
        "render_report": "render_report.json",
        "final_master_analysis": "final_master_analysis.json",
        "reference_analysis": "reference_analysis.json",
        "reference_match_summary": "reference_match_summary.json",
        "audio_restore": "audio_restore_report.json",
        "compare_suite": "compare_suite.json",
        "youtube_upload_qc": "youtube_upload_qc.json",
        "youtube_qc": "youtube_qc.json",
        "codec_preview_project": "codec_preview_project.json",
        "auto_ceiling_repair": "auto_ceiling_repair.json",
        "export_bundle": "export_bundle.json",
    }
    return {key: _optional_json(analysis_dir, filename) for key, filename in mapping.items() if _optional_json(analysis_dir, filename) is not None}


def _markdown_report(project: dict, analysis_bundle: dict, summary: dict, module_log: dict) -> str:
    assistant = analysis_bundle.get("ai_assistant") or {}
    chain = analysis_bundle.get("processing_chain") or {}
    reference = analysis_bundle.get("reference_match_summary") or {}
    compare = analysis_bundle.get("compare_suite") or {}
    youtube_qc = analysis_bundle.get("youtube_upload_qc") or analysis_bundle.get("youtube_qc") or {}
    codec = analysis_bundle.get("codec_preview_project") or {}
    auto_ceiling = analysis_bundle.get("auto_ceiling_repair") or {}
    export_bundle = analysis_bundle.get("export_bundle") or {}
    audio_restore = analysis_bundle.get("audio_restore") or {}

    module_lines = []
    for item in module_log.get("modules", []):
        module_lines.append(
            "\n".join(
                [
                    f"### {item['module']}",
                    f"- Behavior: `{item['render_behavior']}`",
                    f"- LUFS: `{item['input_lufs']}` -> `{item['output_lufs']}`",
                    f"- True Peak: `{item['input_true_peak_dbtp']}` -> `{item['output_true_peak_dbtp']}`",
                    f"- Dynamic Range: `{item['input_dynamic_range']}` -> `{item['output_dynamic_range']}`",
                    f"- Stereo Width: `{item['input_stereo_width']}` -> `{item['output_stereo_width']}`",
                    f"- Output File: `{item['output_file']}`",
                    "",
                    "```json",
                    json.dumps(item["module_report"], indent=2, ensure_ascii=False),
                    "```",
                ]
            )
        )

    qc_lines = [f"- {check['name']}: `{check['status']}` ({check['value']})" for check in youtube_qc.get("checks", [])]
    export_lines = [f"- `{item['format']}` -> `{item['file']}`" for item in export_bundle.get("export_files", [])]

    return "\n".join(
        [
            "# Full Research Report",
            "",
            "## Project",
            "",
            f"- Project ID: `{project['project_id']}`",
            f"- Project Name: `{project['project_name']}`",
            f"- Source: `{project['source_filename']}`",
            "",
            "## Summary",
            "",
            f"- Source LUFS: `{summary['source_lufs']}`",
            f"- Final LUFS: `{summary['final_lufs']}`",
            f"- Source True Peak: `{summary['source_true_peak_dbtp']}`",
            f"- Final True Peak: `{summary['final_true_peak_dbtp']}`",
            f"- Source Dynamic Range: `{summary['source_dynamic_range']}`",
            f"- Final Dynamic Range: `{summary['final_dynamic_range']}`",
            f"- Source Stereo Width: `{summary['source_stereo_width']}`",
            f"- Final Stereo Width: `{summary['final_stereo_width']}`",
            f"- Target LUFS: `{summary['target_lufs']}`",
            f"- Target Ceiling: `{summary['target_ceiling_dbtp']}`",
            f"- YouTube QC Overall: `{summary['youtube_qc_overall']}`",
            f"- Audio Restore Used: `{summary['audio_restore_used']}`",
            f"- Noise Floor: `{summary['source_noise_floor_dbfs']}` -> `{summary['restored_noise_floor_dbfs']}`",
            "",
            "## Assistant Chain",
            "",
            f"- Genre: `{assistant.get('genre')}`",
            f"- Style: `{assistant.get('style')}`",
            f"- Active Chain: `{chain.get('ordered_chain', [])}`",
            "",
            "## Reference Match",
            "",
            f"- Ready: `{bool(reference)}`",
            f"- Tone Distance: `{reference.get('tone_distance_before')}` -> `{reference.get('tone_distance_after')}`",
            f"- Loudness Distance: `{reference.get('loudness_diff_before')}` -> `{reference.get('loudness_diff_after')}`",
            f"- Dynamics Distance: `{reference.get('dynamics_distance_before')}` -> `{reference.get('dynamics_distance_after')}`",
            f"- Stereo Distance: `{reference.get('stereo_distance_before')}` -> `{reference.get('stereo_distance_after')}`",
            "",
            "## Audio Restore",
            "",
            f"- Status: `{audio_restore.get('status')}`",
            f"- Air Index: `{audio_restore.get('air_index_before')}` -> `{audio_restore.get('air_index_after')}`",
            f"- Brightness Index: `{audio_restore.get('brightness_index_before')}` -> `{audio_restore.get('brightness_index_after')}`",
            f"- Crest Factor: `{audio_restore.get('crest_factor_before')}` -> `{audio_restore.get('crest_factor_after')}`",
            "",
            "## Compare Suite",
            "",
            f"- Sources: `{compare.get('available_sources')}`",
            f"- Spectrum Pairs: `{compare.get('spectrum_pairs')}`",
            f"- Stereo Pairs: `{compare.get('stereo_pairs')}`",
            f"- Dynamics Warnings: `{compare.get('dynamics_warnings')}`",
            "",
            "## QC",
            "",
            *(qc_lines or ["- No QC checks recorded."]),
            "",
            "## Codec Preview",
            "",
            f"- Preview OK: `{codec.get('all_previews_ok')}`",
            f"- Peak Warning: `{codec.get('warning_report', {}).get('has_peak_warning')}`",
            f"- Clipping Warning: `{codec.get('warning_report', {}).get('has_clipping_warning')}`",
            "",
            "## Auto Ceiling Repair",
            "",
            f"- Status: `{auto_ceiling.get('status')}`",
            f"- Attempts: `{len(auto_ceiling.get('attempts', []))}`",
            f"- Repaired WAV: `{auto_ceiling.get('repaired_wav')}`",
            "",
            "## Export",
            "",
            *(export_lines or ["- No export bundle recorded."]),
            "",
            "## Module Processing",
            "",
            *(module_lines or ["- No module processing steps recorded."]),
        ]
    )


def _html_report(project: dict, report: dict, markdown_text: str) -> str:
    pretty_json = html.escape(json.dumps(report, indent=2, ensure_ascii=False))
    pretty_md = html.escape(markdown_text)
    return f"""<html>
<head>
  <meta charset="utf-8" />
  <title>Full Research Report - {html.escape(project['project_name'])}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; line-height: 1.5; }}
    h1, h2, h3 {{ margin-top: 1.2em; }}
    pre {{ background: #f5f5f5; padding: 12px; border-radius: 6px; overflow-x: auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }}
    .card {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
  </style>
</head>
<body>
  <h1>Full Research Report</h1>
  <div class="grid">
    <div class="card"><strong>Project</strong><br />{html.escape(project['project_name'])}</div>
    <div class="card"><strong>Source</strong><br />{html.escape(project['source_filename'])}</div>
  </div>
  <h2>Readable Report</h2>
  <pre>{pretty_md}</pre>
  <h2>JSON Artifact</h2>
  <pre>{pretty_json}</pre>
</body>
</html>"""


def build_research_report(project_json: str | Path, pipeline_result: dict) -> dict:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    analysis_bundle = _load_analysis_bundle(analysis_dir)
    module_log = build_module_processing_log(project_json)
    summary = _build_summary(project, analysis_bundle)

    report = {
        "task": "Task 090 - Full Research Report",
        "status": "success",
        "project": {
            "project_id": project["project_id"],
            "project_name": project["project_name"],
            "source_filename": project["source_filename"],
            "project_root": project["paths"]["root"],
        },
        "summary": summary,
        "analysis_snapshot": _analysis_snapshot(analysis_bundle),
        "module_processing_log": module_log,
        "pipeline_artifacts": pipeline_result,
        "available_analysis_files": _available_report_files(analysis_dir),
    }

    markdown_text = _markdown_report(project, analysis_bundle, summary, module_log)
    json_path = write_json(analysis_dir / "report.json", report)
    md_path = write_markdown(analysis_dir / "report.md", markdown_text)
    html_path = write_html(analysis_dir / "report.html", _html_report(project, report, markdown_text))

    return {
        "report": report,
        "files": {
            "json": str(json_path),
            "md": str(md_path),
            "html": str(html_path),
        },
    }


def build_module_processing_log(project_json: str | Path) -> dict:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    render_report = _optional_json(analysis_dir, "render_report.json") or {}
    modules = [_module_change_summary(step) for step in render_report.get("steps", [])]

    report_files = sorted(p.name for p in analysis_dir.glob("*_report.json"))
    by_module_file = []
    for filename in report_files:
        module_report = read_json(analysis_dir / filename)
        by_module_file.append(
            {
                "file": filename,
                "task": module_report.get("task"),
                "status": module_report.get("status"),
                "highlights": {
                    key: module_report.get(key)
                    for key in [
                        "target_lufs",
                        "final_lufs",
                        "ceiling_dbtp",
                        "clipping_samples_after",
                        "noise_floor_after_dbfs",
                        "noise_floor_delta_db",
                        "air_index_after",
                        "brightness_index_after",
                        "phase_protection_passed",
                        "mono_compatibility_passed",
                        "recovery_applied",
                    ]
                    if key in module_report
                },
            }
        )

    lines = []
    for item in modules:
        lines.append(
            {
                "module": item["module"],
                "summary": (
                    f"{item['module']}: LUFS {item['input_lufs']} -> {item['output_lufs']}, "
                    f"TP {item['input_true_peak_dbtp']} -> {item['output_true_peak_dbtp']}, "
                    f"DR {item['input_dynamic_range']} -> {item['output_dynamic_range']}"
                ),
            }
        )

    log = {
        "task": "Task 091 - Module Processing Log",
        "status": "success",
        "project_id": project["project_id"],
        "project_name": project["project_name"],
        "module_count": len(modules),
        "modules": modules,
        "module_summary_lines": lines,
        "available_reports": report_files,
        "report_highlights": by_module_file,
    }
    write_json(analysis_dir / "module_processing_log.json", log)
    return log

