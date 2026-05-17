from __future__ import annotations

from pathlib import Path
import json

from pro.assistant.refinement_controls import build_refinement_controls
from shared.utils.config_loader import load_config


ASSISTANT_VERSIONS = ["legacy", "refined"]
REFINEMENT_PROFILE_OPTIONS = ["auto", "balanced", "low_end_preserve", "vocal_forward", "gentle_cleanup"]


def load_genre_options() -> list[str]:
    genres = load_config("genres").get("genres", [])
    cleaned = [str(item).strip() for item in genres if str(item).strip()]
    return cleaned or ["hiphop", "pop", "ballad", "edm", "rock"]


def init_ui_state(st) -> None:
    st.session_state.setdefault("current_project_json", "")
    st.session_state.setdefault("current_project_root", "")
    st.session_state.setdefault("current_project_name", "")
    st.session_state.setdefault("current_reference_path", "")
    st.session_state.setdefault("last_suite_report", None)
    st.session_state.setdefault("last_compare_report", None)
    st.session_state.setdefault("last_export_report", None)
    st.session_state.setdefault("last_queue_report", None)
    st.session_state.setdefault("last_batch_result", None)
    st.session_state.setdefault("assistant_version", "legacy")
    st.session_state.setdefault("refinement_profile", "auto")
    st.session_state.setdefault("refinement_low_end_cut_scale", 0.48)
    st.session_state.setdefault("refinement_presence_restraint", 0.72)
    st.session_state.setdefault("refinement_air_restraint", 0.64)
    st.session_state.setdefault("refinement_stereo_restraint", 0.78)
    st.session_state.setdefault("refinement_loudness_restraint", 0.74)


def set_current_project(st, project_json: str | Path, project_root: str | Path | None = None, project_name: str | None = None) -> None:
    project_json = str(Path(project_json))
    st.session_state["current_project_json"] = project_json
    if project_root is not None:
        st.session_state["current_project_root"] = str(Path(project_root))
    if project_name is not None:
        st.session_state["current_project_name"] = project_name


def set_reference_path(st, reference_path: str | Path | None) -> None:
    st.session_state["current_reference_path"] = str(reference_path or "")


def extract_project_context(result: dict) -> dict:
    import_report = result.get("import_report") or {}
    project_root = import_report.get("project_root")
    project_name = None
    if "research_report" in result:
        project_name = result["research_report"]["report"]["project"]["project_name"]
    elif project_root:
        project_name = Path(project_root).name

    project_json = str(Path(project_root) / "project.json") if project_root else ""
    return {
        "project_root": project_root or "",
        "project_json": project_json,
        "project_name": project_name or "",
    }


def render_project_context(st) -> None:
    project_json = st.session_state.get("current_project_json", "")
    project_name = st.session_state.get("current_project_name", "")
    reference_path = st.session_state.get("current_reference_path", "")
    if project_json:
        st.caption(f"Current Project: `{project_name or Path(project_json).stem}`")
        st.caption(f"Project JSON: `{project_json}`")
    if reference_path:
        st.caption(f"Reference: `{reference_path}`")


def summarize_qc_failures(qc_report: dict) -> list[str]:
    failures = []
    for check in qc_report.get("checks", []):
        if check.get("status") in {"fail", "warn"}:
            failures.append(f"{check['name']}: {check['status']} ({check['value']})")
    return failures


def list_preset_files(project_root: str | Path) -> list[Path]:
    preset_dir = Path(project_root) / "presets"
    if not preset_dir.exists():
        return []
    return sorted(preset_dir.glob("*.json"))


def save_module_preset(project_root: str | Path, preset_name: str, params: dict) -> Path:
    preset_dir = Path(project_root) / "presets"
    preset_dir.mkdir(parents=True, exist_ok=True)
    preset_path = preset_dir / f"{preset_name}.json"
    preset_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    return preset_path


def load_module_preset(preset_path: str | Path) -> dict:
    return json.loads(Path(preset_path).read_text(encoding="utf-8"))


def as_file_url(path_like: str | Path | None) -> str:
    if not path_like:
        return ""
    return Path(path_like).resolve().as_uri()


def build_refinement_override_payload(
    low_end_cut_scale: float,
    presence_restraint: float,
    air_restraint: float,
    stereo_restraint: float,
    loudness_restraint: float,
) -> dict:
    return {
        "low_end_cut_scale": round(float(low_end_cut_scale), 4),
        "presence_restraint": round(float(presence_restraint), 4),
        "air_restraint": round(float(air_restraint), 4),
        "stereo_restraint": round(float(stereo_restraint), 4),
        "loudness_restraint": round(float(loudness_restraint), 4),
    }


def resolve_refinement_ui_payload(
    genre: str,
    style: str,
    assistant_version: str,
    refinement_profile: str,
    low_end_cut_scale: float,
    presence_restraint: float,
    air_restraint: float,
    stereo_restraint: float,
    loudness_restraint: float,
) -> dict:
    if assistant_version != "refined":
        return {
            "assistant_version": assistant_version,
            "refinement_profile": None,
            "refinement_overrides": None,
            "resolved_controls": None,
        }
    overrides = build_refinement_override_payload(
        low_end_cut_scale,
        presence_restraint,
        air_restraint,
        stereo_restraint,
        loudness_restraint,
    )
    profile_name = None if refinement_profile == "auto" else refinement_profile
    controls = build_refinement_controls(
        genre,
        style,
        requested_profile=profile_name,
        overrides=overrides,
    )
    return {
        "assistant_version": assistant_version,
        "refinement_profile": profile_name,
        "refinement_overrides": overrides,
        "resolved_controls": controls,
    }
