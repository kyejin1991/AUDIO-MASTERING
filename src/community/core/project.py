from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime

from community.core.audio_io import create_project_from_audio
from community.core.project_loader import load_project, read_json


@dataclass
class WorkspacePaths:
    root: Path
    output_dir: Path
    temp_dir: Path
    projects_dir: Path
    reports_dir: Path
    batch_dir: Path


def resolve_workspace_paths(root: str | Path | None = None) -> WorkspacePaths:
    root_path = Path(root) if root else Path.cwd()
    output_dir = root_path / "output"
    temp_dir = root_path / "temp"
    projects_dir = output_dir / "projects"
    reports_dir = output_dir / "reports"
    batch_dir = output_dir / "batch"
    for path in [output_dir, temp_dir, projects_dir, reports_dir, batch_dir]:
        path.mkdir(parents=True, exist_ok=True)
    return WorkspacePaths(
        root=root_path,
        output_dir=output_dir,
        temp_dir=temp_dir,
        projects_dir=projects_dir,
        reports_dir=reports_dir,
        batch_dir=batch_dir,
    )


def create_project(
    input_path: str | Path,
    workspace_root: str | Path | None = None,
    project_name: str | None = None,
    working_sample_rate: int = 48000,
    working_channels: int = 2,
) -> dict:
    workspace = resolve_workspace_paths(workspace_root)
    return create_project_from_audio(
        input_path=input_path,
        projects_dir=workspace.projects_dir,
        project_name=project_name,
        working_sample_rate=working_sample_rate,
        working_channels=working_channels,
    )


def load_project_bundle(project_json: str | Path) -> dict:
    loaded = load_project(project_json, save_status=True)
    project = loaded["project"]
    return {
        "project": project,
        "paths": project["paths"],
        "analysis_dir": Path(project["paths"]["analysis_dir"]),
    }


def create_album_project(
    input_paths: list[str | Path],
    workspace_root: str | Path | None = None,
    album_name: str = "album_project",
) -> dict:
    workspace = resolve_workspace_paths(workspace_root)
    album_root = workspace.batch_dir / f"{datetime.now():%Y%m%d_%H%M%S}_{album_name}"
    album_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "task": "Task 070 - Batch Import",
        "status": "success",
        "album_name": album_name,
        "album_root": str(album_root),
        "tracks": [str(Path(p).resolve()) for p in input_paths],
        "track_count": len(input_paths),
    }
    (album_root / "album_project.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload

