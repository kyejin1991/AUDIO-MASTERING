from pathlib import Path
import json
from .project_logger import ProjectLogger

REQUIRED_PROJECT_KEYS = [
    "project_id", "project_name", "created_at", "source_filename", "source_sha256", "paths"
]

# Reduced for Community Edition compatibility
REQUIRED_PATH_KEYS = [
    "root", "working_wav", "analysis_dir", "project_log"
]

def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def load_project(project_json: str | Path, save_status: bool = True) -> dict:
    """
    Task 002: Load project.json + Validation + Status Save
    """
    project_json = Path(project_json).expanduser().resolve()
    if not project_json.exists():
        raise FileNotFoundError(f"project.json not found: {project_json}")

    project = read_json(project_json)
    missing_keys = [k for k in REQUIRED_PROJECT_KEYS if k not in project]
    if missing_keys:
        raise ValueError(f"Invalid project.json. Missing keys: {missing_keys}")

    paths = project.get("paths", {})
    missing_path_keys = [k for k in REQUIRED_PATH_KEYS if k not in paths]
    if missing_path_keys:
        raise ValueError(f"Invalid project paths. Missing path keys: {missing_path_keys}")

    # Ensure log path exists for logger
    log_path = Path(paths["project_log"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = ProjectLogger(paths["project_log"])
    logger.info("Task 002 project load started.")

    path_status = {k: Path(v).exists() for k, v in paths.items() if isinstance(v, str)}
    critical = ["root", "working_wav", "analysis_dir"]
    missing_critical = [k for k in critical if not path_status.get(k, False)]

    available_analysis = []
    analysis_dir = Path(paths["analysis_dir"])
    if analysis_dir.exists():
        available_analysis = sorted([p.name for p in analysis_dir.glob("*.json")])

    status = "ready" if not missing_critical else "broken"
    result = {
        "task": "Task 002 - Project Loader",
        "status": status,
        "project_id": project["project_id"],
        "project_name": project["project_name"],
        "project_json": str(project_json),
        "working_wav": paths["working_wav"],
        "path_status": path_status,
        "missing_critical": missing_critical,
        "available_analysis": available_analysis,
        "project": project,
    }

    if save_status:
        write_json(analysis_dir / "load_status.json", {
            k: v for k, v in result.items() if k != "project"
        })
        logger.info("Task 002 load_status.json saved.")

    if missing_critical:
        logger.error(f"Project load failed. Missing critical: {missing_critical}")
        raise FileNotFoundError(f"Project is broken. Missing critical paths: {missing_critical}")

    logger.info("Task 002 project load completed. Status ready.")
    return result
