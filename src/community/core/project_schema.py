from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json

@dataclass
class ProjectPaths:
    root: str
    original_dir: str
    working_dir: str
    analysis_dir: str
    logs_dir: str
    original_audio: str
    working_wav: str
    project_json: str
    import_report_json: str
    original_ffprobe_json: str
    working_ffprobe_json: str
    project_log: str

@dataclass
class AudioImportProject:
    project_id: str
    project_name: str
    created_at: str
    source_filename: str
    source_extension: str
    source_sha256: str
    working_sample_rate: int
    working_channels: int
    working_format: str
    paths: ProjectPaths

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self) -> Path:
        path = Path(self.paths.project_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        return path

def now_string() -> str:
    return datetime.now().isoformat(timespec="seconds")

