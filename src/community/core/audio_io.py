from __future__ import annotations
from pathlib import Path
import tempfile

from community.core.audio_formats import SUPPORTED_AUDIO_EXTENSIONS, is_supported_audio, require_supported_audio
from community.core.audio_reader import read_working_wav, save_audio_reader_info, sanitize_audio, ensure_stereo
from community.core.project_importer import import_audio_project


def supported_upload_extensions() -> list[str]:
    return sorted(SUPPORTED_AUDIO_EXTENSIONS)


def create_project_from_audio(
    input_path: str | Path,
    projects_dir: str | Path,
    project_name: str | None = None,
    working_sample_rate: int = 48000,
    working_channels: int = 2,
) -> dict:
    require_supported_audio(input_path)
    return import_audio_project(
        input_path=input_path,
        projects_dir=projects_dir,
        project_name=project_name,
        working_sample_rate=working_sample_rate,
        working_channels=working_channels,
    )


def persist_uploaded_bytes(filename: str, data: bytes, temp_dir: str | Path) -> Path:
    temp_dir = Path(temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(prefix="upload_", suffix=suffix, delete=False, dir=temp_dir) as tmp:
        tmp.write(data)
        return Path(tmp.name)

