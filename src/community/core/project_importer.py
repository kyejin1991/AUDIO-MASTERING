from pathlib import Path
from datetime import datetime
import json
import re
import shutil

from .audio_formats import require_supported_audio
from .checksum import sha256_file
from .ffmpeg_tools import ffprobe_json, convert_to_working_wav, require_ffmpeg
from .project_logger import ProjectLogger
from .project_schema import AudioImportProject, ProjectPaths, now_string

def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-z가-힣_-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "audio_project"

def make_project_id(input_path: Path, checksum: str, project_name: str | None = None) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = slugify(project_name if project_name else input_path.stem)
    return f"{stamp}_{base}_{checksum[:8]}"

def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

def extract_basic_stream_summary(ffprobe_data: dict) -> dict:
    audio_streams = [s for s in ffprobe_data.get("streams", []) if s.get("codec_type") == "audio"]
    first = audio_streams[0] if audio_streams else {}
    fmt = ffprobe_data.get("format", {})
    return {
        "format_name": fmt.get("format_name"),
        "duration": fmt.get("duration"),
        "size": fmt.get("size"),
        "bit_rate": fmt.get("bit_rate"),
        "codec_name": first.get("codec_name"),
        "codec_long_name": first.get("codec_long_name"),
        "sample_rate": first.get("sample_rate"),
        "channels": first.get("channels"),
        "channel_layout": first.get("channel_layout"),
        "bits_per_sample": first.get("bits_per_sample"),
        "sample_fmt": first.get("sample_fmt"),
    }

def validate_import_outputs(project_dict: dict, working_ffprobe: dict) -> dict:
    paths = project_dict["paths"]
    required_paths = ["original_audio", "working_wav", "project_json", "original_ffprobe_json", "working_ffprobe_json", "project_log"]
    path_checks = {key: Path(paths[key]).exists() for key in required_paths}
    audio_streams = [s for s in working_ffprobe.get("streams", []) if s.get("codec_type") == "audio"]
    stream = audio_streams[0] if audio_streams else {}
    sample_rate_ok = str(stream.get("sample_rate")) == str(project_dict["working_sample_rate"])
    channels_ok = int(stream.get("channels", 0) or 0) == int(project_dict["working_channels"])
    codec_ok = stream.get("codec_name") in {"pcm_f32le", "pcm_f64le"} or stream.get("sample_fmt") in {"flt", "dbl"}
    return {
        "required_paths_exist": path_checks,
        "all_required_paths_exist": all(path_checks.values()),
        "working_sample_rate_ok": sample_rate_ok,
        "working_channels_ok": channels_ok,
        "working_float_pcm_ok": codec_ok,
        "working_stream": {
            "codec_name": stream.get("codec_name"),
            "sample_fmt": stream.get("sample_fmt"),
            "sample_rate": stream.get("sample_rate"),
            "channels": stream.get("channels"),
            "channel_layout": stream.get("channel_layout"),
        }
    }

def import_audio_project(
    input_path: str | Path,
    projects_dir: str | Path = "projects",
    project_name: str | None = None,
    working_sample_rate: int = 48000,
    working_channels: int = 2,
    overwrite: bool = False,
) -> dict:
    require_ffmpeg()
    input_path = Path(input_path).expanduser().resolve()
    projects_dir = Path(projects_dir).expanduser().resolve()
    require_supported_audio(input_path)

    checksum = sha256_file(input_path)
    project_id = make_project_id(input_path, checksum, project_name)
    root = projects_dir / project_id
    if root.exists() and not overwrite:
        raise FileExistsError(f"Project already exists: {root}")

    original_dir = root / "original"
    working_dir = root / "working"
    analysis_dir = root / "analysis"
    logs_dir = root / "logs"
    for d in [original_dir, working_dir, analysis_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log = ProjectLogger(logs_dir / "project.log")
    log.info("Task 001 import started.")
    log.info(f"Input path: {input_path}")
    log.info(f"SHA256: {checksum}")

    original_audio = original_dir / input_path.name
    shutil.copy2(input_path, original_audio)
    log.info(f"Original copied to: {original_audio}")

    original_ffprobe = ffprobe_json(original_audio)
    write_json(analysis_dir / "original_ffprobe.json", original_ffprobe)

    working_wav = working_dir / f"{slugify(input_path.stem)}_48k_float.wav"
    convert_to_working_wav(original_audio, working_wav, sample_rate=working_sample_rate, channels=working_channels)

    working_ffprobe = ffprobe_json(working_wav)
    write_json(analysis_dir / "working_ffprobe.json", working_ffprobe)

    project = AudioImportProject(
        project_id=project_id,
        project_name=project_name or input_path.stem,
        created_at=now_string(),
        source_filename=input_path.name,
        source_extension=input_path.suffix.lower(),
        source_sha256=checksum,
        working_sample_rate=working_sample_rate,
        working_channels=working_channels,
        working_format="wav_pcm_f32le",
        paths=ProjectPaths(
            root=str(root), original_dir=str(original_dir), working_dir=str(working_dir),
            analysis_dir=str(analysis_dir), logs_dir=str(logs_dir),
            original_audio=str(original_audio), working_wav=str(working_wav),
            project_json=str(root / "project.json"),
            import_report_json=str(analysis_dir / "import_report.json"),
            original_ffprobe_json=str(analysis_dir / "original_ffprobe.json"),
            working_ffprobe_json=str(analysis_dir / "working_ffprobe.json"),
            project_log=str(logs_dir / "project.log"),
        )
    )
    project.save()

    import_report = {
        "task": "Task 001 - Project Import",
        "status": "success",
        "project_id": project_id,
        "project_root": str(root),
        "source": {
            "path": str(input_path), "filename": input_path.name, "extension": input_path.suffix.lower(),
            "sha256": checksum, "summary": extract_basic_stream_summary(original_ffprobe)
        },
        "working": {
            "path": str(working_wav), "required_sample_rate": working_sample_rate,
            "required_channels": working_channels, "required_codec": "pcm_f32le",
            "summary": extract_basic_stream_summary(working_ffprobe)
        },
        "validation": validate_import_outputs(project.to_dict(), working_ffprobe),
    }
    write_json(analysis_dir / "import_report.json", import_report)
    log.info("Task 001 import completed successfully.")
    return import_report

