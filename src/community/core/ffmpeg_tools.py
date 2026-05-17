from pathlib import Path
import json
import shutil
from .shell import run_command

def require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise EnvironmentError("ffmpeg was not found in PATH.")
    if shutil.which("ffprobe") is None:
        raise EnvironmentError("ffprobe was not found in PATH.")

def ffprobe_json(path: str | Path) -> dict:
    require_ffmpeg()
    result = run_command([
        "ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)
    ])
    return json.loads(result.stdout)

def convert_to_working_wav(input_path: str | Path, output_path: str | Path, sample_rate: int = 48000, channels: int = 2) -> Path:
    require_ffmpeg()
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_command([
        "ffmpeg", "-hide_banner", "-y", "-i", str(input_path),
        "-vn", "-ac", str(channels), "-ar", str(sample_rate),
        "-c:a", "pcm_f32le", str(output_path)
    ])
    return output_path

