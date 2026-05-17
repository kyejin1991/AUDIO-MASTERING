from __future__ import annotations

from pathlib import Path

from community.core.ffmpeg_tools import require_ffmpeg
from community.core.shell import run_command


DEFAULT_YOUTUBE_TEMPLATE = Path(r"D:\SING VIDEO MACHINE\youtube\base_video.mp4")
DEFAULT_YOUTUBE_OUTPUT_DIR = Path(r"D:\SING VIDEO MACHINE\youtube\mastered_videos")
DEFAULT_VISUAL_ASSET_DIR = Path(r"D:\SING VIDEO MACHINE\PNG ASSET")

VIDEO_TEMPLATE_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm"}
IMAGE_TEMPLATE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def list_visual_templates(asset_dir: str | Path | None = None) -> list[Path]:
    root = Path(asset_dir) if asset_dir else DEFAULT_VISUAL_ASSET_DIR
    if not root.exists():
        return []
    allowed = VIDEO_TEMPLATE_EXTENSIONS | IMAGE_TEMPLATE_EXTENSIONS
    return sorted(
        [path.resolve() for path in root.iterdir() if path.is_file() and path.suffix.lower() in allowed],
        key=lambda path: path.name.lower(),
    )


def recommend_visual_template(genre: str | None = None, asset_dir: str | Path | None = None) -> Path:
    assets = list_visual_templates(asset_dir)
    if not assets:
        return DEFAULT_YOUTUBE_TEMPLATE

    genre_key = (genre or "").strip().lower()
    preferred_names = {
        "edm": ["edm_video.png", "edm_video.mp4"],
        "hiphop": ["hiphop_video.mp4", "hiphop_video.png"],
    }
    for candidate in preferred_names.get(genre_key, []):
        for asset in assets:
            if asset.name.lower() == candidate.lower():
                return asset
    for asset in assets:
        if asset.name.lower() == DEFAULT_YOUTUBE_TEMPLATE.name.lower():
            return asset
    return assets[0]


def _build_video_source_command(visual_template_path: Path) -> list[str]:
    suffix = visual_template_path.suffix.lower()
    if suffix in VIDEO_TEMPLATE_EXTENSIONS:
        return [
            "-stream_loop",
            "-1",
            "-i",
            str(visual_template_path),
        ]
    if suffix in IMAGE_TEMPLATE_EXTENSIONS:
        return [
            "-loop",
            "1",
            "-framerate",
            "30",
            "-i",
            str(visual_template_path),
        ]
    raise ValueError(f"Unsupported visual template type: {visual_template_path.suffix}")


def _build_video_encode_command(visual_template_path: Path) -> list[str]:
    suffix = visual_template_path.suffix.lower()
    if suffix in VIDEO_TEMPLATE_EXTENSIONS:
        return ["-c:v", "copy"]
    return [
        "-vf",
        "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:black,format=yuv420p",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
    ]


def render_mastered_video(
    audio_path: str | Path,
    video_template_path: str | Path,
    output_dir: str | Path,
    output_name: str | None = None,
    overwrite: bool = True,
) -> dict:
    require_ffmpeg()
    audio_path = Path(audio_path)
    video_template_path = Path(video_template_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not audio_path.exists():
        raise FileNotFoundError(f"Master audio not found: {audio_path}")
    if not video_template_path.exists():
        raise FileNotFoundError(f"Video template not found: {video_template_path}")

    resolved_name = output_name or f"{audio_path.stem}.mp4"
    output_path = output_dir / resolved_name

    command = [
        "ffmpeg",
        "-hide_banner",
        "-y" if overwrite else "-n",
        *_build_video_source_command(video_template_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-shortest",
        *_build_video_encode_command(video_template_path),
        "-c:a",
        "aac",
        "-b:a",
        "320k",
        str(output_path),
    ]
    run_command(command)
    return {
        "status": "success",
        "audio_path": str(audio_path),
        "video_template_path": str(video_template_path),
        "output_path": str(output_path),
        "template_kind": "image" if video_template_path.suffix.lower() in IMAGE_TEMPLATE_EXTENSIONS else "video",
        "command": command,
    }

