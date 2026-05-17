from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = {
    ".wav", ".wave", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus", ".aiff", ".aif"
}

def is_supported_audio(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS

def require_supported_audio(path: str | Path) -> None:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")
    if not is_supported_audio(path):
        supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"Unsupported audio extension '{path.suffix}'. Supported: {supported}")

