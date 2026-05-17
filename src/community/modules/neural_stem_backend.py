from __future__ import annotations
from pathlib import Path
import shutil
import subprocess
import sys
import json
import os

SUPPORTED_DEMUCS_MODELS = [
    "htdemucs",
    "htdemucs_ft",
    "htdemucs_6s",
    "hdemucs_mmi",
    "mdx",
    "mdx_extra",
    "mdx_q",
    "mdx_extra_q",
]

STANDARD_STEMS = {
    "vocals": "vocals.wav",
    "drums": "drums.wav",
    "bass": "bass.wav",
    "other": "music.wav",
    "music": "music.wav",
    "guitar": "guitar.wav",
    "piano": "piano.wav",
}

class DemucsBackend:
    def __init__(
        self,
        model: str = "htdemucs",
        device: str = "auto",
        shifts: int = 1,
        overlap: float = 0.25,
        clip_mode: str = "rescale",
        jobs: int = 0,
        two_stems: str | None = None,
    ):
        if model not in SUPPORTED_DEMUCS_MODELS:
            model = "htdemucs"
        self.model = model
        self.device = device if device in {"auto", "cpu", "cuda"} else "auto"
        self.shifts = max(1, int(shifts))
        self.overlap = max(0.0, min(0.99, float(overlap)))
        self.clip_mode = clip_mode if clip_mode in {"rescale", "clamp", "none"} else "rescale"
        self.jobs = max(0, int(jobs))
        self.two_stems = two_stems if two_stems in {"vocals", "drums", "bass", "other"} else None

    def is_available(self) -> tuple[bool, str]:
        exe = shutil.which("demucs")
        if exe:
            return True, exe

        # Also accept python -m demucs.separate if package is installed.
        try:
            proc = subprocess.run(
                [sys.executable, "-m", "demucs.separate", "--help"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            if proc.returncode == 0:
                return True, f"{sys.executable} -m demucs.separate"
        except Exception as e:
            return False, str(e)

        return False, "demucs executable/module not found"

    def build_command(self, input_audio: str | Path, output_dir: str | Path) -> list[str]:
        input_audio = Path(input_audio)
        output_dir = Path(output_dir)

        exe = shutil.which("demucs")
        if exe:
            cmd = [exe]
        else:
            cmd = [sys.executable, "-m", "demucs.separate"]

        cmd += [
            "-n", self.model,
            "-o", str(output_dir),
            "--shifts", str(self.shifts),
            "--overlap", str(self.overlap),
            "--clip-mode", self.clip_mode,
        ]

        if self.device in {"cpu", "cuda"}:
            cmd += ["-d", self.device]
        if self.jobs > 0:
            cmd += ["-j", str(self.jobs)]
        if self.two_stems:
            cmd += ["--two-stems", self.two_stems]

        cmd += [str(input_audio)]
        return cmd

    def locate_output_dir(self, output_root: str | Path, input_audio: str | Path) -> Path | None:
        output_root = Path(output_root)
        stem_name = Path(input_audio).stem
        candidates = [
            output_root / self.model / stem_name,
            output_root / stem_name,
        ]
        for c in candidates:
            if c.exists():
                return c

        # Last resort: find any folder that contains common stem wavs.
        for p in output_root.rglob("*"):
            if p.is_dir() and any((p / name).exists() for name in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]):
                return p
        return None

    def normalize_outputs(self, demucs_stem_dir: Path, project_stems_dir: Path) -> dict:
        project_stems_dir.mkdir(parents=True, exist_ok=True)
        copied = {}
        for src_name, dest_name in STANDARD_STEMS.items():
            src = demucs_stem_dir / f"{src_name}.wav"
            if src.exists():
                dest = project_stems_dir / dest_name
                shutil.copy2(src, dest)
                copied[src_name] = str(dest)

        # Ensure other/music alias.
        if "other" in copied and "music" not in copied:
            copied["music"] = copied["other"]
        if "music" in copied and "other" not in copied:
            copied["other"] = copied["music"]

        return copied

    def run(self, input_audio: str | Path, output_root: str | Path, project_stems_dir: str | Path, timeout_sec: int | None = None) -> dict:
        input_audio = Path(input_audio)
        output_root = Path(output_root)
        project_stems_dir = Path(project_stems_dir)
        output_root.mkdir(parents=True, exist_ok=True)

        available, available_detail = self.is_available()
        if not available:
            return {
                "status": "unavailable",
                "backend": "demucs",
                "available": False,
                "available_detail": available_detail,
                "error": "Demucs is not installed or not available on PATH.",
            }

        cmd = self.build_command(input_audio, output_root)
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except Exception as e:
            return {
                "status": "error",
                "backend": "demucs",
                "available": True,
                "available_detail": available_detail,
                "command": cmd,
                "error": str(e),
            }

        if proc.returncode != 0:
            return {
                "status": "error",
                "backend": "demucs",
                "available": True,
                "available_detail": available_detail,
                "command": cmd,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
                "error": "Demucs command failed.",
            }

        stem_dir = self.locate_output_dir(output_root, input_audio)
        if stem_dir is None:
            return {
                "status": "error",
                "backend": "demucs",
                "available": True,
                "available_detail": available_detail,
                "command": cmd,
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
                "error": "Demucs output stem directory not found.",
            }

        copied = self.normalize_outputs(stem_dir, project_stems_dir)
        return {
            "status": "success",
            "backend": "demucs",
            "available": True,
            "available_detail": available_detail,
            "model": self.model,
            "device": self.device,
            "shifts": self.shifts,
            "overlap": self.overlap,
            "clip_mode": self.clip_mode,
            "jobs": self.jobs,
            "two_stems": self.two_stems,
            "command": cmd,
            "returncode": proc.returncode,
            "demucs_output_dir": str(stem_dir),
            "copied_stems": copied,
            "stdout_tail": proc.stdout[-2000:],
            "stderr_tail": proc.stderr[-2000:],
        }

def backend_config_from_params(params: dict | None):
    params = params or {}
    return {
        "backend": params.get("backend", "auto"),
        "internal_model": params.get("internal_model", params.get("model", "arc_internal_dummy_4stem")),
        "demucs_model": params.get("demucs_model", params.get("model", "htdemucs")),
        "device": params.get("device", "auto"),
        "segment_seconds": float(params.get("segment_seconds", 7.8)),
        "shifts": int(params.get("shifts", 1)),
        "overlap": float(params.get("overlap", 0.25)),
        "clip_mode": params.get("clip_mode", "rescale"),
        "jobs": int(params.get("jobs", 0)),
        "two_stems": params.get("two_stems"),
        "timeout_sec": params.get("timeout_sec"),
    }


