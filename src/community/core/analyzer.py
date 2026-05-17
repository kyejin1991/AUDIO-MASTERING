from __future__ import annotations
from pathlib import Path
from community.core.audio_reader import read_working_wav
from community.core.project_loader import load_project, read_json

def analyze_project(project_json: str | Path) -> dict:
    from community.analysis.run_analysis import run_full_analysis
    return run_full_analysis(project_json)

def load_analysis_bundle(project_json: str | Path) -> dict:
    project = load_project(project_json, save_status=True)["project"]
    analysis_dir = Path(project["paths"]["analysis_dir"])
    names = [
        "full_analysis.json",
        "basic_audio.json",
        "loudness.json",
        "spectrum.json",
        "genre_match.json",
        "stereo.json",
        "dynamics.json",
    ]
    return {
        name.replace(".json", ""): read_json(analysis_dir / name)
        for name in names
        if (analysis_dir / name).exists()
    }

def analyze_audio_file(audio_path: str | Path) -> dict:
    from community.analysis.basic_audio import analyze_basic_audio
    from community.analysis.loudness import analyze_loudness
    from community.analysis.spectrum import analyze_spectrum
    from community.analysis.stereo import analyze_stereo
    from community.analysis.dynamics import analyze_dynamics
    
    audio, sr, info = read_working_wav(audio_path)
    return {
        "audio_reader_info": info,
        "basic_audio": analyze_basic_audio(audio, sr),
        "loudness": analyze_loudness(audio, sr),
        "spectrum": analyze_spectrum(audio, sr),
        "stereo": analyze_stereo(audio, sr),
        "dynamics": analyze_dynamics(audio, sr),
    }
