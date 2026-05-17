from pathlib import Path
from community.core.project_loader import load_project
from community.core.audio_reader import read_working_wav, save_audio_reader_info
from community.core.project_logger import ProjectLogger
from .basic_audio import save_basic_audio_analysis
from .loudness import save_loudness_analysis
from .spectrum import save_spectrum_analysis
from .genre_match import save_genre_match_analysis
from .stereo import save_stereo_analysis
from .dynamics import save_dynamics_analysis
from .aggregator import aggregate_full_analysis

def run_full_analysis(project_json: str | Path) -> dict:
    # Task 002
    project_load = load_project(project_json, save_status=True)
    project = project_load["project"]
    logger = ProjectLogger(project["paths"]["project_log"])
    analysis_dir = project["paths"]["analysis_dir"]

    # Task 003
    logger.info("Task 003 working WAV read started.")
    audio, sr, audio_info = read_working_wav(project["paths"]["working_wav"])
    save_audio_reader_info(audio_info, analysis_dir)
    logger.info(f"Task 003 working WAV read completed and audio_reader_info.json saved: {audio_info}")

    # Task 004~008
    logger.info("Task 004 basic audio analysis started.")
    basic = save_basic_audio_analysis(audio, sr, analysis_dir)
    logger.info("Task 004 basic audio analysis completed.")

    logger.info("Task 005 loudness analysis started.")
    loudness = save_loudness_analysis(audio, sr, analysis_dir)
    logger.info("Task 005 loudness analysis completed.")

    logger.info("Task 006 spectrum analysis started.")
    spectrum = save_spectrum_analysis(audio, sr, analysis_dir)
    logger.info("Task 006 spectrum analysis completed.")

    logger.info("Task 006A genre spectrum match started.")
    genre_match = save_genre_match_analysis(audio, sr, analysis_dir)
    logger.info("Task 006A genre spectrum match completed.")

    logger.info("Task 007 stereo/phase analysis started.")
    stereo = save_stereo_analysis(audio, sr, analysis_dir)
    logger.info("Task 007 stereo/phase analysis completed.")

    logger.info("Task 008 dynamics/transient analysis started.")
    dynamics = save_dynamics_analysis(audio, sr, analysis_dir)
    logger.info("Task 008 dynamics/transient analysis completed.")

    # Task 009
    full = aggregate_full_analysis(project_load, basic, loudness, spectrum, genre_match, stereo, dynamics)
    return full



