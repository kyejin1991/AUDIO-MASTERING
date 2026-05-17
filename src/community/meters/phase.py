from __future__ import annotations

from community.analysis.stereo import analyze_stereo


def analyze_phase(audio, sr: int) -> dict:
    stereo = analyze_stereo(audio, sr)
    return {
        "phase_correlation": stereo["phase_correlation"],
        "mono_collapse_loss_db": stereo["mono_collapse_loss_db"],
    }



