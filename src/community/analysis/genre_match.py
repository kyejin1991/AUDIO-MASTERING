from __future__ import annotations

from pathlib import Path
import json

import numpy as np
from scipy import signal

from shared.utils.config_loader import load_shared_config
from community.meters.common import mono


def _load_dataset() -> dict:
    return load_shared_config("genre_spectrum_dataset")


def _band_labels(edges: list[float]) -> list[str]:
    labels = []
    for low, high in zip(edges[:-1], edges[1:]):
        labels.append(f"{int(round(low))}-{int(round(high))}Hz")
    return labels


def _welch(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    x = mono(audio)
    nperseg = min(16384, max(2048, len(x) // 2))
    freqs, pxx = signal.welch(x, fs=sr, nperseg=nperseg)
    return freqs, pxx + 1e-20


def _band_energies(freqs: np.ndarray, pxx: np.ndarray, edges: list[float]) -> np.ndarray:
    energies = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        energies.append(float(np.sum(pxx[mask]) + 1e-20))
    return np.asarray(energies, dtype=np.float64)


def _relative_curve_db(energies: np.ndarray) -> np.ndarray:
    spectral_floor = max(float(np.mean(energies)) * 0.05, 1e-12)
    stabilized = np.maximum(energies, spectral_floor)
    curve_db = 10.0 * np.log10(stabilized)
    kernel = np.asarray([1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0], dtype=np.float64)
    kernel /= np.sum(kernel)
    smoothed = np.convolve(curve_db, kernel, mode="same")
    centered = smoothed - float(np.mean(smoothed))
    return np.clip(centered, -12.0, 12.0)


def _diagonal_mahalanobis(curve_db: np.ndarray, mean_curve_db: np.ndarray, std_curve_db: np.ndarray) -> float:
    safe_std = np.maximum(std_curve_db, 1e-6)
    z = (curve_db - mean_curve_db) / safe_std
    return float(np.sqrt(np.mean(z ** 2)))


def _macro_profile(energies: np.ndarray, edges: list[float]) -> dict[str, float]:
    buckets = {
        "sub": 0.0,
        "bass": 0.0,
        "low_mid": 0.0,
        "mid": 0.0,
        "presence": 0.0,
        "air": 0.0,
    }
    for energy, low, high in zip(energies, edges[:-1], edges[1:]):
        center = (float(low) + float(high)) * 0.5
        if center < 60.0:
            buckets["sub"] += float(energy)
        elif center < 150.0:
            buckets["bass"] += float(energy)
        elif center < 500.0:
            buckets["low_mid"] += float(energy)
        elif center < 2500.0:
            buckets["mid"] += float(energy)
        elif center < 8000.0:
            buckets["presence"] += float(energy)
        else:
            buckets["air"] += float(energy)
    total = sum(buckets.values()) + 1e-20
    return {name: float(value / total) for name, value in buckets.items()}


def _macro_distance(current_macro: dict[str, float], target_macro: dict[str, float]) -> float:
    keys = ["sub", "bass", "low_mid", "mid", "presence", "air"]
    return float(np.sqrt(np.mean([(current_macro[key] - target_macro[key]) ** 2 for key in keys])))


def _softmax_confidence(distances: dict[str, float]) -> tuple[dict[str, float], float]:
    genres = list(distances.keys())
    dist_array = np.asarray([distances[name] for name in genres], dtype=np.float64)
    logits = -dist_array
    logits -= np.max(logits)
    weights = np.exp(logits)
    probs = weights / np.sum(weights)
    confidence = {name: round(float(prob), 6) for name, prob in zip(genres, probs)}
    sorted_probs = sorted(confidence.values(), reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    return confidence, round(float(margin), 6)


def analyze_genre_match(audio: np.ndarray, sr: int) -> dict:
    dataset = _load_dataset()
    edges = [float(value) for value in dataset["band_edges_hz"]]
    labels = _band_labels(edges)
    freqs, pxx = _welch(audio, sr)
    energies = _band_energies(freqs, pxx, edges)
    curve_db = _relative_curve_db(energies)
    current_macro = _macro_profile(energies, edges)

    distances = {}
    distance_components = {}
    for genre, profile in dataset["genres"].items():
        mean_curve = np.asarray(profile["mean_curve_db"], dtype=np.float64)
        std_curve = np.asarray(profile["std_curve_db"], dtype=np.float64)
        mahalanobis = _diagonal_mahalanobis(curve_db, mean_curve, std_curve)
        target_macro = _macro_profile(np.power(10.0, mean_curve / 10.0), edges)
        macro_distance = _macro_distance(current_macro, target_macro)
        combined = 0.72 * mahalanobis + 9.5 * macro_distance
        distances[genre] = combined
        distance_components[genre] = {
            "mahalanobis": round(float(mahalanobis), 6),
            "macro_distance": round(float(macro_distance), 6),
            "combined_distance": round(float(combined), 6),
            "target_macro_profile": {name: round(float(value), 6) for name, value in target_macro.items()},
        }

    sorted_matches = sorted(distances.items(), key=lambda item: item[1])
    confidence, margin = _softmax_confidence(distances)
    inferred_genre, best_distance = sorted_matches[0]

    return {
        "task": "Task 001A - Genre Spectrum Match",
        "status": "success",
        "method": "diagonal_mahalanobis_with_macro_energy_prior",
        "dataset_version": dataset["version"],
        "band_edges_hz": edges,
        "band_labels": labels,
        "band_curve_db": [round(float(value), 6) for value in curve_db],
        "band_energy": [round(float(value), 12) for value in energies],
        "macro_profile": {name: round(float(value), 6) for name, value in current_macro.items()},
        "inferred_genre": inferred_genre,
        "best_distance": round(float(best_distance), 6),
        "confidence": confidence,
        "confidence_margin": margin,
        "distance_components": distance_components,
        "top_matches": [
            {
                "genre": genre,
                "distance": round(float(distance), 6),
                "confidence": confidence[genre],
            }
            for genre, distance in sorted_matches[:5]
        ],
        "distance_map": {genre: round(float(distance), 6) for genre, distance in sorted_matches},
    }


def save_genre_match_analysis(audio: np.ndarray, sr: int, analysis_dir: str | Path) -> dict:
    report = analyze_genre_match(audio, sr)
    path = Path(analysis_dir) / "genre_match.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report



