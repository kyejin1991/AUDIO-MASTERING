from __future__ import annotations
import numpy as np
from community.stem_inference import InferenceConfig, InternalSeparatorEngine


def run_internal_neural_backend(audio: np.ndarray, sample_rate: int, params: dict | None = None):
    config = InferenceConfig.from_params(params)
    engine = InternalSeparatorEngine(config)
    stems, report = engine.separate(audio, sample_rate)
    return stems, report



