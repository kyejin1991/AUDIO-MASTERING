from __future__ import annotations
import numpy as np
import torch
from .chunker import Chunker
from .gpu_utils import resolve_device
from .inference_config import InferenceConfig
from .normalize import peak_normalize, standardize_audio, restore_audio
from .overlap_add import OverlapAdd
from community.stem_models.weight_loader import load_stem_model


class InternalSeparatorEngine:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.device = resolve_device(config.device)
        self.model, self.model_config = load_stem_model(config.model, device=self.device, strict=config.strict_weights)

    def _run_chunk(self, chunk: np.ndarray) -> dict[str, np.ndarray]:
        tensor = torch.from_numpy(chunk.T[None, ...].astype(np.float32)).to(self.device)
        with torch.inference_mode():
            stems = self.model(tensor)
        out = {}
        for name, stem in stems.items():
            arr = stem.detach().cpu().numpy()[0].T.astype(np.float32)
            out[name] = arr
        return out

    def separate(self, audio: np.ndarray, sample_rate: int) -> tuple[dict[str, np.ndarray], dict]:
        normalized, stats = standardize_audio(audio)
        chunker = Chunker(sample_rate, self.config.segment_seconds, self.config.overlap)
        chunks = chunker.split(normalized)
        separated_chunks = []
        for chunk in chunks:
            separated_chunks.append((self._run_chunk(chunk.audio), chunk.start, chunk.end))
        merged = OverlapAdd(total_samples=audio.shape[0], channels=2).merge(separated_chunks)

        restored = {}
        for name, stem in merged.items():
            stem = restore_audio(stem, stats)
            stem, gain = peak_normalize(stem, self.config.peak_dbfs)
            restored[name] = stem

        report = {
            "backend": "internal_neural",
            "model": self.config.model,
            "device": self.device,
            "sample_rate": sample_rate,
            "segment_seconds": self.config.segment_seconds,
            "overlap": self.config.overlap,
            "chunks": len(chunks),
            "sources": list(restored.keys()),
            "model_untrained": bool(getattr(self.model_config, "untrained", True)),
            "note": "Task 031C internal neural path verifies architecture and chunked inference. Separation quality requires trained weights.",
        }
        return restored, report



