from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class AudioChunk:
    audio: np.ndarray
    start: int
    end: int
    index: int


class Chunker:
    def __init__(self, sample_rate: int, segment_seconds: float = 7.8, overlap: float = 0.25):
        self.sample_rate = int(sample_rate)
        self.segment_samples = max(1024, int(round(float(segment_seconds) * self.sample_rate)))
        self.overlap = max(0.0, min(0.95, float(overlap)))
        self.hop_samples = max(1, int(round(self.segment_samples * (1.0 - self.overlap))))

    def split(self, audio: np.ndarray) -> list[AudioChunk]:
        total = audio.shape[0]
        chunks = []
        start = 0
        index = 0
        while start < total:
            end = min(total, start + self.segment_samples)
            chunk = audio[start:end]
            if chunk.shape[0] < self.segment_samples:
                pad = self.segment_samples - chunk.shape[0]
                chunk = np.pad(chunk, ((0, pad), (0, 0)))
            chunks.append(AudioChunk(chunk.astype(np.float32), start, end, index))
            index += 1
            if end == total:
                break
            start += self.hop_samples
        return chunks


