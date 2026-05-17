from __future__ import annotations
import numpy as np


class OverlapAdd:
    def __init__(self, total_samples: int, channels: int = 2):
        self.total_samples = int(total_samples)
        self.channels = int(channels)

    def merge(self, separated_chunks: list[tuple[dict[str, np.ndarray], int, int]]) -> dict[str, np.ndarray]:
        if not separated_chunks:
            return {}
        sources = list(separated_chunks[0][0].keys())
        output = {s: np.zeros((self.total_samples, self.channels), dtype=np.float64) for s in sources}
        weight = np.zeros((self.total_samples, 1), dtype=np.float64)

        for stems, start, end in separated_chunks:
            chunk_len = max(1, end - start)
            window = np.hanning(chunk_len * 2)[chunk_len:]
            if chunk_len == 1:
                window = np.ones(1)
            window = window[:, None]
            weight[start:end] += window
            for source in sources:
                output[source][start:end] += stems[source][:chunk_len] * window

        weight = np.maximum(weight, 1e-8)
        return {source: (audio / weight).astype(np.float32) for source, audio in output.items()}


