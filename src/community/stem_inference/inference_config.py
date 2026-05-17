from __future__ import annotations
from dataclasses import dataclass


@dataclass
class InferenceConfig:
    model: str = "arc_internal_dummy_4stem"
    device: str = "cpu"
    segment_seconds: float = 7.8
    overlap: float = 0.25
    sample_rate: int = 44100
    peak_dbfs: float = -1.0
    batch_size: int = 1
    strict_weights: bool = True

    @classmethod
    def from_params(cls, params: dict | None) -> "InferenceConfig":
        params = params or {}
        return cls(
            model=params.get("internal_model", params.get("model", "arc_internal_dummy_4stem")),
            device=params.get("device", "cpu"),
            segment_seconds=float(params.get("segment_seconds", 7.8)),
            overlap=float(params.get("overlap", 0.25)),
            sample_rate=int(params.get("sample_rate", 44100)),
            peak_dbfs=float(params.get("peak_dbfs", -1.0)),
            batch_size=int(params.get("batch_size", 1)),
            strict_weights=bool(params.get("strict_weights", True)),
        )


