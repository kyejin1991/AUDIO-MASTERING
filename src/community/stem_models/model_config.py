from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class StemModelConfig:
    model_name: str = "arc_internal_dummy_4stem"
    sources: tuple[str, ...] = ("vocals", "drums", "bass", "music")
    sample_rate: int = 44100
    channels: int = 2
    hidden_channels: int = 32
    depth: int = 3
    n_fft: int = 1024
    hop_length: int = 256
    transformer_heads: int = 4
    transformer_layers: int = 2
    segment_seconds: float = 7.8
    overlap: float = 0.25
    untrained: bool = True

    def to_dict(self) -> dict:
        data = asdict(self)
        data["sources"] = list(self.sources)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "StemModelConfig":
        clean = dict(data)
        if "sources" in clean:
            clean["sources"] = tuple(clean["sources"])
        return cls(**{k: v for k, v in clean.items() if k in cls.__dataclass_fields__})


def default_4stem_config(**overrides) -> StemModelConfig:
    data = {
        "model_name": "arc_internal_dummy_4stem",
        "sources": ("vocals", "drums", "bass", "music"),
        "hidden_channels": 32,
        "untrained": True,
    }
    data.update(overrides)
    return StemModelConfig(**data)


def default_6stem_config(**overrides) -> StemModelConfig:
    data = {
        "model_name": "arc_internal_dummy_6stem",
        "sources": ("vocals", "drums", "bass", "guitar", "piano", "music"),
        "hidden_channels": 32,
        "untrained": True,
    }
    data.update(overrides)
    return StemModelConfig(**data)


