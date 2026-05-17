from __future__ import annotations
from pathlib import Path
import hashlib
import json
import torch
from .model_config import StemModelConfig, default_4stem_config, default_6stem_config
from .htdemucs_model import HybridStemSeparator


WEIGHTS_DIR = Path(__file__).resolve().parents[1] / "model_weights"


def load_manifest(path: str | Path | None = None) -> dict:
    manifest_path = Path(path) if path else WEIGHTS_DIR / "model_manifest.json"
    if not manifest_path.exists():
        return {"models": {}}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config_from_manifest(model_name: str, manifest: dict) -> StemModelConfig:
    models = manifest.get("models", {})
    if model_name in models:
        return StemModelConfig.from_dict(models[model_name].get("config", models[model_name]))
    if model_name.endswith("6stem"):
        return default_6stem_config(model_name=model_name)
    return default_4stem_config(model_name=model_name)


def load_stem_model(model_name: str = "arc_internal_dummy_4stem", device: str = "cpu", strict: bool = True):
    manifest = load_manifest()
    entry = manifest.get("models", {}).get(model_name)
    config = config_from_manifest(model_name, manifest)
    model = HybridStemSeparator(config)

    if entry and entry.get("file"):
        weight_path = WEIGHTS_DIR / entry["file"]
        if not weight_path.exists():
            raise FileNotFoundError(f"Stem model weight file is missing: {weight_path}")
        expected = entry.get("sha256")
        if expected:
            actual = sha256_file(weight_path)
            if actual.lower() != expected.lower():
                raise ValueError(f"Checksum mismatch for {weight_path.name}: expected {expected}, got {actual}")
        state = torch.load(weight_path, map_location=device)
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=strict)
        model.config.untrained = False

    model.to(device)
    model.eval()
    return model, config


