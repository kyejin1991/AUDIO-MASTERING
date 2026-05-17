import json
from pathlib import Path

def load_shared_config(name: str) -> dict:
    """Loads a JSON config from src/shared/config/."""
    # Find src root
    current = Path(__file__).resolve()
    # Path: src/shared/utils/config_loader.py -> up 3 levels to src/
    src_root = current.parents[2]
    config_path = src_root / "shared" / "config" / f"{name}.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Shared config not found: {config_path}")
        
    return json.loads(config_path.read_text(encoding="utf-8"))

def load_config(name: str) -> dict:
    """Unified loader that tries shared config."""
    return load_shared_config(name)
