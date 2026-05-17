from __future__ import annotations
import torch


def resolve_device(requested: str = "cpu") -> str:
    requested = (requested or "cpu").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return requested if requested in {"cpu", "cuda"} else "cpu"


