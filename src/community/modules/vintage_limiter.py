from __future__ import annotations

from community.modules.dsp_utils import sanitize, soft_limiter


def process_vintage_limiter(audio, sr: int, params: dict | None = None):
    params = params or {}
    ceiling_db = float(params.get("ceiling_db", -1.0))
    limited, reduction_db = soft_limiter(audio, ceiling_db=ceiling_db)
    out = sanitize(limited)
    return out, {
        "task": "Task 034 - Vintage Limiter",
        "ceiling_db": ceiling_db,
        "gain_reduction_db": reduction_db,
    }



