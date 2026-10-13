from __future__ import annotations

import json
import re
from typing import Any


def json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object even when a chat model wraps it in a code fence."""
    text = text.strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, flags=re.I))
    candidates.extend(re.findall(r"(\{[\s\S]*\})", text))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def bounded_score(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        score = int(float(value))
    except (TypeError, ValueError):
        return None
    return min(100, max(0, score))
