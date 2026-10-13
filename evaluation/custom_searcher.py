from __future__ import annotations

import json
from pathlib import Path


class CustomEvalRetriever:
    """Loads precomputed retrieval results, as CuRLRank's custom retriever does."""

    @staticmethod
    def load(path: Path) -> dict[str, dict[str, float]]:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("search_results", data)
