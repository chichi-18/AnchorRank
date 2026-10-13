from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import prompts
from .json_utils import json_object
from .types import Document, TextGenerator


class AnchorBuilder:
    """Builds each query anchor once and optionally persists it in a JSON cache."""

    def __init__(self, generator: TextGenerator, strategy: str = "ae", top_k: int = 40, chunk_size: int = 10, cache_path: str | None = None):
        if strategy not in {"ae", "trd", "ga"}:
            raise ValueError("strategy must be one of: ae, trd, ga")
        self.generator, self.strategy = generator, strategy
        self.top_k, self.chunk_size = top_k, chunk_size
        self.cache_path = Path(cache_path) if cache_path else None
        self._cache: dict[str, str] = self._load_cache()

    def build(self, query: str, documents: list[Document]) -> str:
        key = self._key(query, documents)
        if key in self._cache:
            return self._cache[key]
        if not documents:
            anchor = query
        elif self.strategy == "trd":
            anchor = documents[0].content
        elif self.strategy == "ga":
            anchor = self.generator.generate([prompts.generated_answer(query)], max_new_tokens=512)[0].strip()
        else:
            anchor = self._aggregated_evidence(query, documents[: self.top_k])
        self._cache[key] = anchor
        self._save_cache()
        return anchor

    def _aggregated_evidence(self, query: str, documents: list[Document]) -> str:
        chunks = [documents[i:i + self.chunk_size] for i in range(0, len(documents), self.chunk_size)]
        answers = self.generator.generate([prompts.evidence_extraction(query, chunk) for chunk in chunks], max_new_tokens=768)
        evidence: list[dict[str, str]] = []
        for answer in answers:
            payload = json_object(answer) or {}
            for item in payload.get("evidence", []):
                if isinstance(item, dict) and isinstance(item.get("doc_id"), str) and isinstance(item.get("evidence"), str):
                    evidence.append({"doc_id": item["doc_id"], "evidence": item["evidence"]})
        if not evidence:
            return documents[0].content
        answer = self.generator.generate([prompts.evidence_aggregation(query, evidence)], max_new_tokens=768)[0]
        anchor = (json_object(answer) or {}).get("anchor")
        return anchor.strip() if isinstance(anchor, str) and anchor.strip() else "\n".join(item["evidence"] for item in evidence)

    def _key(self, query: str, documents: list[Document]) -> str:
        content = json.dumps([self.strategy, query, [(d.doc_id, d.content) for d in documents[:self.top_k]]], ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _load_cache(self) -> dict[str, str]:
        if self.cache_path and self.cache_path.is_file():
            with self.cache_path.open(encoding="utf-8") as file:
                data = json.load(file)
            return data if isinstance(data, dict) else {}
        return {}

    def _save_cache(self) -> None:
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(self._cache, ensure_ascii=False, indent=2), encoding="utf-8")
