from __future__ import annotations

from . import prompts
from .anchor import AnchorBuilder
from .json_utils import bounded_score, json_object
from .types import Document, QueryExample, TextGenerator


class AnchorReranker:
    def __init__(self, generator: TextGenerator, anchor_builder: AnchorBuilder, mode: str = "listwise", window_size: int = 10):
        if mode not in {"pointwise", "listwise"}:
            raise ValueError("mode must be 'pointwise' or 'listwise'")
        self.generator, self.anchor_builder = generator, anchor_builder
        self.mode, self.window_size = mode, window_size

    def rerank(self, example: QueryExample) -> tuple[str, list[tuple[Document, float]]]:
        docs = sorted(example.documents, key=lambda doc: doc.retrieval_score, reverse=True)
        anchor = self.anchor_builder.build(example.query, docs)
        scores = self._pointwise(example.query, anchor, docs) if self.mode == "pointwise" else self._listwise(example.query, anchor, docs)
        return anchor, sorted(zip(docs, scores), key=lambda item: item[1], reverse=True)

    def _pointwise(self, query: str, anchor: str, docs: list[Document]) -> list[float]:
        outputs = self.generator.generate([prompts.pointwise(query, anchor, doc) for doc in docs], max_new_tokens=96)
        return [float(bounded_score((json_object(output) or {}).get("score")) or 0) for output in outputs]

    def _listwise(self, query: str, anchor: str, docs: list[Document]) -> list[float]:
        scores: dict[str, float] = {}
        for start in range(0, len(docs), self.window_size):
            window = docs[start:start + self.window_size]
            answer = self.generator.generate([prompts.listwise(query, anchor, window)], max_new_tokens=256)[0]
            payload = json_object(answer) or {}
            for doc in window:
                score = bounded_score(payload.get(doc.doc_id))
                if score is not None:
                    scores[doc.doc_id] = float(score)
        # A malformed listwise answer should not silently reorder a retrieval result.
        return [scores.get(doc.doc_id, doc.retrieval_score) for doc in docs]
