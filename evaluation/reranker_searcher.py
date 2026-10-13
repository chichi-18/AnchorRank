from __future__ import annotations

from typing import Any

from anchorrank import Document


class AnchorRankEvalReranker:
    """Consumes retrieval runs and replaces their scores with AnchorRank scores."""

    def __init__(self, reranker, rerank_top_k: int = 100):
        self.reranker = reranker
        self.rerank_top_k = rerank_top_k

    def __str__(self) -> str:
        return self.reranker.model_name_or_path.rsplit("/", 1)[-1]

    def __call__(self, corpus: dict[str, dict[str, Any]], queries: dict[str, str], search_results: dict[str, dict[str, float]], **_: Any):
        reranked: dict[str, dict[str, float]] = {}
        anchors: dict[str, str] = {}
        for qid, retrieved in search_results.items():
            docs = [Document(str(docid), corpus[str(docid)]["text"], float(score)) for docid, score in sorted(retrieved.items(), key=lambda item: item[1], reverse=True)[:self.rerank_top_k] if str(docid) in corpus]
            if not docs or qid not in queries:
                continue
            anchor, ranked = self.reranker.rerank(str(qid), queries[qid], docs)
            anchors[str(qid)] = anchor
            reranked[str(qid)] = {document.doc_id: score for document, score in ranked}
        return reranked, anchors
