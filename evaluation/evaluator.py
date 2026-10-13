from __future__ import annotations

import json
from pathlib import Path


class AnchorRankEvaluator:
    def __init__(self, data_loader, reranker, retriever_results_dir: str, reranker_results_dir: str, rerank_top_k: int, overwrite: bool):
        self.data_loader, self.reranker = data_loader, reranker
        self.retriever_results_dir = Path(retriever_results_dir)
        self.reranker_results_dir = Path(reranker_results_dir)
        self.rerank_top_k, self.overwrite = rerank_top_k, overwrite

    def evaluate(self, dataset_name: str) -> Path:
        output = self.reranker_results_dir / f"{dataset_name}.json"
        if output.is_file() and not self.overwrite:
            return output
        corpus = self.data_loader.load_corpus(dataset_name)
        queries = self.data_loader.load_queries(dataset_name)
        source = self.retriever_results_dir / f"{dataset_name}.json"
        retrieval = json.loads(source.read_text(encoding="utf-8"))
        retrieval = retrieval.get("search_results", retrieval)
        results, anchors = self.reranker(corpus, queries, retrieval, dataset_name=dataset_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"dataset_name": dataset_name, "reranker_name": str(self.reranker), "anchors": anchors, "search_results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
        return output
