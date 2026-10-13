from __future__ import annotations

from anchorrank import AnchorBuilder, AnchorReranker, Document, QueryExample
from anchorrank.sglang_backend import SGLangGenerator


class SGLangAnchorRankReranker:
    """AnchorRank model adapter, parallel to CuRLRank's SGLang reranker."""

    def __init__(self, model_name_or_path: str, mode: str, anchor_strategy: str, ae_anchor_model_name_or_path: str, ga_anchor_model_name_or_path: str, anchor_top_k: int, window_size: int, context_length: int, tp_size: int, dp_size: int, random_seed: int, enable_thinking: bool, anchor_cache_path: str):
        self.model_name_or_path = model_name_or_path
        self.reranker_generator = SGLangGenerator(model_name_or_path, context_length, tp_size, dp_size, random_seed, enable_thinking)
        anchor_model = ae_anchor_model_name_or_path if anchor_strategy == "ae" else ga_anchor_model_name_or_path
        self.anchor_generator = SGLangGenerator(anchor_model, context_length, tp_size, dp_size, random_seed, enable_thinking) if anchor_strategy in {"ae", "ga"} else None
        builder = AnchorBuilder(self.anchor_generator, anchor_strategy, anchor_top_k, cache_path=anchor_cache_path) if self.anchor_generator else AnchorBuilder(self.reranker_generator, anchor_strategy, anchor_top_k, cache_path=anchor_cache_path)
        self.reranker = AnchorReranker(self.reranker_generator, builder, mode, window_size)

    def rerank(self, qid: str, query: str, documents: list[Document]):
        return self.reranker.rerank(QueryExample(qid, query, documents))

    def shutdown(self) -> None:
        self.reranker_generator.shutdown()
        if self.anchor_generator is not None:
            self.anchor_generator.shutdown()
