from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AnchorRankEvalArgs:
    dataset_dir: str
    retriever_results_dir: str
    reranker_results_dir: str
    dataset_names: list[str]
    rerank_top_k: int = 100
    overwrite: bool = False


@dataclass
class AnchorRankEvalModelArgs:
    reranker_name_or_path: str = "model/AnchorRank-P-7B"
    reranker_model_class: str = "sglang-anchor"
    reranker_mode: str = "pointwise"
    anchor_strategy: str = "ae"
    ae_anchor_model_name_or_path: str = "Qwen/Qwen3-8B"
    ga_anchor_model_name_or_path: str = "GPT-6-Luna"
    anchor_top_k: int = 40
    reranker_window_size: int = 10
    reranker_sglang_context_length: int = 12288
    reranker_sglang_tp_size: int = 1
    reranker_sglang_dp_size: int = 1
    reranker_sglang_seed: int = 42
    reranker_enable_thinking: bool = False
