from __future__ import annotations

from pathlib import Path

from .data_loader import AnchorRankEvalDataLoader
from .evaluator import AnchorRankEvaluator
from .reranker_searcher import AnchorRankEvalReranker
from .reranker_sglang import SGLangAnchorRankReranker


class AnchorRankEvalRunner:
    def __init__(self, eval_args, model_args):
        self.eval_args, self.model_args = eval_args, model_args

    def run(self) -> list[Path]:
        data_loader = AnchorRankEvalDataLoader(self.eval_args.dataset_dir)
        model = SGLangAnchorRankReranker(
            self.model_args.reranker_name_or_path, self.model_args.reranker_mode, self.model_args.anchor_strategy,
            self.model_args.ae_anchor_model_name_or_path, self.model_args.ga_anchor_model_name_or_path,
            self.model_args.anchor_top_k, self.model_args.reranker_window_size, self.model_args.reranker_sglang_context_length,
            self.model_args.reranker_sglang_tp_size, self.model_args.reranker_sglang_dp_size, self.model_args.reranker_sglang_seed,
            self.model_args.reranker_enable_thinking, str(Path(self.eval_args.reranker_results_dir) / "anchor_cache.json"),
        )
        reranker = AnchorRankEvalReranker(model, self.eval_args.rerank_top_k)
        evaluator = AnchorRankEvaluator(data_loader, reranker, self.eval_args.retriever_results_dir, self.eval_args.reranker_results_dir, self.eval_args.rerank_top_k, self.eval_args.overwrite)
        try:
            return [evaluator.evaluate(name) for name in self.eval_args.dataset_names]
        finally:
            model.shutdown()
