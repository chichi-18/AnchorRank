from __future__ import annotations

import argparse

from .arguments import AnchorRankEvalArgs, AnchorRankEvalModelArgs
from .runner import AnchorRankEvalRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="CuRLRank-style AnchorRank evaluation")
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--retriever_results_dir", required=True)
    parser.add_argument("--reranker_results_dir", required=True)
    parser.add_argument("--dataset_names", nargs="+", required=True)
    parser.add_argument("--reranker_name_or_path", default="model/AnchorRank-P-7B")
    parser.add_argument("--rerank_top_k", type=int, default=100)
    parser.add_argument("--reranker_mode", choices=["pointwise", "listwise"], default="pointwise")
    parser.add_argument("--anchor_strategy", choices=["ae", "trd", "ga"], default="ae")
    parser.add_argument("--ae_anchor_model_name_or_path", default="Qwen/Qwen3-8B")
    parser.add_argument("--ga_anchor_model_name_or_path", default="GPT-6-Luna")
    parser.add_argument("--anchor_top_k", type=int, default=40)
    parser.add_argument("--reranker_window_size", type=int, default=10)
    parser.add_argument("--reranker_sglang_context_length", type=int, default=12288)
    parser.add_argument("--reranker_sglang_tp_size", type=int, default=1)
    parser.add_argument("--reranker_sglang_dp_size", type=int, default=1)
    parser.add_argument("--reranker_sglang_seed", type=int, default=42)
    parser.add_argument("--reranker_enable_thinking", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    eval_args = AnchorRankEvalArgs(args.dataset_dir, args.retriever_results_dir, args.reranker_results_dir, args.dataset_names, args.rerank_top_k, args.overwrite)
    model_args = AnchorRankEvalModelArgs(args.reranker_name_or_path, reranker_mode=args.reranker_mode, anchor_strategy=args.anchor_strategy, ae_anchor_model_name_or_path=args.ae_anchor_model_name_or_path, ga_anchor_model_name_or_path=args.ga_anchor_model_name_or_path, anchor_top_k=args.anchor_top_k, reranker_window_size=args.reranker_window_size, reranker_sglang_context_length=args.reranker_sglang_context_length, reranker_sglang_tp_size=args.reranker_sglang_tp_size, reranker_sglang_dp_size=args.reranker_sglang_dp_size, reranker_sglang_seed=args.reranker_sglang_seed, reranker_enable_thinking=args.reranker_enable_thinking)
    for output in AnchorRankEvalRunner(eval_args, model_args).run():
        print(output)


if __name__ == "__main__":
    main()
