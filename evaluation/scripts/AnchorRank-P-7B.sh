#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${ROOT_DIR:-"$(cd "$(dirname "$0")/../.." && pwd)"}
RETRIEVER_RUN=${RETRIEVER_RUN:-"diver_retriever_4b_top100"}

python -m evaluation.main \
  --dataset_dir "$ROOT_DIR/data/text/bright" \
  --retriever_results_dir "$ROOT_DIR/data/retrieval/$RETRIEVER_RUN" \
  --reranker_results_dir "$ROOT_DIR/data/reranking/$RETRIEVER_RUN/AnchorRank-P-7B" \
  --dataset_names biology earth_science economics robotics psychology stackoverflow sustainable_living leetcode pony aops theoremqa_questions theoremqa_theorems \
  --reranker_name_or_path "$ROOT_DIR/model/AnchorRank-P-7B" \
  --ae_anchor_model_name_or_path "$ROOT_DIR/model/Qwen3-8B" \
  --rerank_top_k 100 \
  --reranker_mode pointwise \
  --anchor_strategy ae \
  --reranker_window_size 10 \
  --anchor_top_k 40 \
  --reranker_sglang_context_length 12288 \
  --reranker_sglang_tp_size 1 \
  --reranker_sglang_dp_size 4 \
  --overwrite
