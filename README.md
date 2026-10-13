# AnchorRank: Towards Globally Consistent Relevance Scoring for Reasoning-Intensive Reranking

This repository contains a partial research implementation of **AnchorRank**. It provides the core anchor-guided inference code and selected training/evaluation utilities.

> **Reproducibility Note**
> This repository is intentionally a partial public release. Benchmark downloads, complete training data preparation, full model checkpoints, and the remaining reproduction scripts will be released after paper acceptance.

## 1. Environments

The recommended environment is:

```text
OS: Ubuntu 22.04+
Python: 3.11
CUDA: 12.4+
PyTorch: 2.8+
```

Install the inference dependencies with:

```bash
conda create -n AnchorRank python=3.11
conda activate AnchorRank
pip install -r requirements.txt
```

The training configurations are intended to run in separate LLaMA-Factory and verl environments. The full environment and installation scripts are not included in this partial release.

## 2. Repository Structure

```text
AnchorRank/
|- anchorrank/                  # Shared anchor construction and reranking logic
|- evaluation/                  # AnchorRank evaluation entry points
|  |- main.py
|  |- runner.py
|  |- reranker_sglang.py
|  `- scripts/AnchorRank-P-7B.sh
|- train/
|  |- llamafactory/              # Selected SFT configurations
|  |- verl/                      # Selected GRPO reward/configuration code
|  `- scripts/
`- README.md
```

AnchorRank constructs one relevance anchor per query and reuses it for all pointwise documents or listwise windows. The available anchor strategies are:

- `ae`: Aggregated Evidence from the top 40 retrieved documents, processed in groups of 10.
- `trd`: The top-ranked retrieved document.
- `ga`: A teacher-generated answer from the query.

## 3. Training

The repository contains selected training configurations only. The complete Retro*/ReasonRank data conversion and internal training pipeline are intentionally not included in this snapshot.

### 3.1 Supervised Fine-Tuning (SFT)

The paper uses full-parameter SFT on Qwen2.5-7B-Instruct with a learning rate of `1e-5`, batch size 1, gradient accumulation 8, BF16, ZeRO-3, sequence length 8192, and 3 epochs.

The corresponding configurations are:

```text
train/llamafactory/configs/pointwise_sft.yaml
train/llamafactory/configs/listwise_sft.yaml
```

Launch through an external LLaMA-Factory checkout:

```bash
export LLAMAFACTORY_DIR=/path/to/LlamaFactory
export CONFIG=/path/to/AnchorRank/train/llamafactory/configs/pointwise_sft.yaml
bash train/scripts/train_sft.sh
```

### 3.2 Group Relative Policy Optimization (GRPO)

The selected verl utilities implement the AnchorRank pointwise and listwise rewards. The pointwise reward follows the paper's composite definition:

```text
R_P = 0.75 * R_intra + 0.25 * R_inter
```

The GRPO launcher uses the paper's main settings: learning rate `1e-6`, batch size `64`, mini-batch size `4`, micro-batch size `1`, 8 rollouts, temperature `1.0`, KL coefficient `0.005`, and 3 epochs.

```bash
export VERL_DIR=/path/to/verl
export TRAIN_FILES=/path/to/train.parquet
export SFT_MODEL_PATH=/path/to/AnchorRank-SFT
export SAVING_PATH=/path/to/AnchorRank-GRPO
export TASK=pointwise
bash train/scripts/train_grpo.sh
```

Training anchors are intentionally excluded, matching the paper's training strategy.

## 4. Evaluation

The evaluation interface uses the following local dataset and retrieval-result layout:

```text
data/text/<benchmark>/<dataset>/{corpus.jsonl,queries.jsonl,qrels.jsonl}
data/retrieval/<retrieval_run>/<dataset>.json
```

The main paper experiments use Diver-Retriever-4B to retrieve the top 100 documents and AnchorRank-P (7B) as the default pointwise reranker. For BRIGHT, retrieval uses GPT-4-rewritten queries while reranking uses the original queries, as described in the paper.

Run the default AnchorRank-P evaluation from the repository root:

```bash
python -m evaluation.main \
  --dataset_dir data/text/bright \
  --retriever_results_dir data/retrieval/diver_retriever_4b_top100 \
  --reranker_results_dir data/reranking/diver_retriever_4b_top100/AnchorRank-P-7B \
  --dataset_names biology \
  --reranker_name_or_path model/AnchorRank-P-7B \
  --ae_anchor_model_name_or_path model/Qwen3-8B \
  --reranker_mode pointwise \
  --anchor_strategy ae \
  --rerank_top_k 100
```

For the paper's model separation:

- `--reranker_name_or_path` points to the AnchorRank reranker.
- `--ae_anchor_model_name_or_path` points to Qwen3-8B for AE.
- `--ga_anchor_model_name_or_path` points to the GPT-6 Luna teacher-model checkpoint or serving path for GA.

The AnchorRank-P launch script is:

```bash
bash evaluation/scripts/AnchorRank-P-7B.sh
```

The output JSON preserves `search_results` and records the fixed query-level `anchors` used during reranking.
