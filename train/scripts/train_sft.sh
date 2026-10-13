#!/usr/bin/env bash
set -euo pipefail
: "${LLAMAFACTORY_DIR:?Set LLAMAFACTORY_DIR to your LLaMA-Factory checkout}"
: "${CONFIG:?Set CONFIG to train/llamafactory/configs/pointwise_sft.yaml or listwise_sft.yaml}"
cd "$LLAMAFACTORY_DIR"
FORCE_TORCHRUN=1 llamafactory-cli train "$CONFIG"
