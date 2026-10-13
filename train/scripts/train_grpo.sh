#!/usr/bin/env bash
set -euo pipefail
: "${VERL_DIR:?Set VERL_DIR to your verl checkout}"
: "${TRAIN_FILES:?Set TRAIN_FILES to an AnchorRank GRPO parquet file}"
: "${SFT_MODEL_PATH:?Set SFT_MODEL_PATH to the SFT checkpoint}"
: "${SAVING_PATH:?Set SAVING_PATH for GRPO checkpoints}"
: "${TASK:?Set TASK to pointwise or listwise}"

if [[ "$TASK" != "pointwise" && "$TASK" != "listwise" ]]; then
  echo "TASK must be pointwise or listwise" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
python "$REPO_DIR/train/verl/patch_verl_reward.py" --verl-dir "$VERL_DIR"
cd "$VERL_DIR"
COMMON_ARGS=(
  algorithm.adv_estimator=grpo
  data.train_files="$TRAIN_FILES"
  data.train_batch_size=64
  data.max_prompt_length=8192
  actor_rollout_ref.model.path="$SFT_MODEL_PATH"
  actor_rollout_ref.actor.optim.lr=1e-6
  actor_rollout_ref.actor.ppo_mini_batch_size=4
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
  actor_rollout_ref.actor.use_kl_loss=true
  actor_rollout_ref.actor.kl_loss_coef=0.005
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=1.0
  trainer.total_epochs=3
  trainer.default_local_dir="$SAVING_PATH"
)

if [[ "$TASK" == "pointwise" ]]; then
  COMMON_ARGS+=(
    data.max_response_length=128
    reward.reward_manager.source=importlib
    reward.reward_manager.name=AnchorRankPointwiseRewardManager
    reward.reward_manager.module.path="$REPO_DIR/train/verl/anchorrank_reward_manager.py"
    reward.custom_reward_function.path="$REPO_DIR/train/verl/reward_score/anchorrank.py"
    reward.custom_reward_function.name=compute_pointwise_batch_scores
  )
else
  COMMON_ARGS+=(data.max_response_length=512)
fi

python -m verl.trainer.main_ppo "${COMMON_ARGS[@]}" \
  "$@"
