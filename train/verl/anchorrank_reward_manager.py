"""Batch reward manager required for AnchorRank pointwise GRPO rewards."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager


class AnchorRankPointwiseRewardManager(AbstractRewardManager):
    """Passes the whole expanded GRPO batch to the paper's pairwise reward."""

    def __init__(self, config, tokenizer, compute_score, **_: Any) -> None:
        self.tokenizer = tokenizer
        self.compute_score = compute_score
        self.num_examine = config.reward.reward_model.get("num_examine", 1)

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_from_rm_scores = self._extract_reward_from_rm_scores(data, return_dict)
        if reward_from_rm_scores is not None:
            return reward_from_rm_scores

        responses = data.batch["responses"]
        prompts = data.batch["prompts"]
        prompt_length = prompts.shape[-1]
        lengths = data.batch["attention_mask"][:, prompt_length:].sum(dim=-1)
        solution_strs = [self.tokenizer.decode(responses[index][: lengths[index]], skip_special_tokens=True) for index in range(len(data))]
        sources = list(data.non_tensor_batch["data_source"])
        truths = [item.non_tensor_batch["reward_model"]["ground_truth"] for item in data]
        extras = list(data.non_tensor_batch.get("extra_info", [{} for _ in range(len(data))]))
        scores = self.compute_score(data_sources=sources, solution_strs=solution_strs, ground_truths=truths, extra_infos=extras)

        reward_tensor = torch.zeros_like(responses, dtype=torch.float32)
        for index, (length, score) in enumerate(zip(lengths.tolist(), scores)):
            reward_tensor[index, length - 1] = score
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": defaultdict(list)}
        return reward_tensor
