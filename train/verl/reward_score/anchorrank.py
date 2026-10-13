"""Custom verl reward functions for AnchorRank JSON score outputs."""
from __future__ import annotations

import json
from typing import Any, Sequence


def _object(text: str) -> dict[str, Any] | None:
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _score(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if 0 <= number <= 100 and number.is_integer() else None


def parse_pointwise(completion: str) -> float | None:
    payload = _object(completion)
    return _score(payload.get("score")) if payload else None


def parse_listwise(completion: str, doc_ids: Sequence[str]) -> list[float] | None:
    payload = _object(completion)
    if not payload:
        return None
    scores = [_score(payload.get(doc_id)) for doc_id in doc_ids]
    return [float(score) for score in scores] if all(score is not None for score in scores) else None


def pointwise_group_rewards(completions: Sequence[str], reference_score: float, is_positive: bool, opposite_scores: Sequence[float], alpha: float = 0.75, threshold: float = 20.0) -> list[float]:
    """Paper Eq. 9--11 for one GRPO completion group.

    `opposite_scores` must be the sampled scores of the paired positive/negative
    document, so the inter-document term is computed over N rollouts.
    """
    rewards = []
    for completion in completions:
        predicted = parse_pointwise(completion)
        if predicted is None:
            rewards.append(-1.0)
            continue
        intra = 1.0 if abs(predicted - reference_score) <= threshold else -1.0
        if not opposite_scores:
            inter = 0.0
        elif is_positive:
            inter = sum(predicted > score for score in opposite_scores) / len(opposite_scores)
        else:
            inter = sum(predicted < score for score in opposite_scores) / len(opposite_scores)
        rewards.append(alpha * intra + (1.0 - alpha) * inter)
    return rewards


def compute_pointwise_batch_scores(
    data_sources: Sequence[str],
    solution_strs: Sequence[str],
    ground_truths: Sequence[str | dict[str, Any]],
    extra_infos: Sequence[dict[str, Any]],
    alpha: float = 0.75,
    threshold: float = 20.0,
) -> list[float]:
    """Compute Eq. 9--11 over all rollouts in each positive-negative pair.

    Every pointwise GRPO row must set ``extra_info.pair_id`` to identify one
    positive-negative document pair and ``extra_info.is_positive`` to identify
    the row's document. verl expands each prompt into N rollouts; this function
    receives that expanded batch and compares each score against all valid N
    scores from its paired opposite document, exactly as the paper defines.
    """
    if not (len(data_sources) == len(solution_strs) == len(ground_truths) == len(extra_infos)):
        raise ValueError("Pointwise batch fields must have matching lengths")

    records: list[dict[str, Any]] = []
    pairs: dict[str, dict[bool, list[int]]] = {}
    for index, (source, completion, truth, extra) in enumerate(zip(data_sources, solution_strs, ground_truths, extra_infos)):
        if source != "anchorrank_pointwise":
            raise ValueError(f"Unexpected data source for pointwise manager: {source}")
        if "pair_id" not in extra or "is_positive" not in extra:
            raise ValueError("Pointwise GRPO data requires extra_info.pair_id and extra_info.is_positive")
        label = bool(extra["is_positive"])
        reference = json.loads(truth) if isinstance(truth, str) else truth
        score = parse_pointwise(completion)
        records.append({"score": score, "reference": float(reference["score"]), "is_positive": label})
        pairs.setdefault(str(extra["pair_id"]), {True: [], False: []})[label].append(index)

    rewards = [-1.0] * len(records)
    for members in pairs.values():
        positive_scores = [records[index]["score"] for index in members[True] if records[index]["score"] is not None]
        negative_scores = [records[index]["score"] for index in members[False] if records[index]["score"] is not None]
        for label, indices in members.items():
            opposite = negative_scores if label else positive_scores
            for index in indices:
                record = records[index]
                if record["score"] is None or not opposite:
                    continue
                intra = 1.0 if abs(record["score"] - record["reference"]) <= threshold else -1.0
                inter = sum(record["score"] > other for other in opposite) / len(opposite) if label else sum(record["score"] < other for other in opposite) / len(opposite)
                rewards[index] = alpha * intra + (1.0 - alpha) * inter
    return rewards


def listwise_reward(completion: str, reference_scores: Sequence[float], doc_ids: Sequence[str], weight: float = 0.75, threshold: float = 20.0) -> float:
    """Paper Eq. 12--14; returns -1 for malformed JSON output."""
    predicted = parse_listwise(completion, doc_ids)
    if predicted is None or len(predicted) != len(reference_scores):
        return -1.0
    pairs = [(i, j) for i in range(len(reference_scores)) for j in range(len(reference_scores)) if reference_scores[i] > reference_scores[j]]
    pair_reward = sum(predicted[i] > predicted[j] for i, j in pairs) / len(pairs) if pairs else 0.0
    threshold_reward = sum(1.0 if abs(score - target) <= threshold else -1.0 for score, target in zip(predicted, reference_scores)) / len(predicted)
    return weight * pair_reward + (1.0 - weight) * threshold_reward


def compute_score(data_source: str, solution_str: str, ground_truth: str | dict[str, Any], extra_info: dict[str, Any] | None = None, **_: Any) -> float:
    """Drop-in function for `verl.utils.reward_score.default_compute_score`.

    For listwise records use data_source `anchorrank_listwise` and put
    `doc_ids` in `extra_info`; for pointwise use `anchorrank_pointwise`.
    The batch-aware pointwise inter-document reward is supplied by
    `pointwise_group_rewards` in the custom reward manager.
    """
    truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    if data_source == "anchorrank_listwise":
        return listwise_reward(solution_str, truth["scores"], (extra_info or {})["doc_ids"])
    if data_source == "anchorrank_pointwise":
        predicted = parse_pointwise(solution_str)
        if predicted is None:
            return -1.0
        return 1.0 if abs(predicted - float(truth["score"])) <= 20.0 else -1.0
    raise ValueError(f"Unsupported data source: {data_source}")
