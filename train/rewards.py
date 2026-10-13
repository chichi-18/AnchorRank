"""AnchorRank GRPO rewards from Appendix: pointwise and listwise formulas."""
from __future__ import annotations

from itertools import product
from statistics import mean
from typing import Sequence


def pointwise_rewards(predicted: Sequence[float], reference: Sequence[float], positive_mask: Sequence[bool], alpha: float = 0.75, threshold: float = 20.0) -> list[float]:
    if not (len(predicted) == len(reference) == len(positive_mask)):
        raise ValueError("predicted, reference, and positive_mask must have equal length")
    positives = [score for score, label in zip(predicted, positive_mask) if label]
    negatives = [score for score, label in zip(predicted, positive_mask) if not label]
    rewards = []
    for score, target, label in zip(predicted, reference, positive_mask):
        intra = 1.0 if abs(score - target) <= threshold else -1.0
        others = negatives if label else positives
        inter = 0.0 if not others else mean(float(score > other) if label else float(score < other) for other in others)
        rewards.append(alpha * intra + (1.0 - alpha) * inter)
    return rewards


def listwise_reward(predicted: Sequence[float], reference: Sequence[float], weight: float = 0.75, threshold: float = 20.0) -> float:
    if len(predicted) != len(reference):
        raise ValueError("predicted and reference must have equal length")
    pairs = [(i, j) for i, j in product(range(len(reference)), repeat=2) if reference[i] > reference[j]]
    pair = mean(float(predicted[i] > predicted[j]) for i, j in pairs) if pairs else 0.0
    bounded = mean(1.0 if abs(p - r) <= threshold else -1.0 for p, r in zip(predicted, reference)) if predicted else 0.0
    return weight * pair + (1.0 - weight) * bounded
