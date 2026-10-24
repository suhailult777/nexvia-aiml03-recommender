"""Ranking metrics: Recall@K and NDCG@K over a full candidate list.

Relevance is binary: an item is relevant to a user if it appears in that
user's held out (test) interactions. Items the user already interacted with
in training are masked out before scoring, the same way a live recommender
would not recommend something the user has already watched.
"""
from __future__ import annotations

import numpy as np


def _dcg(hits: np.ndarray) -> float:
    """Discounted cumulative gain for binary gains at rank positions 1..n."""
    if len(hits) == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, len(hits) + 2))
    return float(np.sum(hits * discounts))


def recall_at_k(ranked: list[list[int]], relevant: list[set[int]], k: int) -> float:
    scores = []
    for recs, rel in zip(ranked, relevant):
        if not rel:
            continue
        top = set(recs[:k])
        scores.append(len(top & rel) / len(rel))
    return float(np.mean(scores)) if scores else 0.0


def ndcg_at_k(ranked: list[list[int]], relevant: list[set[int]], k: int) -> float:
    scores = []
    for recs, rel in zip(ranked, relevant):
        if not rel:
            continue
        hits = np.array([1.0 if item in rel else 0.0 for item in recs[:k]])
        ideal = np.ones(min(len(rel), k), dtype=np.float64)
        idcg = _dcg(ideal)
        scores.append(_dcg(hits) / idcg if idcg > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def evaluate_rankings(ranked: list[list[int]], relevant: list[set[int]],
                      ks: tuple[int, ...] = (10, 50)) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(ranked, relevant, k), 6)
        out[f"ndcg@{k}"] = round(ndcg_at_k(ranked, relevant, k), 6)
    out["n_users_evaluated"] = len([r for r in relevant if r])
    return out


def top_k_from_scores(scores: np.ndarray, k: int,
                      blocked: np.ndarray | None = None) -> list[int]:
    """Return the k highest scoring item indices, ignoring blocked ones."""
    s = scores.copy()
    if blocked is not None and len(blocked):
        s[blocked] = -np.inf
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    return idx[np.argsort(-s[idx])].tolist()
