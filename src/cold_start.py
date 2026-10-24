"""Step 6: cold-start fallback.

A user with fewer than 3 training interactions has no usable collaborative
signal, so the two-tower user embedding is meaningless. The fallback scores
the catalog with:

    0.6 * normalized popularity  +  0.4 * genre affinity to the seed items

With no seed items at all the score collapses to popularity, which is also
what a completely unknown user id gets from the API.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import COLD_START_MAX_HISTORY  # noqa: E402
from features import FeatureStore  # noqa: E402


class ColdStartRecommender:
    def __init__(self, store: FeatureStore, popularity_weight: float = 0.6,
                 genre_weight: float = 0.4):
        self.store = store
        self.popularity_weight = popularity_weight
        self.genre_weight = genre_weight
        max_pop = max(float(store.item_pop.max()), 1.0)
        self.pop_norm = (store.item_pop / max_pop).astype(np.float32)
        gn = np.linalg.norm(store.genre, axis=1, keepdims=True)
        self.genre_norm = store.genre / np.maximum(gn, 1e-9)

    def scores(self, seed_item_idxs: np.ndarray | None = None) -> np.ndarray:
        if seed_item_idxs is None or len(seed_item_idxs) == 0:
            return self.pop_norm.copy()
        profile = self.store.genre[np.asarray(seed_item_idxs, dtype=int)].mean(axis=0)
        n = np.linalg.norm(profile)
        affinity = self.genre_norm @ (profile / max(n, 1e-9))
        affinity = np.clip(affinity, 0.0, 1.0)
        return (self.popularity_weight * self.pop_norm
                + self.genre_weight * affinity).astype(np.float32)

    def recommend(self, seed_item_idxs: np.ndarray | None, k: int,
                  exclude: np.ndarray | None = None) -> tuple[list[int], np.ndarray]:
        s = self.scores(seed_item_idxs)
        if exclude is not None and len(exclude):
            s = s.copy()
            s[np.asarray(exclude, dtype=int)] = -np.inf
        k = min(k, int(np.isfinite(s).sum()))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx])]
        return idx.tolist(), s[idx]


def is_cold_user(history_len: int | None) -> bool:
    """Unknown user or a history shorter than the cold-start threshold."""
    return history_len is None or history_len < COLD_START_MAX_HISTORY
