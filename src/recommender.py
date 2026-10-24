"""Serving logic shared by the FastAPI app, the latency benchmark and tests.

One code path, three callers, so the API response and the reported metrics can
never drift apart.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cold_start import ColdStartRecommender, is_cold_user  # noqa: E402
from config import ARTIFACTS_DIR, COLD_START_MAX_HISTORY, PROCESSED_DIR, RETRIEVAL_K  # noqa: E402
from faiss_index import FaissRetriever  # noqa: E402
from features import load as load_features  # noqa: E402
from ranker import load_model, score_candidates  # noqa: E402


class RecommenderService:
    def __init__(self, artifacts_dir: Path | None = None):
        art = Path(artifacts_dir or ARTIFACTS_DIR)
        self.store = load_features(art / "features.npz")
        self.user_emb = np.load(art / "user_embeddings.npy")
        self.item_emb = np.load(art / "item_embeddings.npy")
        self.faiss = FaissRetriever.load(art / "item_index.faiss")
        self.ranker = load_model(art / "ranker.pt")
        self.cold = ColdStartRecommender(self.store)

        train = pd.read_csv(PROCESSED_DIR / "train.csv")
        u_map = {int(u): i for i, u in enumerate(self.store.user_ids)}
        seen: dict[int, list[int]] = {}
        recent: dict[int, list[int]] = {}
        i_map = {int(m): i for i, m in enumerate(self.store.item_ids)}
        ordered = train.sort_values(["user_id", "timestamp"])
        for u, m in zip(ordered["user_id"].to_numpy(), ordered["item_id"].to_numpy()):
            ui = u_map[int(u)]
            seen.setdefault(ui, []).append(i_map[int(m)])
            recent.setdefault(ui, []).append(i_map[int(m)])
        self.user_seen = {k: np.array(v, dtype=np.int64) for k, v in seen.items()}
        self.user_recent = {k: np.array(v[-COLD_START_MAX_HISTORY:], dtype=np.int64)
                            for k, v in recent.items()}

        items = pd.read_csv(PROCESSED_DIR / "items.csv")
        genre_cols = [c for c in items.columns if c not in
                      ("item_id", "title", "release_date", "video_release_date",
                       "imdb_url")]
        genre_matrix = items[genre_cols].to_numpy()
        self.item_meta = {}
        for pos, item_id in enumerate(items["item_id"].to_numpy()):
            gs = [genre_cols[j] for j in np.where(genre_matrix[pos] == 1)[0]]
            self.item_meta[int(item_id)] = (str(items["title"].iloc[pos]), gs)

        self.n_users = self.store.n_users
        self.n_items = self.store.n_items

    # ------------------------------------------------------------------ paths
    def _two_stage(self, ui: int, k: int) -> tuple[list[int], np.ndarray, int]:
        blocked = self.user_seen.get(ui, np.array([], dtype=np.int64))
        _, ids = self.faiss.search(self.user_emb[ui], RETRIEVAL_K + len(blocked))
        cand = ids[~np.isin(ids, blocked)][:RETRIEVAL_K]
        scores = score_candidates(self.ranker, self.store, ui, cand,
                                  self.user_emb, self.item_emb)
        order = np.argsort(-scores)[:k]
        return cand[order].tolist(), scores[order], len(cand)

    def _cold(self, raw_user_id: int, k: int) -> tuple[list[int], np.ndarray, str]:
        ui = self.store.user_index(raw_user_id)
        seeds = self.user_recent.get(ui, np.array([], dtype=np.int64)) if ui is not None \
            else np.array([], dtype=np.int64)
        recs, scores = self.cold.recommend(seeds, k, exclude=seeds)
        strategy = "cold_start_genre_affinity" if len(seeds) else "cold_start_popularity"
        return recs, scores, strategy

    # ------------------------------------------------------------------ public
    def recommend(self, user_id: int, k: int = 10) -> dict:
        t0 = time.perf_counter()
        ui = self.store.user_index(user_id)
        hist_len = int(self.store.user_hist_len[ui]) if ui is not None else None
        cold = is_cold_user(hist_len)

        if cold:
            recs, scores, strategy = self._cold(user_id, k)
            n_candidates = self.n_items
        else:
            assert ui is not None
            recs, scores, n_candidates = self._two_stage(ui, k)
            strategy = "two_stage_faiss_ranker"

        latency_ms = round((time.perf_counter() - t0) * 1000, 3)
        return {
            "user_id": int(user_id),
            "k": int(k),
            "cold_start": bool(cold),
            "strategy": strategy,
            "history_length": hist_len if hist_len is not None else 0,
            "n_candidates": int(n_candidates),
            "latency_ms": latency_ms,
            "recommendations": [
                {
                    "rank": rank + 1,
                    "item_id": int(self.store.item_ids[item]),
                    "title": self.item_meta.get(int(self.store.item_ids[item]), ("", []))[0],
                    "genres": self.item_meta.get(int(self.store.item_ids[item]), ("", []))[1],
                    "score": round(float(score), 6),
                }
                for rank, (item, score) in enumerate(zip(recs, scores))
            ],
        }

    def health(self) -> dict:
        return {
            "status": "ok",
            "users": self.n_users,
            "items": self.n_items,
            "faiss_vectors": int(self.faiss.index.ntotal),
            "faiss_index": "IndexFlatIP",
            "embedding_dim": int(self.item_emb.shape[1]),
            "retrieval_k": RETRIEVAL_K,
            "cold_start_threshold": COLD_START_MAX_HISTORY,
            "ranker": "wide_and_deep_mlp",
        }
