"""Step 2: feature engineering in pandas (this project has no Spark).

Builds and persists:
    user index / item index          contiguous 0..N-1 ids used by the models
    genre multi-hot matrix           19 columns from u.item
    item popularity                  training interaction count + rank percentile
    item rating stats                mean rating, rating count
    user history stats               history length, mean rating
    user genre affinity              19 dim profile from the user's train items

All statistics are computed on the training split only. Nothing from the test
split leaks into a feature.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS_DIR, PROCESSED_DIR  # noqa: E402
from data_prep import GENRES  # noqa: E402


@dataclass
class FeatureStore:
    user_ids: np.ndarray          # (U,) raw MovieLens ids
    item_ids: np.ndarray          # (I,) raw MovieLens ids
    genre: np.ndarray             # (I, 19) float32 multi hot
    item_pop: np.ndarray          # (I,) train interaction counts
    item_pop_pct: np.ndarray      # (I,) popularity percentile in [0, 1]
    item_mean_rating: np.ndarray  # (I,)
    item_rating_count: np.ndarray  # (I,)
    user_hist_len: np.ndarray     # (U,)
    user_mean_rating: np.ndarray  # (U,)
    user_genre: np.ndarray        # (U, 19) l1 normalized genre profile
    meta: dict = None             # descriptive stats, filled by build_features

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def user_index(self, raw_id: int) -> int | None:
        return self._user_to_idx.get(int(raw_id))

    def item_index(self, raw_id: int) -> int | None:
        return self._item_to_idx.get(int(raw_id))

    def __post_init__(self) -> None:
        self._user_to_idx = {int(u): i for i, u in enumerate(self.user_ids)}
        self._item_to_idx = {int(m): i for i, m in enumerate(self.item_ids)}


def build_features(train: pd.DataFrame, items: pd.DataFrame,
                   test: pd.DataFrame | None = None) -> FeatureStore:
    """Fit every feature on train, over the full u.item catalog."""
    item_ids = items["item_id"].to_numpy(dtype=np.int64)
    genre = items[GENRES].to_numpy(dtype=np.float32)

    counts = train.groupby("item_id").size()
    item_pop = np.array([counts.get(int(m), 0) for m in item_ids], dtype=np.float32)

    # percentile rank of popularity, 0 for items nobody interacted with
    order = np.argsort(item_pop, kind="mergesort")
    pct = np.zeros(len(item_ids), dtype=np.float32)
    if len(item_ids) > 1:
        pct[order] = np.arange(len(item_ids), dtype=np.float32) / (len(item_ids) - 1)
    pct[item_pop == 0] = 0.0

    rating_stats = train.groupby("item_id")["rating"].agg(["mean", "count"])
    item_mean = np.array(
        [rating_stats["mean"].get(int(m), 0.0) for m in item_ids], dtype=np.float32)
    item_rcount = np.array(
        [rating_stats["count"].get(int(m), 0) for m in item_ids], dtype=np.float32)

    user_ids = np.sort(train["user_id"].unique()).astype(np.int64)
    user_stats = train.groupby("user_id")["rating"].agg(["size", "mean"])
    user_hist = user_stats["size"].reindex(user_ids).to_numpy(dtype=np.float32)
    user_mean = user_stats["mean"].reindex(user_ids).to_numpy(dtype=np.float32)

    # user genre affinity: sum of genre vectors of the items the user touched
    u_idx = {int(u): i for i, u in enumerate(user_ids)}
    i_idx = {int(m): i for i, m in enumerate(item_ids)}
    profile = np.zeros((len(user_ids), len(GENRES)), dtype=np.float32)
    u_rows = np.array([u_idx[int(u)] for u in train["user_id"].to_numpy()])
    m_rows = np.array([i_idx[int(m)] for m in train["item_id"].to_numpy()])
    np.add.at(profile, u_rows, genre[m_rows])
    denom = profile.sum(axis=1, keepdims=True)
    profile = np.divide(profile, np.maximum(denom, 1e-9))

    store = FeatureStore(
        user_ids=user_ids,
        item_ids=item_ids,
        genre=genre,
        item_pop=item_pop,
        item_pop_pct=pct,
        item_mean_rating=item_mean,
        item_rating_count=item_rcount,
        user_hist_len=user_hist,
        user_mean_rating=user_mean,
        user_genre=profile,
    )
    store.meta = {
        "n_users": int(store.n_users),
        "n_items": int(store.n_items),
        "n_genres": len(GENRES),
        "items_with_zero_train_interactions": int((item_pop == 0).sum()),
        "train_interactions": int(len(train)),
        "test_rows_available": int(len(test)) if test is not None else None,
        "cold_items_in_test": int(
            len(set(test["item_id"].unique()) - set(train["item_id"].unique()))
        ) if test is not None else None,
        "mean_item_popularity": float(item_pop.mean()),
        "median_item_popularity": float(np.median(item_pop)),
        "mean_user_history": float(user_hist.mean()),
        "min_user_history": float(user_hist.min()),
    }
    return store


def save(store: FeatureStore, out_dir: Path | None = None) -> Path:
    out = out_dir or ARTIFACTS_DIR
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "features.npz",
        user_ids=store.user_ids,
        item_ids=store.item_ids,
        genre=store.genre,
        item_pop=store.item_pop,
        item_pop_pct=store.item_pop_pct,
        item_mean_rating=store.item_mean_rating,
        item_rating_count=store.item_rating_count,
        user_hist_len=store.user_hist_len,
        user_mean_rating=store.user_mean_rating,
        user_genre=store.user_genre,
    )
    (out / "feature_meta.json").write_text(json.dumps(store.meta, indent=2))
    return out / "features.npz"


def load(path: Path | None = None) -> FeatureStore:
    path = path or ARTIFACTS_DIR / "features.npz"
    z = np.load(path)
    store = FeatureStore(
        user_ids=z["user_ids"],
        item_ids=z["item_ids"],
        genre=z["genre"],
        item_pop=z["item_pop"],
        item_pop_pct=z["item_pop_pct"],
        item_mean_rating=z["item_mean_rating"],
        item_rating_count=z["item_rating_count"],
        user_hist_len=z["user_hist_len"],
        user_mean_rating=z["user_mean_rating"],
        user_genre=z["user_genre"],
    )
    meta_path = Path(path).with_name("feature_meta.json")
    store.meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return store


def main() -> None:
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    items = pd.read_csv(PROCESSED_DIR / "items.csv")
    store = build_features(train, items, test)
    out = save(store)
    print(f"[features] users={store.n_users} items={store.n_items} "
          f"genres={store.genre.shape[1]}")
    print(f"[features] items with zero train interactions: "
          f"{store.meta['items_with_zero_train_interactions']}")
    print(f"[features] saved {out}")


if __name__ == "__main__":
    main()
