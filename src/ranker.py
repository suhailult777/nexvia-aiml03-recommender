"""Step 5: wide and deep ranker in PyTorch.

Input per (user, candidate item) pair:
    frozen two-tower user embedding (64)
    frozen two-tower item embedding (64)
    dot product, popularity, popularity percentile, item mean rating,
    item rating count, user history length, user mean rating, genre cosine,
    genre count, rating gap, popularity x genre cross
    plus two bucketed crosses fed to the wide part

Structure:
    wide  = Linear(features) + Embedding(pop_bucket x genre_bucket)
                             + Embedding(dot_bucket x hist_bucket)
    deep  = MLP(features) -> 64 -> 1
    logit = wide + deep

Trained pointwise with BCE on positives from train and two kinds of negatives
(hard negatives mined from FAISS retrieval, random catalog items).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    ARTIFACTS_DIR,
    PROCESSED_DIR,
    RANKER_BATCH,
    RANKER_EPOCHS,
    RANKER_HARD_NEG_PER_USER,
    RANKER_HIDDEN,
    RANKER_LR,
    RANKER_POS_PER_USER,
    RANKER_RANDOM_NEG_PER_USER,
    RESULTS_DIR,
    RETRIEVAL_K,
    SEED,
)
from features import FeatureStore, load as load_features  # noqa: E402

N_POP_BUCKETS = 5
N_GENRE_BUCKETS = 5
N_DOT_BUCKETS = 5
N_HIST_BUCKETS = 5


def _buckets(x: np.ndarray, n: int) -> np.ndarray:
    """Quantile free bucketing on [0, 1] values."""
    return np.clip((x * n).astype(np.int64), 0, n - 1)


def build_matrix(store: FeatureStore, u_idx: np.ndarray, i_idx: np.ndarray,
                 u_emb: np.ndarray, i_emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (dense features, two wide cross indices) for index arrays."""
    ue = u_emb[u_idx]
    ie = i_emb[i_idx]
    dot = np.sum(ue * ie, axis=1)

    pop = store.item_pop[i_idx]
    pop_log = np.log1p(pop) / np.log1p(max(store.item_pop.max(), 1.0))
    pop_pct = store.item_pop_pct[i_idx]
    item_mean = store.item_mean_rating[i_idx] / 5.0
    rcount = np.log1p(store.item_rating_count[i_idx]) / np.log1p(
        max(store.item_rating_count.max(), 1.0))
    hist = np.log1p(store.user_hist_len[u_idx]) / np.log1p(
        max(store.user_hist_len.max(), 1.0))
    umean = store.user_mean_rating[u_idx] / 5.0

    ug = store.user_genre[u_idx]
    ig = store.genre[i_idx]
    genre_cos = np.sum(ug * ig, axis=1) / (
        np.linalg.norm(ug, axis=1) * np.linalg.norm(ig, axis=1) + 1e-9)
    genre_count = ig.sum(axis=1) / store.genre.shape[1]
    rating_gap = np.abs(umean - item_mean)

    feats = np.column_stack([
        ue, ie, dot, pop_log, pop_pct, item_mean, rcount, hist, umean,
        genre_cos, genre_count, rating_gap, pop_pct * genre_cos,
        hist * genre_cos,
    ]).astype(np.float32)

    cross1 = _buckets(pop_pct, N_POP_BUCKETS) * N_GENRE_BUCKETS + _buckets(
        np.clip(genre_cos, 0, 1), N_GENRE_BUCKETS)
    cross2 = _buckets(np.clip((dot + 1) / 2, 0, 1), N_DOT_BUCKETS) * N_HIST_BUCKETS + \
        _buckets(hist, N_HIST_BUCKETS)
    return feats, np.column_stack([cross1, cross2]).astype(np.int64)


class WideAndDeep(nn.Module):
    def __init__(self, n_features: int, hidden: list[int] = RANKER_HIDDEN):
        super().__init__()
        self.wide_linear = nn.Linear(n_features, 1)
        self.cross1 = nn.Embedding(N_POP_BUCKETS * N_GENRE_BUCKETS, 1)
        self.cross2 = nn.Embedding(N_DOT_BUCKETS * N_HIST_BUCKETS, 1)
        layers: list[nn.Module] = []
        prev = n_features
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.deep = nn.Sequential(*layers)
        nn.init.zeros_(self.cross1.weight)
        nn.init.zeros_(self.cross2.weight)

    def forward(self, x: torch.Tensor, cross: torch.Tensor) -> torch.Tensor:
        wide = self.wide_linear(x)
        wide = wide + self.cross1(cross[:, 0]) + self.cross2(cross[:, 1])
        return (wide + self.deep(x)).squeeze(-1)


def mine_training_rows(store: FeatureStore, train: pd.DataFrame,
                       u_emb: np.ndarray, i_emb: np.ndarray,
                       retriever, rng: np.random.Generator) -> tuple[np.ndarray, ...]:
    """Sample positives, FAISS hard negatives and random negatives per user."""
    u_map = {int(u): i for i, u in enumerate(store.user_ids)}
    i_map = {int(m): i for i, m in enumerate(store.item_ids)}

    history: dict[int, set[int]] = {}
    for u, m in zip(train["user_id"].to_numpy(), train["item_id"].to_numpy()):
        history.setdefault(u_map[int(u)], set()).add(i_map[int(m)])

    all_items = np.arange(store.n_items)
    rows_u, rows_i, labels = [], [], []
    for ui in range(store.n_users):
        seen = history.get(ui, set())
        if not seen:
            continue
        seen_arr = np.fromiter(seen, dtype=np.int64)
        pos_pool = np.array(sorted(seen))
        n_pos = min(RANKER_POS_PER_USER, len(pos_pool))
        pos = rng.choice(pos_pool, size=n_pos, replace=False)

        # hard negatives: retrieve a wide pool, drop anything the user saw
        _, cand = retriever.search(u_emb[ui], RETRIEVAL_K + len(seen))
        hard_pool = np.array([c for c in cand if int(c) not in seen], dtype=np.int64)
        if len(hard_pool) == 0:
            hard_pool = np.setdiff1d(all_items, seen_arr)
        n_hard = min(RANKER_HARD_NEG_PER_USER, len(hard_pool))
        hard = rng.choice(hard_pool, size=n_hard, replace=False)

        unseen = np.setdiff1d(all_items, seen_arr)
        n_rand = min(RANKER_RANDOM_NEG_PER_USER, len(unseen))
        rand = rng.choice(unseen, size=n_rand, replace=False)

        rows_u.extend([ui] * n_pos)
        rows_i.extend(pos.tolist())
        labels.extend([1.0] * n_pos)
        rows_u.extend([ui] * (n_hard + n_rand))
        rows_i.extend(hard.tolist())
        rows_i.extend(rand.tolist())
        labels.extend([0.0] * (n_hard + n_rand))

    u_arr = np.array(rows_u, dtype=np.int64)
    i_arr = np.array(rows_i, dtype=np.int64)
    y = np.array(labels, dtype=np.float32)
    X, cross = build_matrix(store, u_arr, i_arr, u_emb, i_emb)
    return X, cross, y, u_arr


def train(trials_verbose: bool = True) -> dict:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    store = load_features()
    train_df = pd.read_csv(PROCESSED_DIR / "train.csv")
    u_emb = np.load(ARTIFACTS_DIR / "user_embeddings.npy")
    i_emb = np.load(ARTIFACTS_DIR / "item_embeddings.npy")

    from faiss_index import FaissRetriever
    retriever = FaissRetriever(i_emb)

    X, cross, y, users = mine_training_rows(
        store, train_df, u_emb, i_emb, retriever, rng)
    print(f"[ranker] mined {len(y)} rows: {int(y.sum())} positives, "
          f"{int((1 - y).sum())} negatives", flush=True)

    # split rows by user so validation users never appear in training rows
    uniq = np.unique(users)
    rng.shuffle(uniq)
    n_val = max(1, int(0.1 * len(uniq)))
    val_users = set(uniq[:n_val].tolist())
    is_val = np.array([u in val_users for u in users])
    Xtr, Ctr, ytr = X[~is_val], cross[~is_val], y[~is_val]
    Xva, Cva, yva = X[is_val], cross[is_val], y[is_val]

    model = WideAndDeep(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=RANKER_LR)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(ytr)
    history = []
    t0 = time.time()
    for epoch in range(1, RANKER_EPOCHS + 1):
        model.train()
        perm = rng.permutation(n)
        total = 0.0
        for s in range(0, n, RANKER_BATCH):
            b = perm[s:s + RANKER_BATCH]
            xb = torch.from_numpy(Xtr[b])
            cb = torch.from_numpy(Ctr[b])
            yb = torch.from_numpy(ytr[b])
            opt.zero_grad()
            loss = loss_fn(model(xb, cb), yb)
            loss.backward()
            opt.step()
            total += float(loss) * len(b)
        model.eval()
        with torch.no_grad():
            val_logits = model(torch.from_numpy(Xva), torch.from_numpy(Cva))
            val_loss = float(loss_fn(val_logits, torch.from_numpy(yva)))
            val_auc = float(roc_auc_score(yva, val_logits.numpy()))
        train_loss = total / n
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_loss": round(val_loss, 5),
                        "val_auc": round(val_auc, 5)})
        if trials_verbose:
            print(f"[ranker] epoch {epoch:2d}/{RANKER_EPOCHS} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_auc={val_auc:.4f}", flush=True)

    seconds = round(time.time() - t0, 2)
    torch.save({"state_dict": model.state_dict(),
                "n_features": int(X.shape[1]),
                "hidden": RANKER_HIDDEN},
               ARTIFACTS_DIR / "ranker.pt")

    info = {
        "train_rows": int(len(ytr)), "val_rows": int(len(yva)),
        "positives": int(y.sum()), "negatives": int((1 - y).sum()),
        "train_seconds": seconds, "history": history,
        "final_val_auc": history[-1]["val_auc"],
        "final_train_loss": history[-1]["train_loss"],
        "n_features": int(X.shape[1]),
        "hard_negative_source": "FAISS top 200 from the two-tower, seen items removed",
        "val_split": f"{len(val_users)} users held out of ranker training",
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "ranker_training.json").write_text(json.dumps(info, indent=2))
    (ARTIFACTS_DIR / "ranker_meta.json").write_text(json.dumps(
        {"n_features": int(X.shape[1]), "hidden": RANKER_HIDDEN,
         "feature_layout": "user_emb(64) item_emb(64) dot pop_log pop_pct "
                           "item_mean rcount hist umean genre_cos genre_count "
                           "rating_gap pop_x_genre hist_x_genre"}, indent=2))
    return info


def load_model(path: Path | None = None) -> WideAndDeep:
    path = path or ARTIFACTS_DIR / "ranker.pt"
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = WideAndDeep(ckpt["n_features"], ckpt["hidden"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def score_candidates(model: WideAndDeep, store: FeatureStore,
                     u_idx: int, cand: np.ndarray, u_emb: np.ndarray,
                     i_emb: np.ndarray, batch_size: int = 512) -> np.ndarray:
    out = []
    with torch.no_grad():
        for s in range(0, len(cand), batch_size):
            chunk = cand[s:s + batch_size]
            X, cross = build_matrix(store, np.full(len(chunk), u_idx),
                                    chunk, u_emb, i_emb)
            logits = model(torch.from_numpy(X), torch.from_numpy(cross))
            out.append(torch.sigmoid(logits).numpy())
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


def main() -> None:
    info = train()
    print(f"[ranker] trained in {info['train_seconds']}s, "
          f"final val AUC {info['final_val_auc']}")


if __name__ == "__main__":
    main()
