"""Step 7: evaluation of the three systems plus the cold-start simulation.

(a) popularity baseline        global train interaction count, seen items removed
(b) two-tower retrieval alone  full catalog scored by cosine, seen items removed
(c) full two-stage             FAISS top 200 candidates, wide and deep re-rank

Every number written to results/metrics.json comes from this script.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cold_start import ColdStartRecommender, is_cold_user  # noqa: E402
from config import (  # noqa: E402
    ARTIFACTS_DIR,
    COLD_START_MAX_HISTORY,
    K_LIST,
    PROCESSED_DIR,
    RESULTS_DIR,
    RETRIEVAL_K,
    SEED,
    TEST_FRACTION,
)
from faiss_index import FaissRetriever  # noqa: E402
from features import load as load_features  # noqa: E402
from metrics import evaluate_rankings, top_k_from_scores  # noqa: E402
from ranker import load_model, score_candidates  # noqa: E402

COLD_SIM_USERS = 60
COLD_SIM_SEEDS = 2


def build_ground_truth(store, train: pd.DataFrame, test: pd.DataFrame):
    u_map = {int(u): i for i, u in enumerate(store.user_ids)}
    i_map = {int(m): i for i, m in enumerate(store.item_ids)}
    seen: dict[int, set[int]] = {}
    for u, m in zip(train["user_id"].to_numpy(), train["item_id"].to_numpy()):
        seen.setdefault(u_map[int(u)], set()).add(i_map[int(m)])
    rel: dict[int, set[int]] = {}
    for u, m in zip(test["user_id"].to_numpy(), test["item_id"].to_numpy()):
        rel.setdefault(u_map[int(u)], set()).add(i_map[int(m)])
    return u_map, i_map, seen, rel


def main() -> None:
    t_start = time.time()
    store = load_features()
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")
    items = pd.read_csv(PROCESSED_DIR / "items.csv")
    u_emb = np.load(ARTIFACTS_DIR / "user_embeddings.npy")
    i_emb = np.load(ARTIFACTS_DIR / "item_embeddings.npy")
    faiss_retriever = FaissRetriever(i_emb)
    ranker = load_model()
    cold = ColdStartRecommender(store)

    u_map, i_map, seen, rel = build_ground_truth(store, train, test)
    n_users = len(store.user_ids)
    relevant = [rel.get(ui, set()) for ui in range(n_users)]
    test_sizes = [len(r) for r in relevant]

    # ---------------------------------------------------------- (a) popularity
    pop_scores = store.item_pop.astype(np.float32)
    pop_ranked = []
    for ui in range(n_users):
        blocked = np.fromiter(seen.get(ui, set()), dtype=np.int64)
        pop_ranked.append(top_k_from_scores(pop_scores, max(K_LIST), blocked))

    # ---------------------------------------------------------- (b) two-tower alone
    tt_scores = u_emb @ i_emb.T
    tt_ranked = []
    for ui in range(n_users):
        blocked = np.fromiter(seen.get(ui, set()), dtype=np.int64)
        tt_ranked.append(top_k_from_scores(tt_scores[ui], max(K_LIST), blocked))

    # ---------------------------------------------------------- (c) two-stage
    two_stage_ranked = []
    cand_recall_hits = 0
    cand_sizes = []
    n_seen_total = 0
    for ui in range(n_users):
        blocked = np.fromiter(seen.get(ui, set()), dtype=np.int64)
        n_seen_total += len(blocked)
        # over-fetch so that masking seen items still leaves RETRIEVAL_K candidates
        _, ids = faiss_retriever.search(u_emb[ui], RETRIEVAL_K + len(blocked))
        mask = ~np.isin(ids, blocked)
        cand = ids[mask][:RETRIEVAL_K]
        cand_sizes.append(len(cand))
        cand_recall_hits += len(set(cand.tolist()) & relevant[ui])

        scores = score_candidates(ranker, store, ui, cand, u_emb, i_emb)
        order = np.argsort(-scores)[:max(K_LIST)]
        two_stage_ranked.append(cand[order].tolist())

    systems = {
        "popularity": evaluate_rankings(pop_ranked, relevant, K_LIST),
        "two_tower_retrieval": evaluate_rankings(tt_ranked, relevant, K_LIST),
        "two_stage": evaluate_rankings(two_stage_ranked, relevant, K_LIST),
    }

    total_relevant = int(sum(test_sizes))
    candidate_block = {
        "k": RETRIEVAL_K,
        "mean_candidates_per_user": round(float(np.mean(cand_sizes)), 2),
        "mean_seen_items_per_user": round(n_seen_total / n_users, 2),
        "candidate_recall_all_test_items": round(cand_recall_hits / total_relevant, 6),
        "note": ("share of all test interactions that appear somewhere in the 200 "
                 "candidates handed to the ranker; this is the ceiling for stage two"),
    }

    # ---------------------------------------------------------- cold start
    hist = store.user_hist_len
    real_cold_users = [int(store.user_ids[i]) for i in range(n_users)
                       if is_cold_user(int(hist[i]))]

    rng = np.random.default_rng(SEED)
    sim_users = rng.choice(n_users, size=COLD_SIM_USERS, replace=False)
    fallback_ranked, pop_sim_ranked = [], []
    for ui in sim_users:
        ui = int(ui)
        # two most recent training items act as the signup clicks
        user_train = train[train["user_id"] == store.user_ids[ui]].sort_values("timestamp")
        seed_raw = user_train["item_id"].tail(COLD_SIM_SEEDS).tolist()
        seeds = np.array([i_map[int(m)] for m in seed_raw], dtype=np.int64)
        # a cold user has no history, so only the seed items are excluded
        recs, _ = cold.recommend(seeds, max(K_LIST), exclude=seeds)
        fallback_ranked.append(recs)
        pop_only, _ = cold.recommend(None, max(K_LIST), exclude=seeds)
        pop_sim_ranked.append(pop_only)

    sim_relevant = [relevant[int(ui)] for ui in sim_users]
    cold_block = {
        "threshold_rule": f"users with fewer than {COLD_START_MAX_HISTORY} training interactions",
        "real_test_users_below_threshold": len(real_cold_users),
        "real_test_users_total": n_users,
        "real_cold_user_ids": real_cold_users,
        "min_train_history_in_dataset": int(hist.min()),
        "simulation": {
            "n_users": COLD_SIM_USERS,
            "seed_items_per_user": COLD_SIM_SEEDS,
            "scoring": "0.6 * normalized popularity + 0.4 * genre affinity",
            "fallback": evaluate_rankings(fallback_ranked, sim_relevant, K_LIST),
            "popularity_only_same_users": evaluate_rankings(pop_sim_ranked, sim_relevant, K_LIST),
            "note": ("simulated by giving existing users a 2 item seed profile; the "
                     "fallback never reads a user id or user embedding, so this is "
                     "the same code path a brand new user hits"),
        },
    }

    ranker_info = json.loads((RESULTS_DIR / "ranker_training.json").read_text())
    tt_info = json.loads((RESULTS_DIR / "training_history.json").read_text())
    split_stats = json.loads((PROCESSED_DIR / "split_stats.json").read_text())

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {
            "name": "MovieLens 100K",
            "n_ratings": split_stats["n_ratings"],
            "n_users": split_stats["n_users"],
            "n_items": store.n_items,
            "n_train": split_stats["n_train"],
            "n_test": split_stats["n_test"],
            "split_rule": split_stats["split_rule"],
            "global_max_train_datetime_utc": split_stats["global_max_train_datetime_utc"],
            "global_min_test_datetime_utc": split_stats["global_min_test_datetime_utc"],
            "test_fraction_actual": split_stats["test_fraction_actual"],
            "interaction_definition": "every rating row is an implicit positive interaction",
        },
        "config": {
            "embed_dim": int(u_emb.shape[1]),
            "retrieval_k": RETRIEVAL_K,
            "test_fraction": TEST_FRACTION,
            "cold_start_max_history": COLD_START_MAX_HISTORY,
            "k_list": list(K_LIST),
            "seed": SEED,
        },
        "systems": systems,
        "candidate_coverage": candidate_block,
        "cold_start": cold_block,
        "ranking_model": {
            "val_auc": ranker_info["final_val_auc"],
            "train_rows": ranker_info["train_rows"],
            "val_rows": ranker_info["val_rows"],
            "train_seconds": ranker_info["train_seconds"],
        },
        "retrieval_model": {
            "final_train_loss": tt_info["history"][-1]["loss"],
            "epochs": tt_info["epochs"],
            "train_seconds": tt_info["train_seconds"],
            "pairs_per_epoch": tt_info["pairs_per_epoch"],
        },
        "eval_seconds": round(time.time() - t_start, 2),
        "n_users_evaluated": systems["popularity"]["n_users_evaluated"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2))

    header = f"{'system':<22}{'R@10':>8}{'R@50':>8}{'NDCG@10':>10}{'NDCG@50':>10}"
    print(header)
    print("-" * len(header))
    for name, m in systems.items():
        print(f"{name:<22}{m['recall@10']:>8.4f}{m['recall@50']:>8.4f}"
              f"{m['ndcg@10']:>10.4f}{m['ndcg@50']:>10.4f}")
    print(f"\ncandidate recall@{RETRIEVAL_K} over all test items: "
          f"{candidate_block['candidate_recall_all_test_items']:.4f}")
    print(f"real cold-start users (< {COLD_START_MAX_HISTORY} train rows): "
          f"{len(real_cold_users)} of {n_users}")
    print(f"cold simulation ({COLD_SIM_USERS} users, {COLD_SIM_SEEDS} seeds): "
          f"fallback R@10={cold_block['simulation']['fallback']['recall@10']:.4f} "
          f"vs popularity R@10="
          f"{cold_block['simulation']['popularity_only_same_users']['recall@10']:.4f}")
    print(f"wrote {RESULTS_DIR / 'metrics.json'} in {out['eval_seconds']}s")


if __name__ == "__main__":
    main()
