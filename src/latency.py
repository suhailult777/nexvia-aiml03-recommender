"""Step 9: latency benchmark.

Measures, per user, single request at a time, on CPU:
    stage 1  FAISS search + seen item masking + slice to 200 candidates
    stage 2  ranker scoring of those 200 candidates + top k sort
    pipeline the whole RecommenderService.recommend call

250 users, 10 warmup calls. p50 and p95 are reported because a mean alone
hides the tail.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS_DIR, LATENCY_USERS, RESULTS_DIR, RETRIEVAL_K, SEED  # noqa: E402
from features import load as load_features  # noqa: E402
from recommender import RecommenderService  # noqa: E402
from ranker import score_candidates  # noqa: E402


def pct(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 3)


def main() -> None:
    svc = RecommenderService()
    store = svc.store
    rng = np.random.default_rng(SEED)
    users = rng.choice(store.n_users, size=min(LATENCY_USERS, store.n_users),
                       replace=False)

    # warmup: first calls pay for lazy allocation inside FAISS and torch
    for ui in users[:10]:
        svc.recommend(int(store.user_ids[ui]), k=10)

    stage1, stage2, pipeline = [], [], []
    for ui in users:
        ui = int(ui)
        blocked = svc.user_seen.get(ui, np.array([], dtype=np.int64))

        t0 = time.perf_counter()
        _, ids = svc.faiss.search(svc.user_emb[ui], RETRIEVAL_K + len(blocked))
        cand = ids[~np.isin(ids, blocked)][:RETRIEVAL_K]
        t1 = time.perf_counter()

        scores = score_candidates(svc.ranker, store, ui, cand,
                                  svc.user_emb, svc.item_emb)
        order = np.argsort(-scores)[:10]
        _ = cand[order]
        t2 = time.perf_counter()

        svc.recommend(int(store.user_ids[ui]), k=10)
        t3 = time.perf_counter()

        stage1.append((t1 - t0) * 1000)
        stage2.append((t2 - t1) * 1000)
        pipeline.append((t3 - t2) * 1000)

    def block(v: list[float]) -> dict:
        return {"mean_ms": round(float(np.mean(v)), 3), "p50_ms": pct(v, 50),
                "p95_ms": pct(v, 95), "min_ms": round(float(np.min(v)), 3),
                "max_ms": round(float(np.max(v)), 3)}

    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "users_measured": int(len(users)),
        "warmup_calls": 10,
        "k_returned": 10,
        "candidates_per_request": RETRIEVAL_K,
        "stage1_retrieval": block(stage1),
        "stage2_ranking": block(stage2),
        "full_pipeline": block(pipeline),
        "environment": {
            "device": "CPU only, no GPU used",
            "concurrency": "single threaded, one request at a time, no batching",
            "python": "3.12",
            "faiss_index": "IndexFlatIP, 1682 vectors, exact search",
            "note": ("measured in process through RecommenderService, which is the "
                     "same object the FastAPI app serves from; HTTP overhead is not "
                     "included"),
        },
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "latency.json").write_text(json.dumps(out, indent=2))

    for name in ("stage1_retrieval", "stage2_ranking", "full_pipeline"):
        b = out[name]
        print(f"{name:<18} mean {b['mean_ms']:>7.3f} ms  p50 {b['p50_ms']:>7.3f} ms  "
              f"p95 {b['p95_ms']:>7.3f} ms")
    print(f"wrote {RESULTS_DIR / 'latency.json'} ({len(users)} users)")


if __name__ == "__main__":
    main()
