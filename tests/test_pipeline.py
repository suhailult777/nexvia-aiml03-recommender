"""Pipeline tests.

They run against the artifacts produced by the training scripts, so run
run_all.py (or the individual scripts in order) first. Every assertion below
checks a property that would silently break the recommender if it regressed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

REQUIRED_ARTIFACTS = ["features.npz", "user_embeddings.npy",
                      "item_embeddings.npy", "item_index.faiss", "ranker.pt"]


@pytest.fixture(scope="module")
def service():
    missing = [f for f in REQUIRED_ARTIFACTS if not (ARTIFACTS / f).exists()]
    if missing:
        pytest.exit(f"missing artifacts {missing}; run run_all.py first", returncode=2)
    from recommender import RecommenderService
    return RecommenderService()


def test_artifacts_exist():
    for name in REQUIRED_ARTIFACTS:
        assert (ARTIFACTS / name).exists(), f"missing {name}"


def test_faiss_index_size_equals_item_count(service):
    n_items = len(service.store.item_ids)
    assert service.faiss.index.ntotal == n_items
    assert n_items == 1682
    assert service.item_emb.shape == (n_items, 64)


def test_embeddings_are_unit_norm(service):
    for matrix in (service.user_emb, service.item_emb):
        norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-5)


def test_retrieval_returns_k_unique_ids(service):
    k = 200
    scores, ids = service.faiss.search_batch(service.user_emb[:10], k)
    assert ids.shape == (10, k)
    for row in ids:
        assert len(set(row.tolist())) == k
        assert row.min() >= 0 and row.max() < service.n_items
    # cosine similarities must be sorted descending
    for row in scores:
        assert np.all(np.diff(row) <= 1e-5)


def test_cold_start_fallback_triggers_for_brand_new_user(service):
    resp = service.recommend(user_id=987654, k=5)
    assert resp["cold_start"] is True
    assert resp["strategy"] == "cold_start_popularity"
    assert resp["history_length"] == 0
    assert len(resp["recommendations"]) == 5
    assert len({r["item_id"] for r in resp["recommendations"]}) == 5
    # an unknown user must still get the globally most popular training item
    top_train_item = int(service.store.item_ids[int(np.argmax(service.store.item_pop))])
    assert resp["recommendations"][0]["item_id"] == top_train_item


def test_known_user_recommendations_exclude_seen_items(service):
    user_id = int(service.store.user_ids[0])
    ui = service.store.user_index(user_id)
    seen = {int(service.store.item_ids[i]) for i in service.user_seen[ui]}
    resp = service.recommend(user_id=user_id, k=10)
    assert resp["cold_start"] is False
    assert resp["strategy"] == "two_stage_faiss_ranker"
    rec_ids = [r["item_id"] for r in resp["recommendations"]]
    assert len(rec_ids) == 10
    assert len(set(rec_ids)) == 10
    assert not (set(rec_ids) & seen)
    assert all(0.0 <= r["score"] <= 1.0 for r in resp["recommendations"])


def test_metrics_values_are_probabilities():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    systems = metrics["systems"]
    assert set(systems) == {"popularity", "two_tower_retrieval", "two_stage"}
    for name, block in systems.items():
        for key in ("recall@10", "recall@50", "ndcg@10", "ndcg@50"):
            value = block[key]
            assert 0.0 <= value <= 1.0, f"{name} {key} out of range: {value}"
    assert metrics["n_users_evaluated"] == 943
    # the two-tower should beat raw popularity at the top of the list
    assert systems["two_tower_retrieval"]["ndcg@10"] > systems["popularity"]["ndcg@10"]


def test_cold_start_report_is_consistent():
    metrics = json.loads((RESULTS / "metrics.json").read_text())
    cold = metrics["cold_start"]
    assert cold["real_test_users_below_threshold"] == 0
    assert cold["real_test_users_total"] == 943
    assert cold["simulation"]["n_users"] == 60
    for key in ("recall@10", "ndcg@10"):
        value = cold["simulation"]["fallback"][key]
        assert 0.0 <= value <= 1.0
