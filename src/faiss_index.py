"""Step 4: FAISS index over the item embeddings.

IndexFlatIP on L2 normalized vectors == exact cosine nearest neighbour search.
At 1682 items an exact index is the right call: an approximate index would add
recall loss for no measurable latency win at this scale.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import ARTIFACTS_DIR  # noqa: E402


def normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(norms, 1e-12)).astype(np.float32)


class FaissRetriever:
    def __init__(self, item_embeddings: np.ndarray):
        self.item_embeddings = normalize(item_embeddings)
        self.dim = self.item_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(self.item_embeddings)

    def search(self, user_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        q = normalize(np.atleast_2d(user_vector.astype(np.float32)))
        scores, ids = self.index.search(q, min(k, self.index.ntotal))
        return scores[0], ids[0]

    def search_batch(self, user_vectors: np.ndarray, k: int
                     ) -> tuple[np.ndarray, np.ndarray]:
        q = normalize(user_vectors.astype(np.float32))
        return self.index.search(q, min(k, self.index.ntotal))

    def save(self, path: Path | None = None) -> Path:
        path = path or ARTIFACTS_DIR / "item_index.faiss"
        faiss.write_index(self.index, str(path))
        meta = {"ntotal": int(self.index.ntotal), "dim": int(self.dim),
                "metric": "inner_product", "vectors_normalized": True,
                "index_type": "IndexFlatIP"}
        Path(path).with_suffix(".json").write_text(json.dumps(meta, indent=2))
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "FaissRetriever":
        path = path or ARTIFACTS_DIR / "item_index.faiss"
        index = faiss.read_index(str(path))
        obj = cls.__new__(cls)
        obj.index = index
        obj.dim = index.d
        obj.item_embeddings = index.reconstruct_n(0, index.ntotal)
        return obj


def main() -> None:
    iv = np.load(ARTIFACTS_DIR / "item_embeddings.npy")
    retriever = FaissRetriever(iv)
    path = retriever.save()
    print(f"[faiss] {retriever.index.ntotal} vectors of dim {retriever.dim} "
          f"-> {path}")
    # smoke test: a user vector should retrieve itself-ish items, not errors
    uv = np.load(ARTIFACTS_DIR / "user_embeddings.npy")
    scores, ids = retriever.search(uv[0], 5)
    print(f"[faiss] smoke query user 0: ids={ids.tolist()} "
          f"scores={[round(float(s), 4) for s in scores]}")


if __name__ == "__main__":
    main()
