"""Step 3: two-tower retrieval model in TensorFlow / Keras.

User tower : user id embedding + history stats (log history length, mean
             rating, 19 dim genre profile)
Item tower : item id embedding + genre multi hot + popularity stats
Objective  : softmax over the full 1682 item catalog with cosine logits
             (temperature 0.1). This is the sampled-softmax objective with
             every item sampled, which is affordable at this catalog size.

Outputs are L2 normalized so a dot product is a cosine similarity, which is
exactly what the FAISS inner-product index expects.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    ARTIFACTS_DIR,
    EMBED_DIM,
    PROCESSED_DIR,
    RESULTS_DIR,
    SEED,
    TEMPERATURE,
    TWO_TOWER_BATCH,
    TWO_TOWER_EPOCHS,
    TWO_TOWER_LR,
)
from features import FeatureStore, load as load_features  # noqa: E402


def _dense_user(store: FeatureStore) -> np.ndarray:
    h = np.log1p(store.user_hist_len) / np.log1p(store.user_hist_len.max())
    r = store.user_mean_rating / 5.0
    return np.concatenate([h[:, None], r[:, None], store.user_genre],
                          axis=1).astype(np.float32)


def _dense_item(store: FeatureStore) -> np.ndarray:
    p = np.log1p(store.item_pop) / np.log1p(max(store.item_pop.max(), 1.0))
    r = store.item_mean_rating / 5.0
    g = store.genre.sum(axis=1) / store.genre.shape[1]
    return np.concatenate([p[:, None], store.item_pop_pct[:, None], r[:, None],
                           g[:, None], store.genre], axis=1).astype(np.float32)


class TwoTowerRetriever:
    def __init__(self, store: FeatureStore, embed_dim: int = EMBED_DIM):
        self.store = store
        self.embed_dim = embed_dim
        self.user_dense = _dense_user(store)
        self.item_dense = _dense_item(store)
        self.user_tower = self._build_user_tower()
        self.item_tower = self._build_item_tower()
        self.history: list[dict] = []

    # ------------------------------------------------------------ towers
    def _build_user_tower(self) -> keras.Model:
        uid = keras.Input(shape=(), dtype="int32", name="user_id")
        dense = keras.Input(shape=(self.user_dense.shape[1],), dtype="float32",
                            name="user_stats")
        emb = keras.layers.Embedding(
            self.store.n_users, self.embed_dim, name="user_embedding")(uid)
        stats = keras.layers.Dense(128, activation="relu",
                                   name="user_stats_mlp")(dense)
        x = keras.layers.Concatenate()([emb, stats])
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dense(self.embed_dim)(x)
        out = keras.layers.UnitNormalization(axis=1, name="user_vec")(x)
        return keras.Model([uid, dense], out, name="user_tower")

    def _build_item_tower(self) -> keras.Model:
        iid = keras.Input(shape=(), dtype="int32", name="item_id")
        dense = keras.Input(shape=(self.item_dense.shape[1],), dtype="float32",
                            name="item_stats")
        emb = keras.layers.Embedding(
            self.store.n_items, self.embed_dim, name="item_embedding")(iid)
        # genre and popularity enter through a dense block so unseen items
        # still get a usable vector from content features
        stats = keras.layers.Dense(128, activation="relu",
                                   name="item_stats_mlp")(dense)
        x = keras.layers.Concatenate()([emb, stats])
        x = keras.layers.Dense(128, activation="relu")(x)
        x = keras.layers.Dense(self.embed_dim)(x)
        out = keras.layers.UnitNormalization(axis=1, name="item_vec")(x)
        return keras.Model([iid, dense], out, name="item_tower")

    # ------------------------------------------------------------ training
    def train(self, train_df: pd.DataFrame, epochs: int = TWO_TOWER_EPOCHS,
              batch_size: int = TWO_TOWER_BATCH, lr: float = TWO_TOWER_LR,
              verbose: bool = True) -> dict:
        u_idx_map = {int(u): i for i, u in enumerate(self.store.user_ids)}
        i_idx_map = {int(m): i for i, m in enumerate(self.store.item_ids)}

        pairs = train_df[["user_id", "item_id"]].drop_duplicates()
        u_all = np.array([u_idx_map[int(u)] for u in pairs["user_id"]], dtype=np.int32)
        i_all = np.array([i_idx_map[int(m)] for m in pairs["item_id"]], dtype=np.int32)
        del i_idx_map

        all_item_ids = tf.constant(np.arange(self.store.n_items, dtype=np.int32))
        item_dense_t = tf.constant(self.item_dense)
        user_dense_t = tf.constant(self.user_dense)

        optimizer = keras.optimizers.Adam(learning_rate=lr)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        n = len(u_all)
        steps = int(np.ceil(n / batch_size))
        rng = np.random.default_rng(SEED)

        @tf.function(reduce_retracing=True)
        def train_step(u_batch, i_batch):
            u_dense_batch = tf.gather(user_dense_t, u_batch)
            with tf.GradientTape() as tape:
                u_vec = self.user_tower([u_batch, u_dense_batch], training=True)
                i_vec_all = self.item_tower([all_item_ids, item_dense_t],
                                            training=True)
                logits = tf.matmul(u_vec, i_vec_all, transpose_b=True) / TEMPERATURE
                loss = loss_fn(i_batch, logits)
            grads = tape.gradient(loss, self.user_tower.trainable_variables +
                                  self.item_tower.trainable_variables)
            optimizer.apply_gradients(
                zip(grads, self.user_tower.trainable_variables +
                    self.item_tower.trainable_variables))
            return loss

        t0 = time.time()
        for epoch in range(1, epochs + 1):
            order = rng.permutation(n)
            running = 0.0
            for s in range(steps):
                batch_idx = order[s * batch_size:(s + 1) * batch_size]
                loss = train_step(tf.constant(u_all[batch_idx]),
                                  tf.constant(i_all[batch_idx]))
                running += float(loss)
            mean_loss = running / steps
            self.history.append({"epoch": epoch, "loss": round(mean_loss, 6)})
            if verbose:
                print(f"[two-tower] epoch {epoch:2d}/{epochs} loss={mean_loss:.4f} "
                      f"({time.time() - t0:.1f}s elapsed)", flush=True)

        self.train_seconds = round(time.time() - t0, 2)
        return {"history": self.history, "train_seconds": self.train_seconds,
                "epochs": epochs, "batch_size": batch_size,
                "pairs_per_epoch": n, "temperature": TEMPERATURE,
                "optimizer": f"Adam lr={lr}", "embed_dim": self.embed_dim}

    # ------------------------------------------------------------ serving
    def user_vectors(self) -> np.ndarray:
        return self.user_tower.predict(
            [np.arange(self.store.n_users, dtype=np.int32), self.user_dense],
            batch_size=512, verbose=0).astype(np.float32)

    def item_vectors(self) -> np.ndarray:
        return self.item_tower.predict(
            [np.arange(self.store.n_items, dtype=np.int32), self.item_dense],
            batch_size=512, verbose=0).astype(np.float32)

    def save(self, out_dir: Path | None = None) -> dict:
        out = out_dir or ARTIFACTS_DIR
        out.mkdir(parents=True, exist_ok=True)
        uv = self.user_vectors()
        iv = self.item_vectors()
        np.save(out / "user_embeddings.npy", uv)
        np.save(out / "item_embeddings.npy", iv)
        self.user_tower.save(out / "user_tower.keras")
        self.item_tower.save(out / "item_tower.keras")
        return {"user_embeddings": str(out / "user_embeddings.npy"),
                "item_embeddings": str(out / "item_embeddings.npy"),
                "user_tower": str(out / "user_tower.keras"),
                "item_tower": str(out / "item_tower.keras"),
                "user_embedding_shape": list(uv.shape),
                "item_embedding_shape": list(iv.shape)}


def load_embeddings(path: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    path = path or ARTIFACTS_DIR
    return (np.load(Path(path) / "user_embeddings.npy"),
            np.load(Path(path) / "item_embeddings.npy"))


def main() -> None:
    keras.utils.set_random_seed(SEED)
    store = load_features()
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    model = TwoTowerRetriever(store)
    info = model.train(train)
    saved = model.save()
    info["saved"] = saved
    info["n_users"] = store.n_users
    info["n_items"] = store.n_items

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "training_history.json").write_text(json.dumps(info, indent=2))
    print(f"[two-tower] trained in {info['train_seconds']}s, "
          f"final loss {info['history'][-1]['loss']}")
    print(f"[two-tower] item matrix {saved['item_embedding_shape']}")


if __name__ == "__main__":
    main()
