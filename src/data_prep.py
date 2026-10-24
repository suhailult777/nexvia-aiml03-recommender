"""Step 1: download MovieLens 100K, parse it, and build the time based split.

Split rule (per user, not global):
    sort that user's interactions by timestamp ascending
    the last round(0.20 * n) interactions become test
    everything before them becomes train

The split is per user, so a test interaction of a quiet user can carry an
earlier timestamp than a train interaction of a very active user. The exact
global cutoffs are written to data/processed/split_stats.json and quoted in
the README.
"""
from __future__ import annotations

import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (  # noqa: E402
    MIN_TEST_PER_USER,
    ML100K_URL,
    PROCESSED_DIR,
    RAW_DIR,
    TEST_FRACTION,
)

GENRES = [
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror",
    "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western",
]


def download_if_needed() -> Path:
    """Fetch ml-100k.zip into data/ and extract it next to the archive."""
    data_file = RAW_DIR / "u.data"
    if data_file.exists():
        print(f"[data] found existing {data_file}")
        return RAW_DIR

    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR.parent / "ml-100k.zip"
    if not zip_path.exists():
        print(f"[data] downloading {ML100K_URL}")
        with urllib.request.urlopen(ML100K_URL, timeout=120) as resp:
            payload = resp.read()
        zip_path.write_bytes(payload)
        print(f"[data] wrote {zip_path} ({len(payload) / 1e6:.1f} MB)")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW_DIR.parent)
    print(f"[data] extracted to {RAW_DIR}")
    return RAW_DIR


def load_ratings() -> pd.DataFrame:
    df = pd.read_csv(
        RAW_DIR / "u.data",
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "item_id": "int32", "rating": "int8",
               "timestamp": "int64"},
    )
    return df


def load_items() -> pd.DataFrame:
    items = pd.read_csv(
        RAW_DIR / "u.item",
        sep="|",
        encoding="latin-1",
        names=["item_id", "title", "release_date", "video_release_date",
               "imdb_url"] + GENRES,
        dtype={"item_id": "int32"},
    )
    return items


def build_split(ratings: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ordered = ratings.sort_values(["user_id", "timestamp", "item_id"],
                                  kind="mergesort").reset_index(drop=True)
    ordered["rank_in_user"] = ordered.groupby("user_id").cumcount()
    ordered["n_in_user"] = ordered.groupby("user_id")["item_id"].transform("size")

    # at least one test row, and never the user's whole history
    ordered["n_test"] = (ordered["n_in_user"] * TEST_FRACTION).round().astype(int)
    ordered["n_test"] = ordered["n_test"].clip(
        lower=MIN_TEST_PER_USER, upper=ordered["n_in_user"] - 1)

    is_test = ordered["rank_in_user"] >= (ordered["n_in_user"] - ordered["n_test"])
    test = ordered[is_test].copy()
    train = ordered[~is_test].copy()

    for frame in (train, test):
        frame.drop(columns=["rank_in_user", "n_in_user", "n_test"], inplace=True)

    stats = {
        "split_rule": (
            "per user: sort interactions by timestamp ascending, last "
            f"{int(TEST_FRACTION * 100)} percent (rounded, minimum "
            f"{MIN_TEST_PER_USER}) is test, the rest is train"
        ),
        "n_ratings": int(len(ratings)),
        "n_users": int(ratings["user_id"].nunique()),
        "n_items_rated": int(ratings["item_id"].nunique()),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "train_users": int(train["user_id"].nunique()),
        "test_users": int(test["user_id"].nunique()),
        "users_missing_from_train": sorted(
            set(test["user_id"].unique()) - set(train["user_id"].unique())),
        "test_fraction_actual": round(len(test) / len(ratings), 6),
        "global_max_train_timestamp": int(train["timestamp"].max()),
        "global_min_test_timestamp": int(test["timestamp"].min()),
        "global_max_train_datetime_utc": pd.to_datetime(
            int(train["timestamp"].max()), unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "global_min_test_datetime_utc": pd.to_datetime(
            int(test["timestamp"].min()), unit="s").strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp_overlap_note": (
            "the split is per user, so global train and test timestamp ranges "
            "overlap; every individual user's test rows are strictly later "
            "than that same user's train rows"
        ),
        "users_with_no_test_rows": int(
            ratings["user_id"].nunique() - test["user_id"].nunique()),
        "min_train_rows_per_user": int(train.groupby("user_id").size().min()),
        "max_train_rows_per_user": int(train.groupby("user_id").size().max()),
        "median_train_rows_per_user": float(train.groupby("user_id").size().median()),
        "min_test_rows_per_user": int(test.groupby("user_id").size().min()),
        "max_test_rows_per_user": int(test.groupby("user_id").size().max()),
    }
    return train, test, stats


def main() -> None:
    download_if_needed()
    ratings = load_ratings()
    items = load_items()
    train, test, stats = build_split(ratings)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)
    items.to_csv(PROCESSED_DIR / "items.csv", index=False)
    (PROCESSED_DIR / "split_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"[split] ratings={len(ratings)} train={len(train)} test={len(test)}")
    print(f"[split] train ts <= {stats['global_max_train_datetime_utc']} UTC")
    print(f"[split] test  ts >= {stats['global_min_test_datetime_utc']} UTC")
    print(f"[split] min train rows per user = {stats['min_train_rows_per_user']}")
    print(f"[split] wrote {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
