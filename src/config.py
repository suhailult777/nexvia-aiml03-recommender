"""Central configuration for the AIML-03 two-stage recommender.

Every constant that affects a reported number lives here so the README can
reference one file instead of hunting through scripts.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "ml-100k"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"
RESULTS_DIR = ROOT / "results"

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"

# ---------------------------------------------------------------- split
TEST_FRACTION = 0.20          # last 20 percent of each user's interactions
MIN_TEST_PER_USER = 1
COLD_START_MAX_HISTORY = 3    # users with < 3 training interactions -> fallback

# ---------------------------------------------------------------- retrieval
EMBED_DIM = 64
TWO_TOWER_BATCH = 1024
TWO_TOWER_EPOCHS = 25
TWO_TOWER_LR = 0.01
TEMPERATURE = 0.10            # softmax temperature on cosine logits
RETRIEVAL_K = 200             # candidates passed to the ranker

# ---------------------------------------------------------------- ranking
RANKER_HIDDEN = [256, 64]
RANKER_EPOCHS = 12
RANKER_BATCH = 2048
RANKER_LR = 1e-3
RANKER_POS_PER_USER = 30
RANKER_HARD_NEG_PER_USER = 30
RANKER_RANDOM_NEG_PER_USER = 20

# ---------------------------------------------------------------- evaluation
K_LIST = (10, 50)
POPULARITY_MIN_INTERACTIONS = 1
SEED = 42
LATENCY_USERS = 250
