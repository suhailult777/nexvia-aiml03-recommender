"""Step 10: plots for the report.

results/training_loss.png   two-tower loss and ranker train/val loss
results/metrics_comparison.png  Recall and NDCG bars for the three systems

Every value comes from results/training_history.json, results/ranker_training.json
and results/metrics.json, which are written by the training and evaluation runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import K_LIST, RESULTS_DIR  # noqa: E402

COLORS = {"popularity": "#9aa5b1", "two_tower_retrieval": "#3b82f6",
          "two_stage": "#f97316"}


def plot_training() -> Path:
    tt = json.loads((RESULTS_DIR / "training_history.json").read_text())
    rk = json.loads((RESULTS_DIR / "ranker_training.json").read_text())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ep = [h["epoch"] for h in tt["history"]]
    axes[0].plot(ep, [h["loss"] for h in tt["history"]], marker="o", ms=3,
                 color=COLORS["two_tower_retrieval"], lw=1.8)
    axes[0].set_title(f"Two-tower softmax loss ({tt['epochs']} epochs)")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("training loss")
    axes[0].grid(alpha=0.25)

    rep = [h["epoch"] for h in rk["history"]]
    axes[1].plot(rep, [h["train_loss"] for h in rk["history"]], marker="o", ms=3,
                 label="train", color=COLORS["two_stage"], lw=1.8)
    axes[1].plot(rep, [h["val_loss"] for h in rk["history"]], marker="s", ms=3,
                 label="validation", color="#334155", lw=1.8, ls="--")
    axes[1].set_title(f"Ranker BCE loss (val AUC {rk['final_val_auc']:.3f})")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("loss")
    axes[1].legend(frameon=False)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Training curves, MovieLens 100K, per-user time split", y=1.02)
    fig.tight_layout()
    out = RESULTS_DIR / "training_loss.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_metrics() -> Path:
    m = json.loads((RESULTS_DIR / "metrics.json").read_text())["systems"]
    labels = (["Recall@10", "Recall@50"] +
              [f"NDCG@{k}" for k in K_LIST])
    keys = ([f"recall@{k}" for k in K_LIST] + [f"ndcg@{k}" for k in K_LIST])
    systems = ["popularity", "two_tower_retrieval", "two_stage"]
    names = {"popularity": "popularity", "two_tower_retrieval": "two-tower",
             "two_stage": "two-stage"}

    x = np.arange(len(labels))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for i, s in enumerate(systems):
        vals = [m[s][k] for k in keys]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=names[s],
                      color=COLORS[s], edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("score (higher is better)")
    ax.set_ylim(0, max(m[s][k] for s in systems for k in keys) * 1.25)
    ax.set_title("Popularity vs two-tower retrieval vs full two-stage pipeline")
    ax.legend(frameon=False, ncol=3, loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = RESULTS_DIR / "metrics_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"wrote {plot_training()}")
    print(f"wrote {plot_metrics()}")


if __name__ == "__main__":
    main()
