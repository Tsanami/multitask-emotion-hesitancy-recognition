"""
Построение графиков для НИРа из History JSON.

    python -m utils.plotting best_models/E4_fusion_ssl_seed42_history.json

Строит:
  - кривые train/val mF1 по эпохам (обе задачи)
  - SSL coverage по эпохам (доля уверенных псевдо-меток)
  - распределение псевдо-меток эмоций по классам (последняя эпоха)
"""
import sys
import json
import numpy as np

EMO_NAMES = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]


def plot_history(history_path, out_prefix=None):
    import matplotlib.pyplot as plt

    with open(history_path) as f:
        h = json.load(f)

    out_prefix = out_prefix or history_path.replace(".json", "")
    epochs = range(1, len(h["train_loss"]) + 1)

    # ── 1. Кривые mF1 ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, h["train_emo_mf1"], label="train EMO mF1", linestyle="--")
    ax.plot(epochs, h["val_emo_mf1"],   label="val EMO mF1")
    ax.plot(epochs, h["train_ah_mf1"],  label="train AH mF1", linestyle="--")
    ax.plot(epochs, h["val_ah_mf1"],    label="val AH mF1")
    ax.set_xlabel("Эпоха"); ax.set_ylabel("mF1"); ax.legend(); ax.grid(alpha=0.3)
    ax.set_title("Кривые обучения")
    fig.tight_layout(); fig.savefig(f"{out_prefix}_mf1.png", dpi=150)
    print(f"saved {out_prefix}_mf1.png")

    # ── 2. SSL coverage ───────────────────────────────────────────────────────
    if "train_cov_ssl_emo" in h and any(h["train_cov_ssl_emo"]):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, h["train_cov_ssl_emo"], label="EMO pseudo-label coverage")
        ax.plot(epochs, h["train_cov_ssl_ah"],  label="AH pseudo-label coverage")
        ax.set_xlabel("Эпоха"); ax.set_ylabel("Доля уверенных псевдо-меток")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
        ax.set_title("SSL coverage по эпохам")
        fig.tight_layout(); fig.savefig(f"{out_prefix}_ssl_coverage.png", dpi=150)
        print(f"saved {out_prefix}_ssl_coverage.png")

    # ── 3. Распределение псевдо-меток эмоций (последняя эпоха) ────────────────
    if "train_pseudo_emo_hist" in h and h["train_pseudo_emo_hist"]:
        last_hist = np.array(h["train_pseudo_emo_hist"][-1])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(EMO_NAMES, last_hist)
        ax.set_ylabel("Кол-во псевдо-меток"); ax.set_title("Распределение псевдо-меток эмоций (последняя эпоха)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout(); fig.savefig(f"{out_prefix}_pseudo_hist.png", dpi=150)
        print(f"saved {out_prefix}_pseudo_hist.png")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python -m utils.plotting <history.json> [out_prefix]")
        sys.exit(1)
    plot_history(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
