import os
import numpy as np
import pandas as pd

# Порядок колонок в MOSEI CSV → нужный порядок в статье
# CSV:  [Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise]  (индексы 0-6)
# Уже в нужном порядке — Neutral первым, как в ноутбуке inference.ipynb
MOSEI_EMOTION_COLS = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]


def get_cmu_mosei_data(path, part="train"):
    """
    Cell 13.
    part: 'train' | 'validation' | 'test'
    Возвращает: texts [N], labels [N, 7] float (интенсивности, Neutral первым)
    """
    if part not in ("train", "validation", "test"):
        raise ValueError("part must be 'train', 'validation', or 'test'")
    df = pd.read_csv(os.path.join(path, part + ".csv"))
    texts  = df["text"].values
    labels = np.dstack([df[col].to_numpy() for col in MOSEI_EMOTION_COLS])  # [1, N, 7]
    return texts, labels


def get_bah_data(path, part="train"):
    """
    Cell 13.
    part: 'train' | 'val' | 'test'
    Возвращает: texts [N], labels [N] int (0 или 1)
    """
    if part not in ("train", "val", "test"):
        raise ValueError("part must be 'train', 'val', or 'test'")

    with open(os.path.join(path, part + ".txt")) as f:
        rows = [line.replace(",", "|", 2).split("|") for line in f.read().splitlines()]

    df = pd.DataFrame(rows, columns=["id", "label", "text"])
    df["label"] = df["label"].astype(int)
    return df["text"].values, [df["label"].to_numpy()]
