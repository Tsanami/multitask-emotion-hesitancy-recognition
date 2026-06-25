import os
import numpy as np
import pandas as pd

# Порядок колонок в MOSEI CSV
# CSV:  [Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise]  (индексы 0-6)
# Уже в нужном порядке — Neutral первым
MOSEI_EMOTION_COLS = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

# Сплиты EAAI лежат в файлах *_full.csv; внутренние имена частей оставляем прежними
MOSEI_PART_FILES = {
    "train":      "train_full",
    "validation": "dev_full",
    "test":       "test_full",
}


def get_cmu_mosei_data(path, part="train"):
    """
    part: 'train' | 'validation' | 'test'
    Возвращает: texts [N], labels [N, 7] float (интенсивности, Neutral первым)
    """
    if part not in MOSEI_PART_FILES:
        raise ValueError("part must be 'train', 'validation', or 'test'")
    df = pd.read_csv(os.path.join(path, MOSEI_PART_FILES[part] + ".csv"))
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    texts  = df["text"].values
    labels = np.dstack([df[col].to_numpy() for col in MOSEI_EMOTION_COLS])  # [1, N, 7]
    return texts, labels


def get_bah_data(path, part="train"):
    """
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
