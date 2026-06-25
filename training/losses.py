import numpy as np
import torch
import torch.nn as nn


def compute_emo_weights(emo_train_dataset, device):
    """
    Веса классов эмоций по формуле статьи: w_c = (K - k_c) / k_c
    где K — общее число сэмплов, k_c — число сэмплов класса c.

    k_c считается по MULTI-LABEL (intensity > 0), а НЕ по argmax-доминанте:
    при argmax редкие классы (Surprise/Fear) получают многократно завышенный вес
    (на EAAI: Surprise w≈64 вместо ≈9) и модель начинает их гиперпредсказывать —
    UAR растёт за счёт обвала precision. См. конвенции проекта (CLAUDE.md).
    """
    labels_matrix = np.array([
        emo_train_dataset[i]["emo_label"].numpy()
        for i in range(len(emo_train_dataset))
    ])  # [N, 7]

    K   = len(labels_matrix)
    k_c = (labels_matrix > 0).sum(axis=0).clip(min=1)    # [7] multi-label
    w_c = (K - k_c) / k_c                                 # формула статьи

    return torch.tensor(w_c, dtype=torch.float).to(device)


def compute_ah_weights(ah_train_dataset, device):
    """Веса классов AH (2 класса): len / (num_classes * count_c)."""
    labels  = [ah_train_dataset[i]["ah_label"].item() for i in range(len(ah_train_dataset))]
    counts  = np.bincount(np.array(labels).astype(int)).clip(min=1)
    weights = len(labels) / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float).to(device)


def build_criteria(emo_train_dataset, ah_train_dataset, cfg, device):
    """
    Собирает criterion_emo, criterion_ah, criterion_ah_ssl.

    cfg.flag_emo_weight: bool — включить веса классов эмоций (формула статьи).
    cfg.flag_ah_weight:  bool — включить веса классов AH.
    """
    # ── EMO ──────────────────────────────────────────────────────────────────
    if getattr(cfg, "flag_emo_weight", True):
        emo_weights = compute_emo_weights(emo_train_dataset, device)
        print("Веса классов EMO (w_c = (K-k_c)/k_c):",
              np.round(emo_weights.cpu().numpy(), 3))
        criterion_emo = nn.CrossEntropyLoss(weight=emo_weights)
    else:
        criterion_emo = nn.CrossEntropyLoss()

    # ── AH ───────────────────────────────────────────────────────────────────
    if getattr(cfg, "flag_ah_weight", True):
        ah_weights = compute_ah_weights(ah_train_dataset, device)
        print("Веса классов AH:", np.round(ah_weights.cpu().numpy(), 3))
        criterion_ah = nn.CrossEntropyLoss(weight=ah_weights)
    else:
        criterion_ah = nn.CrossEntropyLoss()

    criterion_ah_ssl = nn.CrossEntropyLoss(reduction="none")  # для per-sample маски SSL

    return criterion_emo, criterion_ah, criterion_ah_ssl
