import numpy as np
import torch

# Метрики из measures.py статьи (per-column macro avg, затем mean)
from .measures import mf1, uar, acc_func, ccc  # noqa: F401

# Пороги из train.py статьи
_THR_NEUTRAL = 1.0 - 1.0 / 7.0   # ≈ 0.857
_THR_EMO     = 1.0 / 7.0          # ≈ 0.143


def transform_matrix(matrix: np.ndarray) -> np.ndarray:
    """Cell 11. Бинаризация softmax-вероятностей → multi-label предсказание."""
    mask_neutral = matrix[:, 0] >= _THR_NEUTRAL
    result       = np.zeros_like(matrix[:, 1:])
    transformed  = (matrix[:, 1:] >= _THR_EMO).astype(int)
    result[~mask_neutral] = transformed[~mask_neutral]
    return result


def predict_emotions(logits: torch.Tensor) -> list:
    """process_predictions из train.py статьи. logits [B,7] → list of [6]."""
    probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
    return transform_matrix(probs).tolist()
