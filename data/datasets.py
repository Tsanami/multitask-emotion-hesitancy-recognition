import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

from .preprocessing import get_cmu_mosei_data, get_bah_data

SUPPORTED_MODELS = {
    "bge-small":      "BAAI/bge-small-en-v1.5",
    "xlm-roberta":    "xlm-roberta-base",
    "jina":           "jinaai/jina-embeddings-v3",
}


class DatasetEmotionAHFusion(Dataset):
    """
    Cell 15. Единый датасет для MOSEI и BAH.

    emo_label:
      - MOSEI: [7] float (сырые интенсивности, Neutral первым)
      - BAH:   [7] NaN

    ah_label:
      - MOSEI: scalar NaN
      - BAH:   scalar float (0.0 или 1.0)
    """

    def __init__(
        self,
        dataset="CMU-MOSEI",
        part="train",
        path="data",
        path_to_emb=None,
        model="bge-small",
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if dataset == "CMU-MOSEI":
            texts, labels = get_cmu_mosei_data(path, part)
            self.task_name = "emotion"
        elif dataset == "BAH":
            texts, labels = get_bah_data(path, part)
            self.task_name = "uncertainty"
        else:
            raise ValueError("dataset must be 'CMU-MOSEI' or 'BAH'")

        self.x = texts
        self.y = labels[0]

        if path_to_emb is None:
            if model not in SUPPORTED_MODELS:
                raise ValueError(f"model must be one of {list(SUPPORTED_MODELS)}")

            model_name_hf = SUPPORTED_MODELS[model]
            trust = model == "jina"
            tokenizer = AutoTokenizer.from_pretrained(model_name_hf, trust_remote_code=trust)
            extractor = AutoModel.from_pretrained(model_name_hf, trust_remote_code=trust).to(device)
            extractor.eval()

            print(f"Extracting embeddings for {dataset} ({part}) with {model}...")
            self.text_embedding = []
            for t in tqdm(texts):
                enc = tokenizer(t, padding=True, truncation=True,
                                max_length=128, return_tensors="pt").to(device)
                with torch.no_grad():
                    feat = extractor(**enc).last_hidden_state.squeeze(0).cpu()
                self.text_embedding.append(feat)

            # Автосохранение
            cache_dir  = os.path.join(path, "embeddings_cache")
            os.makedirs(cache_dir, exist_ok=True)
            save_path  = os.path.join(cache_dir, f"{dataset}_{part}_{model}_embeddings.pkl")
            with open(save_path, "wb") as f:
                pickle.dump(self.text_embedding, f)
            print(f"Embeddings saved → {save_path}")

        else:
            print(f"Loading embeddings from {path_to_emb}...")
            with open(path_to_emb, "rb") as f:
                self.text_embedding = pickle.load(f)

        self.n_samples = len(texts)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        if self.task_name == "emotion":
            emo_label = torch.tensor(np.array(self.y[index]), dtype=torch.float32)  # [7] сырые
            ah_label  = torch.tensor(float("nan"), dtype=torch.float32)
        else:
            emo_label = torch.full((7,), float("nan"), dtype=torch.float32)
            ah_label  = torch.tensor(float(self.y[index]), dtype=torch.float32)

        return {
            "text_embedding": self.text_embedding[index],  # [T, D]
            "emo_label":      emo_label,                   # [7] | NaN
            "ah_label":       ah_label,                    # scalar | NaN
        }


def custom_collate_fn(batch):
    """Cell 15."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch  = [x for x in batch if x is not None]
    if not batch:
        return None

    text_tensor = pad_sequence(
        [b["text_embedding"] for b in batch], batch_first=True
    ).to(device)

    emo_labels = torch.stack([b["emo_label"] for b in batch])  # [B, 7]
    ah_labels  = torch.stack([b["ah_label"]  for b in batch])  # [B]

    return {
        "text_embedding": text_tensor,
        "emo_labels":     emo_labels,
        "ah_labels":      ah_labels,
    }
