from torch.utils.data import DataLoader, ConcatDataset
from .datasets import DatasetEmotionAHFusion, custom_collate_fn


def get_stage2_loaders(cfg):
    """Возвращает train/val/test загрузчики для стадии 2 (fusion)."""

    def _make(part, shuffle):
        emo = DatasetEmotionAHFusion(
            dataset="CMU-MOSEI", part=part,
            path=cfg.mosei_path,
            path_to_emb=getattr(cfg, f"mosei_{part}_emb", None),
            model=cfg.encoder_model,
        )
        bah_part = part if part != "validation" else "val"
        ah = DatasetEmotionAHFusion(
            dataset="BAH", part=bah_part,
            path=cfg.bah_path,
            path_to_emb=getattr(cfg, f"bah_{bah_part}_emb", None),
            model=cfg.encoder_model,
        )
        ds = ConcatDataset([emo, ah])
        return DataLoader(ds, batch_size=cfg.batch_size,
                          shuffle=shuffle, collate_fn=custom_collate_fn), emo, ah

    train_loader, emo_train, ah_train = _make("train",      shuffle=True)
    val_loader,   _,         _        = _make("validation",  shuffle=False)
    test_loader,  _,         _        = _make("test",        shuffle=False)

    return train_loader, val_loader, test_loader, emo_train, ah_train


def get_stage1_loaders(cfg, task):
    """
    Загрузчики для стадии 1 (single-task).
    task: 'emotion' → MOSEI | 'ah' → BAH
    Возвращает train_loader, val_loader, test_loader, train_dataset
    """
    if task == "emotion":
        dataset_name = "CMU-MOSEI"
        parts = {"train": "train", "val": "validation", "test": "test"}
        path = cfg.mosei_path
        emb_prefix = "mosei"
    elif task == "ah":
        dataset_name = "BAH"
        parts = {"train": "train", "val": "val", "test": "test"}
        path = cfg.bah_path
        emb_prefix = "bah"
    else:
        raise ValueError("task must be 'emotion' or 'ah'")

    def _make(split, shuffle):
        part = parts[split]
        emb_attr = f"{emb_prefix}_{part}_emb"
        ds = DatasetEmotionAHFusion(
            dataset=dataset_name, part=part, path=path,
            path_to_emb=getattr(cfg, emb_attr, None),
            model=cfg.encoder_model,
        )
        return DataLoader(ds, batch_size=cfg.batch_size,
                          shuffle=shuffle, collate_fn=custom_collate_fn), ds

    train_loader, train_ds = _make("train", True)
    val_loader,   _        = _make("val",   False)
    test_loader,  _        = _make("test",  False)
    return train_loader, val_loader, test_loader, train_ds
