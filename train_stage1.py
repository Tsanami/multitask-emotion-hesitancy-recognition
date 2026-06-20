"""
Точка входа для стадии 1 (обучение unimodal энкодеров).

    python train_stage1.py --task emotion --seed 42
    python train_stage1.py --task ah --seed 42
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn

from configs.configs import Stage1Config
from data.loaders import get_stage1_loaders
from models import EmotionTransformer, AHTransformer
from training.stage1_epochs import (
    train_emo_epoch, eval_emo_epoch,
    train_ah_epoch, eval_ah_epoch,
)
from utils.seed import set_seed


def run(cfg, task, seed):
    set_seed(seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    train_loader, val_loader, test_loader, train_ds = get_stage1_loaders(cfg, task)

    # ── Модель и функции эпохи под задачу ─────────────────────────────────────
    if task == "emotion":
        model = EmotionTransformer(
            input_dim_emotion=384, hidden_dim=256, out_features=256,
            num_transformer_heads=4, tr_layer_number=3, dropout=0.0,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        train_fn, eval_fn = train_emo_epoch, eval_emo_epoch
        out_path = cfg.emo_output_path

    else:  # ah
        model = AHTransformer(
            input_dim_ah=384, hidden_dim=512, out_features=128,
            num_transformer_heads=8, tr_layer_number=1, dropout=0.2,
        ).to(device)
        # Веса классов для несбалансированного BAH
        labels = [train_ds[i]["ah_label"].item() for i in range(len(train_ds))]
        counts = np.bincount(np.array(labels).astype(int)).clip(min=1)
        weights = torch.tensor(len(labels) / (2 * counts), dtype=torch.float).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        train_fn, eval_fn = train_ah_epoch, eval_ah_epoch
        out_path = cfg.ah_output_path

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    best_mf1, patience = -1.0, 0
    for epoch in range(cfg.epochs):
        print(f"\n{'='*12} [{task} seed={seed}] Epoch {epoch+1}/{cfg.epochs} {'='*12}")
        tr = train_fn(model, optimizer, train_loader, criterion, device)
        ev = eval_fn(model, val_loader, criterion, device)
        scheduler.step(ev["mf1"])

        print(f"TRAIN | Loss: {tr['loss']:.4f} | mF1: {tr['mf1']:.4f}")
        print(f"VAL   | Loss: {ev['loss']:.4f} | mF1: {ev['mf1']:.4f} | UAR: {ev['uar']:.4f}")

        if ev["mf1"] > best_mf1:
            best_mf1, patience = ev["mf1"], 0
            torch.save(model.state_dict(), out_path)
            print(f"    ✓ Saved (val mF1: {best_mf1:.4f})")
        else:
            patience += 1
            if patience >= cfg.max_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    model.load_state_dict(torch.load(out_path, map_location=device))
    test_log = eval_fn(model, test_loader, criterion, device)
    print(f"\nTEST [{task}] | mF1: {test_log['mf1']:.4f} | UAR: {test_log['uar']:.4f}")
    return test_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["emotion", "ah"], required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mosei_path", type=str, default=None)
    parser.add_argument("--bah_path",   type=str, default=None)
    args = parser.parse_args()

    cfg = Stage1Config()
    if args.mosei_path: cfg.mosei_path = args.mosei_path
    if args.bah_path:   cfg.bah_path   = args.bah_path

    run(cfg, args.task, args.seed)
