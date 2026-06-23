"""
Точка входа для стадии 2 (Fusion).

Использование:
    python train_stage2.py              # без SSL
    python train_stage2.py --ssl        # с SSL
    python train_stage2.py --ssl --ssl_conf_thr_emo 0.8
"""
import argparse
import os
import torch
import torch.nn as nn

from configs.configs import Stage2NoSSLConfig, Stage2SSLConfig
from data.loaders import get_stage2_loaders
from models import EmotionTransformer, AHTransformer, FusionTransformer
from training.losses import build_criteria
from training.epochs import train_one_epoch, eval_one_epoch
from utils.history import History
from utils.seed import set_seed
from utils.checkpointing import save_checkpoint_with_config
from utils.tracking import track_run


def run(cfg, seed=42):
    set_seed(seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Данные ────────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, emo_train, ah_train = get_stage2_loaders(cfg)

    # ── Модели стадии 1 ───────────────────────────────────────────────────────
    emo_model = EmotionTransformer(
        input_dim_emotion=384, hidden_dim=256, out_features=256,
        num_transformer_heads=4, tr_layer_number=3, dropout=0.0
    ).to(device)
    ah_model = AHTransformer(
        input_dim_ah=384, hidden_dim=512, out_features=128,
        num_transformer_heads=8, tr_layer_number=1, dropout=0.2
    ).to(device)

    emo_model.load_state_dict(torch.load(cfg.emo_model_path, map_location=device))
    ah_model.load_state_dict(torch.load(cfg.ah_model_path,  map_location=device))
    emo_model.eval(); ah_model.eval()

    # ── Fusion ────────────────────────────────────────────────────────────────
    model = FusionTransformer(
        emo_model=emo_model, ah_model=ah_model,
        hidden_dim=cfg.hidden_dim, out_features=cfg.out_features,
        num_transformer_heads=cfg.num_transformer_heads,
        tr_layer_number=cfg.tr_layer_number, dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    criterion_emo, criterion_ah, criterion_ah_ssl = build_criteria(
        emo_train, ah_train, cfg, device
    )

    # GradNorm-вариант: динамическая балансировка лоссов
    use_gradnorm = getattr(cfg, "use_gradnorm", False)
    if use_gradnorm:
        from training.gradnorm import GradNormMultiTaskLoss
        from training.epochs_gradnorm import train_one_epoch_gradnorm
        gradnorm_loss = GradNormMultiTaskLoss(
            criterion_emo, criterion_ah,
            alpha_sup=cfg.alpha_sup, w_lr_sup=cfg.w_lr_sup,
            alpha_ssl=cfg.alpha_ssl, w_lr_ssl=cfg.w_lr_ssl,
            lambda_ssl=cfg.lambda_ssl, w_floor=cfg.w_floor,
            ssl_conf_thr_emo=cfg.ssl_conf_thr_emo,
            ssl_conf_thr_ah=cfg.ssl_conf_thr_ah,
        ).to(device)

    history = History(cfg.history_path)
    best_f1 = -1.0
    patience_counter = 0

    variant   = "gradnorm" if use_gradnorm else ("ssl" if getattr(cfg, "use_ssl", False) else "no_ssl")
    run_name  = getattr(cfg, "run_name", "") or f"stage2_{variant}_seed{seed}"

    with track_run(run_name, cfg, seed, enabled=getattr(cfg, "use_mlflow", True)) as tracker:
        # ── Цикл обучения ──────────────────────────────────────────────────────
        for epoch in range(cfg.epochs):
            print(f"\n{'='*15} Epoch {epoch+1}/{cfg.epochs} {'='*15}")

            if use_gradnorm:
                train_log = train_one_epoch_gradnorm(
                    model, optimizer, train_loader, gradnorm_loss,
                    device, cfg, current_epoch=epoch,
                )
            else:
                train_log = train_one_epoch(
                    model, optimizer, train_loader,
                    criterion_emo, criterion_ah, criterion_ah_ssl,
                    device, cfg, current_epoch=epoch,
                )
            val_log = eval_one_epoch(
                model, val_loader, criterion_emo, criterion_ah, device
            )

            scheduler.step(val_log["overall_f1"])
            history.update(train_log, val_log, optimizer)
            history.save()

            tracker.log_metrics(
                {f"train_{k}": v for k, v in train_log.items()}, step=epoch)
            tracker.log_metrics(
                {f"val_{k}": v for k, v in val_log.items()}, step=epoch)
            tracker.log_metrics({"lr": optimizer.param_groups[0]["lr"]}, step=epoch)

            print(f"TRAIN | {train_log['ssl_status']}"
                  f" | Emo mF1: {train_log['emo_mf1']:.4f} | AH mF1: {train_log['ah_mf1']:.4f}")
            print(f"VAL   | Emo mF1: {val_log['emo_mf1']:.4f} | Emo UAR: {val_log['emo_uar']:.4f}"
                  f" | AH mF1: {val_log['ah_mf1']:.4f} | AH UAR: {val_log['ah_uar']:.4f}"
                  f" | Overall: {val_log['overall_f1']:.4f}"
                  f" | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if val_log["overall_f1"] > best_f1:
                best_f1 = val_log["overall_f1"]
                patience_counter = 0
                save_checkpoint_with_config(model, cfg, cfg.output_path, seed=seed)
                print(f"    ✓ Saved (overall F1: {best_f1:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= cfg.max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # ── Тест ───────────────────────────────────────────────────────────────
        model.load_state_dict(torch.load(cfg.output_path, map_location=device))
        test_log = eval_one_epoch(model, test_loader, criterion_emo, criterion_ah, device)

        tracker.log_metrics({f"test_{k}": v for k, v in test_log.items()})
        tracker.log_metrics({"best_val_overall_f1": best_f1})
        tracker.log_artifact(cfg.output_path)
        tracker.log_artifact(cfg.output_path.replace(".pt", ".config.json"))
        tracker.log_artifact(cfg.history_path)

    print(f"\nTEST | Emo mF1: {test_log['emo_mf1']:.4f} | Emo UAR: {test_log['emo_uar']:.4f}")
    print(f"     | AH mF1:  {test_log['ah_mf1']:.4f}  | AH UAR:  {test_log['ah_uar']:.4f}")
    print(f"     | Overall F1: {test_log['overall_f1']:.4f}")
    return test_log


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ssl", action="store_true", help="Обучать с SSL")
    parser.add_argument("--gradnorm", action="store_true", help="Использовать GradNorm-балансировку")
    # Переопределение любого поля конфига из командной строки
    parser.add_argument("--ssl_conf_thr_emo",  type=float, default=None)
    parser.add_argument("--ssl_conf_thr_ah",   type=float, default=None)
    parser.add_argument("--lambda_ssl",        type=float, default=None)
    parser.add_argument("--output_path",       type=str,   default=None)
    parser.add_argument("--mosei_path",        type=str,   default=None)
    parser.add_argument("--bah_path",          type=str,   default=None)
    parser.add_argument("--seed",              type=int,   default=42)
    args = parser.parse_args()

    from configs.configs import Stage2GradNormConfig
    if args.gradnorm:
        cfg = Stage2GradNormConfig()
    elif args.ssl:
        cfg = Stage2SSLConfig()
    else:
        cfg = Stage2NoSSLConfig()

    # Применяем переопределения
    for key in ("ssl_conf_thr_emo", "ssl_conf_thr_ah", "lambda_ssl",
                "output_path", "mosei_path", "bah_path"):
        val = getattr(args, key)
        if val is not None:
            setattr(cfg, key, val)

    run(cfg, seed=args.seed)
