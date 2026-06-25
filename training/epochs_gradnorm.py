import numpy as np
import torch
from tqdm import tqdm
from .metrics import predict_emotions, mf1, uar


def train_one_epoch_gradnorm(model, optimizer, dataloader, gradnorm_loss,
                             device, cfg, current_epoch=0):
    """
    Трейн-эпоха с GradNorm-балансировкой (gradnorm_loss = GradNormMultiTaskLoss).
    Веса лоссов обновляются динамически внутри gradnorm_loss.forward().
    """
    model.train()
    if not getattr(cfg, "unfreeze_encoders", False):
        model.emo_model.eval()
        model.ah_model.eval()

    use_ssl = cfg.use_ssl and (current_epoch >= cfg.ssl_warmup_epochs)
    shared_params = gradnorm_loss.shared_params_from_model(model)

    running_loss = 0.0
    comp_totals  = {"emo_sup": 0.0, "ah_sup": 0.0, "emo_ssl": 0.0, "ah_ssl": 0.0}
    last_weights = {}
    emo_preds, emo_trues = [], []
    ah_preds,  ah_trues  = [], []

    # ── SSL статистика (pseudo-label coverage) — паритет с epochs.py ───────────
    # GradNorm считает псевдо-метки внутри loss.forward() и наружу их не отдаёт.
    # Пересчитываем здесь тем же argmax по тем же логитам — результат идентичен
    # тому, что использует GradNormMultiTaskLoss._collect (детерминированно).
    n_ssl_emo_conf = n_ssl_emo_total = 0
    n_ssl_ah_conf  = n_ssl_ah_total  = 0
    pseudo_emo_hist = np.zeros(7, dtype=int)

    for batch in tqdm(dataloader, desc=f"Train GradNorm epoch {current_epoch+1}"):
        if batch is None:
            continue

        text_emb   = batch["text_embedding"].to(device)
        emo_labels = batch["emo_labels"].to(device)
        ah_labels  = batch["ah_labels"].to(device)

        valid_emo = ~torch.isnan(emo_labels[:, 0])
        valid_ah  = ~torch.isnan(ah_labels)
        if not valid_emo.any() and not valid_ah.any():
            continue

        optimizer.zero_grad()
        out        = model(emotion_input=text_emb, ah_input=text_emb)
        emo_logits = out["emotion_logits"]
        ah_logits  = out["ah_logits"]

        total, details = gradnorm_loss(
            emo_logits, ah_logits, emo_labels, ah_labels,
            valid_emo, valid_ah, shared_params, use_ssl=use_ssl,
        )

        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += total.item()
        for k, v in details.get("components", {}).items():
            comp_totals[k] += v
        last_weights = {**details.get("weights_sup", {}), **details.get("weights_ssl", {})}

        # Метрики
        if valid_emo.any():
            emo_preds.extend(predict_emotions(emo_logits[valid_emo]))
            emo_trues.extend((emo_labels[valid_emo][:, 1:] > 0).long().cpu().numpy())
        if valid_ah.any():
            _, pred_ah = torch.max(ah_logits[valid_ah], dim=1)
            ah_preds.extend(pred_ah.cpu().numpy())
            ah_trues.extend(ah_labels[valid_ah].long().cpu().numpy())

        # ── SSL coverage + распределение псевдо-меток (для НИРа) ──────────────
        # Реплика логики epochs.py; считаем под no_grad — на лосс не влияет.
        if use_ssl:
            with torch.no_grad():
                if (~valid_emo).any():
                    pred_u = emo_logits[~valid_emo]
                    conf, pseudo = torch.max(torch.softmax(pred_u, dim=1), dim=1)
                    mask = conf > cfg.ssl_conf_thr_emo
                    n_ssl_emo_total += pred_u.size(0)
                    n_ssl_emo_conf  += int(mask.sum().item())
                    for c in pseudo[mask].cpu().numpy():
                        pseudo_emo_hist[c] += 1
                if (~valid_ah).any():
                    pred_u = ah_logits[~valid_ah]
                    conf, pseudo = torch.max(torch.softmax(pred_u, dim=1), dim=1)
                    mask = conf > cfg.ssl_conf_thr_ah
                    n_ssl_ah_total += pred_u.size(0)
                    n_ssl_ah_conf  += int(mask.sum().item())

    n = len(dataloader)
    # Лог весов GradNorm — это важно для анализа в НИРе
    weights_str = " | ".join(f"{k}={v:.3f}" for k, v in last_weights.items())

    cov_emo = n_ssl_emo_conf / n_ssl_emo_total if n_ssl_emo_total > 0 else 0.0
    cov_ah  = n_ssl_ah_conf  / n_ssl_ah_total  if n_ssl_ah_total  > 0 else 0.0

    return {
        "loss":         running_loss      / n,
        "loss_emo_sup": comp_totals["emo_sup"] / n,
        "loss_ah_sup":  comp_totals["ah_sup"]  / n,
        "loss_emo_ssl": comp_totals["emo_ssl"] / n,
        "loss_ah_ssl":  comp_totals["ah_ssl"]  / n,
        "gradnorm_weights": last_weights,
        "ssl_status":   f"weights: {weights_str}",
        # ── SSL статистика для НИРа (паритет с epochs.py) ──
        "n_ssl_emo":       n_ssl_emo_conf,
        "n_ssl_emo_total": n_ssl_emo_total,
        "cov_ssl_emo":     cov_emo,
        "n_ssl_ah":        n_ssl_ah_conf,
        "n_ssl_ah_total":  n_ssl_ah_total,
        "cov_ssl_ah":      cov_ah,
        "pseudo_emo_hist": pseudo_emo_hist.tolist(),
        "emo_mf1":      mf1(emo_trues, emo_preds) if emo_trues else 0.0,
        "ah_mf1":       mf1([[t] for t in ah_trues], [[p] for p in ah_preds]) if ah_trues else 0.0,
    }
