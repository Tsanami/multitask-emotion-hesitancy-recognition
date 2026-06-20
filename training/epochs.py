import torch
from tqdm import tqdm
from .metrics import predict_emotions, mf1, uar


def train_one_epoch(model, optimizer, dataloader,
                    criterion_emo, criterion_ah, criterion_ah_ssl,
                    device, cfg, current_epoch=0):
    model.train()
    model.emo_model.eval()
    model.ah_model.eval()

    use_ssl = cfg.use_ssl and (current_epoch >= cfg.ssl_warmup_epochs)

    running_loss = loss_emo_sup = loss_ah_sup = loss_emo_ssl = loss_ah_ssl = 0.0
    # ── SSL статистика (pseudo-label coverage) ────────────────────────────────
    n_ssl_emo_conf = 0   # сколько unlabeled EMO прошло порог уверенности
    n_ssl_emo_total = 0  # всего unlabeled EMO сэмплов
    n_ssl_ah_conf  = 0
    n_ssl_ah_total = 0
    import numpy as _np
    pseudo_emo_hist = _np.zeros(7, dtype=int)  # распределение псевдо-меток по классам
    emo_preds, emo_trues = [], []
    ah_preds,  ah_trues  = [], []

    for batch in tqdm(dataloader, desc=f"Train epoch {current_epoch+1}"):
        if batch is None:
            continue

        text_emb   = batch["text_embedding"].to(device)
        emo_labels = batch["emo_labels"].to(device)   # [B, 7] float | NaN
        ah_labels  = batch["ah_labels"].to(device)    # [B] float | NaN

        valid_emo = ~torch.isnan(emo_labels[:, 0])
        valid_ah  = ~torch.isnan(ah_labels)
        if not valid_emo.any() and not valid_ah.any():
            continue

        optimizer.zero_grad()
        out        = model(emotion_input=text_emb, ah_input=text_emb)
        emo_logits = out["emotion_logits"]
        ah_logits  = out["ah_logits"]

        loss = torch.tensor(0.0, device=device, requires_grad=False)

        # ── Supervised EMO — CE с dominant class ──────────────────────────
        if valid_emo.any():
            dominant = torch.argmax(emo_labels[valid_emo], dim=1).long()
            l = criterion_emo(emo_logits[valid_emo], dominant)
            loss = loss + l; loss_emo_sup += l.item()

            emo_preds.extend(predict_emotions(emo_logits[valid_emo]))
            emo_trues.extend(
                (emo_labels[valid_emo][:, 1:] > 0).long().cpu().numpy()
            )

        # ── Supervised AH — CE ────────────────────────────────────────────
        if valid_ah.any():
            l = criterion_ah(ah_logits[valid_ah], ah_labels[valid_ah].long())
            loss = loss + l; loss_ah_sup += l.item()

            _, pred_ah = torch.max(ah_logits[valid_ah], dim=1)
            ah_preds.extend(pred_ah.cpu().numpy())
            ah_trues.extend(ah_labels[valid_ah].long().cpu().numpy())

        # ── SSL EMO — softmax → argmax → CE ──────────────────────────────
        if use_ssl and (~valid_emo).any():
            pred_u    = emo_logits[~valid_emo]
            probs     = torch.softmax(pred_u, dim=1)
            conf, pseudo = torch.max(probs, dim=1)
            mask = conf > cfg.ssl_conf_thr_emo
            n_ssl_emo_total += pred_u.size(0)
            n_ssl_emo_conf  += int(mask.sum().item())
            if mask.any():
                # распределение псевдо-меток по классам (для анализа)
                for c in pseudo[mask].cpu().numpy():
                    pseudo_emo_hist[c] += 1
                l = criterion_emo(pred_u[mask], pseudo[mask])
                loss = loss + cfg.lambda_ssl * l
                loss_emo_ssl += l.item()

        # ── SSL AH — softmax → argmax → CE с per-sample маской ───────────
        if use_ssl and (~valid_ah).any():
            pred_u    = ah_logits[~valid_ah]
            probs     = torch.softmax(pred_u, dim=1)
            max_p, pseudo = torch.max(probs, dim=1)
            mask = max_p > cfg.ssl_conf_thr_ah
            n_ssl_ah_total += pred_u.size(0)
            n_ssl_ah_conf  += int(mask.sum().item())
            if mask.any():
                l_elem = criterion_ah_ssl(pred_u, pseudo)   # [U]
                l = (l_elem * mask.float()).sum() / mask.sum()
                loss = loss + cfg.lambda_ssl * l
                loss_ah_ssl += l.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

    n = len(dataloader)
    # Coverage = доля unlabeled сэмплов прошедших порог уверенности
    cov_emo = n_ssl_emo_conf / n_ssl_emo_total if n_ssl_emo_total > 0 else 0.0
    cov_ah  = n_ssl_ah_conf  / n_ssl_ah_total  if n_ssl_ah_total  > 0 else 0.0
    ssl_status = (
        f"emo_ssl: {loss_emo_ssl/n:.4f} (cov {cov_emo:.2f}, n={n_ssl_emo_conf}) | "
        f"ah_ssl: {loss_ah_ssl/n:.4f} (cov {cov_ah:.2f}, n={n_ssl_ah_conf})"
        if use_ssl else "SSL: warmup"
    )
    return {
        "loss":         running_loss  / n,
        "loss_emo_sup": loss_emo_sup  / n,
        "loss_ah_sup":  loss_ah_sup   / n,
        "loss_emo_ssl": loss_emo_ssl  / n,
        "loss_ah_ssl":  loss_ah_ssl   / n,
        "ssl_status":   ssl_status,
        # ── SSL статистика для НИРа ──
        "n_ssl_emo":      n_ssl_emo_conf,
        "n_ssl_emo_total": n_ssl_emo_total,
        "cov_ssl_emo":    cov_emo,
        "n_ssl_ah":       n_ssl_ah_conf,
        "n_ssl_ah_total": n_ssl_ah_total,
        "cov_ssl_ah":     cov_ah,
        "pseudo_emo_hist": pseudo_emo_hist.tolist(),
        "emo_mf1":      mf1(emo_trues, emo_preds),
        "ah_mf1":       mf1([[l] for l in ah_trues], [[p] for p in ah_preds]),
    }


@torch.no_grad()
def eval_one_epoch(model, dataloader, criterion_emo, criterion_ah, device):
    model.eval()

    running_loss = loss_emo_sup = loss_ah_sup = 0.0
    emo_preds, emo_trues = [], []
    ah_preds,  ah_trues  = [], []

    for batch in tqdm(dataloader, desc="Eval"):
        if batch is None:
            continue

        text_emb   = batch["text_embedding"].to(device)
        emo_labels = batch["emo_labels"].to(device)
        ah_labels  = batch["ah_labels"].to(device)

        valid_emo = ~torch.isnan(emo_labels[:, 0])
        valid_ah  = ~torch.isnan(ah_labels)
        if not valid_emo.any() and not valid_ah.any():
            continue

        out        = model(emotion_input=text_emb, ah_input=text_emb)
        emo_logits = out["emotion_logits"]
        ah_logits  = out["ah_logits"]

        if valid_emo.any():
            dominant = torch.argmax(emo_labels[valid_emo], dim=1).long()
            l = criterion_emo(emo_logits[valid_emo], dominant)
            loss_emo_sup += l.item(); running_loss += l.item()

            emo_preds.extend(predict_emotions(emo_logits[valid_emo]))
            emo_trues.extend(
                (emo_labels[valid_emo][:, 1:] > 0).long().cpu().numpy()
            )

        if valid_ah.any():
            l = criterion_ah(ah_logits[valid_ah], ah_labels[valid_ah].long())
            loss_ah_sup += l.item(); running_loss += l.item()

            _, pred_ah = torch.max(ah_logits[valid_ah], dim=1)
            ah_preds.extend(pred_ah.cpu().numpy())
            ah_trues.extend(ah_labels[valid_ah].long().cpu().numpy())

    n       = len(dataloader)
    emo_mf1 = mf1(emo_trues, emo_preds)
    ah_mf1  = mf1([[l] for l in ah_trues], [[p] for p in ah_preds])
    return {
        "loss":         running_loss / n,
        "loss_emo_sup": loss_emo_sup / n,
        "loss_ah_sup":  loss_ah_sup  / n,
        "emo_mf1":      emo_mf1,
        "emo_uar":      uar(emo_trues, emo_preds),
        "ah_mf1":       ah_mf1,
        "ah_uar":       uar([[l] for l in ah_trues], [[p] for p in ah_preds]),
        "overall_f1":   (emo_mf1 + ah_mf1) / 2,
    }
