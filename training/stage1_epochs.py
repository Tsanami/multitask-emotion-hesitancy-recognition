import torch
from tqdm import tqdm
from .metrics import predict_emotions, mf1, uar


# ════════════════════════════════════════════════════════════════════════════
# СТАДИЯ 1 — ЭМОЦИИ (только MOSEI)
# ════════════════════════════════════════════════════════════════════════════

def train_emo_epoch(model, optimizer, dataloader, criterion, device):
    model.train()
    running_loss = 0.0
    preds, trues = [], []

    for batch in tqdm(dataloader, desc="Train EMO"):
        if batch is None:
            continue
        text_emb = batch["text_embedding"].to(device)
        labels   = batch["emo_labels"].to(device)       # [B, 7] float
        valid    = ~torch.isnan(labels[:, 0])
        if not valid.any():
            continue

        optimizer.zero_grad()
        logits = model(emotion_input=text_emb)["emotion_logits"]

        dominant = torch.argmax(labels[valid], dim=1).long()
        loss = criterion(logits[valid], dominant)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        preds.extend(predict_emotions(logits[valid]))
        trues.extend((labels[valid][:, 1:] > 0).long().cpu().numpy())

    return {"loss": running_loss / len(dataloader),
            "mf1": mf1(trues, preds) if trues else 0.0}


@torch.no_grad()
def eval_emo_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []

    for batch in tqdm(dataloader, desc="Eval EMO"):
        if batch is None:
            continue
        text_emb = batch["text_embedding"].to(device)
        labels   = batch["emo_labels"].to(device)
        valid    = ~torch.isnan(labels[:, 0])
        if not valid.any():
            continue

        logits = model(emotion_input=text_emb)["emotion_logits"]
        dominant = torch.argmax(labels[valid], dim=1).long()
        running_loss += criterion(logits[valid], dominant).item()

        preds.extend(predict_emotions(logits[valid]))
        trues.extend((labels[valid][:, 1:] > 0).long().cpu().numpy())

    return {"loss": running_loss / len(dataloader),
            "mf1": mf1(trues, preds) if trues else 0.0,
            "uar": uar(trues, preds) if trues else 0.0}


# ════════════════════════════════════════════════════════════════════════════
# СТАДИЯ 1 — AH (только BAH)
# ════════════════════════════════════════════════════════════════════════════

def train_ah_epoch(model, optimizer, dataloader, criterion, device):
    model.train()
    running_loss = 0.0
    preds, trues = [], []

    for batch in tqdm(dataloader, desc="Train AH"):
        if batch is None:
            continue
        text_emb = batch["text_embedding"].to(device)
        labels   = batch["ah_labels"].to(device)
        valid    = ~torch.isnan(labels)
        if not valid.any():
            continue

        optimizer.zero_grad()
        logits = model(ah_input=text_emb)["ah_scores"]
        loss = criterion(logits[valid], labels[valid].long())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        _, pred = torch.max(logits[valid], dim=1)
        preds.extend(pred.cpu().numpy())
        trues.extend(labels[valid].long().cpu().numpy())

    return {"loss": running_loss / len(dataloader),
            "mf1": mf1([[t] for t in trues], [[p] for p in preds]) if trues else 0.0}


@torch.no_grad()
def eval_ah_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, trues = [], []

    for batch in tqdm(dataloader, desc="Eval AH"):
        if batch is None:
            continue
        text_emb = batch["text_embedding"].to(device)
        labels   = batch["ah_labels"].to(device)
        valid    = ~torch.isnan(labels)
        if not valid.any():
            continue

        logits = model(ah_input=text_emb)["ah_scores"]
        running_loss += criterion(logits[valid], labels[valid].long()).item()

        _, pred = torch.max(logits[valid], dim=1)
        preds.extend(pred.cpu().numpy())
        trues.extend(labels[valid].long().cpu().numpy())

    return {"loss": running_loss / len(dataloader),
            "mf1": mf1([[t] for t in trues], [[p] for p in preds]) if trues else 0.0,
            "uar": uar([[t] for t in trues], [[p] for p in preds]) if trues else 0.0}
