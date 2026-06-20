"""
GradNorm "wallet" — динамическая балансировка лоссов из SSL-MEPR (utils/losses.py).

Адаптация MultiTaskLossWithNaN_v2 под пару задач ER (emotion) + AHR (ah).
Отличия от классического GradNorm:
  - раздельные "кошельки" весов для supervised и SSL компонентов;
  - веса нормируются так, чтобы их сумма равнялась бюджету (budget_sup=2.0, budget_ssl=2*lambda);
  - в итоговом лоссе веса detached — градиент не течёт из весов в основную модель;
  - SSL для эмоций: softmax → argmax → CE по уверенным псевдо-меткам.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradNormMultiTaskLoss(nn.Module):
    SUP_KEYS = ["emo_sup", "ah_sup"]
    SSL_KEYS = ["emo_ssl", "ah_ssl"]

    def __init__(
        self,
        criterion_emo,                 # nn.CrossEntropyLoss(weight=...) — с весами классов
        criterion_ah,                  # nn.CrossEntropyLoss(weight=...)
        weight_emotion=1.0,
        weight_ah=1.0,
        alpha_sup=1.0, w_lr_sup=0.005,
        alpha_ssl=1.5, w_lr_ssl=0.01,
        lambda_ssl=0.2, w_floor=1e-3,
        ssl_conf_thr_emo=0.6,
        ssl_conf_thr_ah=0.6,
    ):
        super().__init__()
        self.emotion_loss = criterion_emo
        self.ah_loss      = criterion_ah

        self.alpha_sup = alpha_sup; self.w_lr_sup = w_lr_sup
        self.alpha_ssl = alpha_ssl; self.w_lr_ssl = w_lr_ssl
        self.lambda_ssl = lambda_ssl; self.w_floor = w_floor
        self.ssl_conf_thr_emo = ssl_conf_thr_emo
        self.ssl_conf_thr_ah  = ssl_conf_thr_ah

        self.budget_sup = 2.0
        self.budget_ssl = 2.0 * lambda_ssl

        self.weight_sup = nn.ParameterDict({
            "emo_sup": nn.Parameter(torch.tensor(float(weight_emotion))),
            "ah_sup":  nn.Parameter(torch.tensor(float(weight_ah))),
        })
        self.weight_ssl = nn.ParameterDict({
            "emo_ssl": nn.Parameter(torch.tensor(float(lambda_ssl))),
            "ah_ssl":  nn.Parameter(torch.tensor(float(lambda_ssl))),
        })
        self._normalize(self.weight_sup, self.SUP_KEYS, self.budget_sup)
        self._normalize(self.weight_ssl, self.SSL_KEYS, self.budget_ssl)

        self.init_sup = {}
        self.init_ssl = {}

    # ── helpers ──────────────────────────────────────────────────────────────
    @staticmethod
    def shared_params_from_model(model):
        """Shared = всё, что не относится к головам emotion/ah."""
        return [p for name, p in model.named_parameters()
                if ("emotion" not in name and "ah" not in name) and p.requires_grad]

    @staticmethod
    def _grad_norm(grads):
        g = [t.detach().flatten() for t in grads if t is not None]
        if not g:
            return None
        return torch.norm(torch.cat(g), p=2)

    def _normalize(self, pdict, keys, target_sum):
        with torch.no_grad():
            s = sum(pdict[k] for k in keys).detach().clamp_min(1e-8)
            for k in keys:
                pdict[k].data = target_sum * (pdict[k].data / s)

    # ── собираем компоненты лосса ────────────────────────────────────────────
    def _collect(self, emo_logits, ah_logits, emo_labels, ah_labels,
                 valid_emo, valid_ah, use_ssl):
        comps = {}

        # Supervised EMO — CE с dominant class
        if valid_emo.any():
            dominant = torch.argmax(emo_labels[valid_emo], dim=1).long()
            comps["emo_sup"] = self.emotion_loss(emo_logits[valid_emo], dominant)

        # Supervised AH — CE
        if valid_ah.any():
            comps["ah_sup"] = self.ah_loss(ah_logits[valid_ah], ah_labels[valid_ah].long())

        if not use_ssl:
            return comps

        # SSL EMO — softmax → argmax → CE по уверенным
        if (~valid_emo).any():
            pred_u = emo_logits[~valid_emo]
            probs  = torch.softmax(pred_u, dim=1)
            conf, pseudo = torch.max(probs, dim=1)
            c_mask = conf > self.ssl_conf_thr_emo
            if c_mask.any():
                comps["emo_ssl"] = self.emotion_loss(pred_u[c_mask], pseudo[c_mask])

        # SSL AH — softmax → argmax → CE по уверенным
        if (~valid_ah).any():
            pred_u = ah_logits[~valid_ah]
            probs  = torch.softmax(pred_u, dim=1)
            conf, pseudo = torch.max(probs, dim=1)
            c_mask = conf > self.ssl_conf_thr_ah
            if c_mask.any():
                comps["ah_ssl"] = self.ah_loss(pred_u[c_mask], pseudo[c_mask])

        return comps

    # ── обновление кошелька весов ────────────────────────────────────────────
    def _update_wallet(self, comps, keys, init_dict, wdict, alpha, w_lr, budget, shared_params):
        for k in keys:
            Li = comps.get(k)
            if Li is not None and k not in init_dict and torch.isfinite(Li).all():
                init_dict[k] = Li.detach().clamp_min(1e-8)

        active = [k for k in keys if k in comps and k in init_dict]
        if not active:
            return

        G_list, r_list, w_list = [], [], []
        for k in active:
            Li = comps[k]
            grads = torch.autograd.grad(Li, shared_params, retain_graph=True, allow_unused=True)
            gn = self._grad_norm(grads) or torch.tensor(0.0, device=Li.device)
            wk = wdict[k]
            G_list.append(wk * gn)
            r_list.append(Li.detach().clamp_min(1e-8) / init_dict[k])
            w_list.append(wk)

        G_stack = torch.stack(G_list); r_stack = torch.stack(r_list)
        G_avg, r_avg = G_stack.mean(), r_stack.mean()

        gn_loss = 0.0
        for i in range(len(active)):
            target = (G_avg * ((r_stack[i] / r_avg) ** alpha)).detach()
            gn_loss = gn_loss + torch.abs(G_stack[i] - target)

        grads_w = torch.autograd.grad(gn_loss, w_list, retain_graph=True, allow_unused=True)
        with torch.no_grad():
            for wk, gw in zip(w_list, grads_w):
                if gw is None:
                    continue
                wk.data -= w_lr * gw
                wk.data.clamp_(min=self.w_floor)
            self._normalize(wdict, active, budget)

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, emo_logits, ah_logits, emo_labels, ah_labels,
                valid_emo, valid_ah, shared_params, use_ssl=True):
        comps = self._collect(emo_logits, ah_logits, emo_labels, ah_labels,
                              valid_emo, valid_ah, use_ssl)

        if not comps:
            device = emo_logits.device
            return torch.tensor(0.0, requires_grad=True, device=device), {}

        self._update_wallet(comps, self.SUP_KEYS, self.init_sup, self.weight_sup,
                            self.alpha_sup, self.w_lr_sup, self.budget_sup, shared_params)
        if use_ssl:
            self._update_wallet(comps, self.SSL_KEYS, self.init_ssl, self.weight_ssl,
                                self.alpha_ssl, self.w_lr_ssl, self.budget_ssl, shared_params)

        total = 0.0
        for k in self.SUP_KEYS:
            if k in comps:
                total = total + self.weight_sup[k].detach() * comps[k]
        if use_ssl:
            for k in self.SSL_KEYS:
                if k in comps:
                    total = total + self.weight_ssl[k].detach() * comps[k]

        details = {
            "components":  {k: v.detach().item() for k, v in comps.items()},
            "weights_sup": {k: self.weight_sup[k].detach().item() for k in self.SUP_KEYS},
            "weights_ssl": {k: self.weight_ssl[k].detach().item() for k in self.SSL_KEYS},
        }
        return total, details
