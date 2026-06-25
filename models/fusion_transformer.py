import torch
import torch.nn as nn
from .blocks import TransformerEncoderLayer


class FusionTransformer(nn.Module):
    """Стадия 2 — cross-domain fusion."""
    def __init__(
        self,
        emo_model,
        ah_model,
        hidden_dim=256,
        out_features=512,
        num_transformer_heads=4,
        tr_layer_number=1,
        dropout=0.1,
        num_emotions=7,
        num_ah_classes=2,
        unfreeze_encoders=False,
    ):
        super().__init__()

        self.emo_model = emo_model
        self.ah_model  = ah_model
        # По умолчанию энкодеры заморожены (как в статье). При unfreeze_encoders=True
        # их веса обучаются — но через дискриминативный LR (см. train_stage2.py),
        # иначе они переобучаются за пару эпох.
        requires_grad = bool(unfreeze_encoders)
        for p in self.emo_model.parameters(): p.requires_grad = requires_grad
        for p in self.ah_model.parameters():  p.requires_grad = requires_grad

        self.emo_proj = nn.Sequential(
            nn.Linear(self.emo_model.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.ah_proj = nn.Sequential(
            nn.Linear(self.ah_model.hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        self.emotion_to_ah_attn = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_transformer_heads,
                                    dropout=dropout, positional_encoding=True)
            for _ in range(tr_layer_number)
        ])
        self.ah_to_emotion_attn = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_transformer_heads,
                                    dropout=dropout, positional_encoding=True)
            for _ in range(tr_layer_number)
        ])

        self.emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions),
        )
        self.ah_fc_out = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_features),
            nn.LayerNorm(out_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_ah_classes),
        )

    @staticmethod
    def _pad(t, length):
        pad_len = length - t.shape[1]
        if pad_len > 0:
            t = torch.cat(
                [t, torch.zeros(t.shape[0], pad_len, t.shape[2], device=t.device)],
                dim=1,
            )
        return t

    def forward(self, emotion_input=None, ah_input=None, return_features=False):
        emo_out = self.emo_model(emotion_input=emotion_input, return_features=True)
        ah_out  = self.ah_model(ah_input=ah_input,           return_features=True)

        emo_seq = self.emo_proj(emo_out["last_encoder_features"])
        ah_seq  = self.ah_proj(ah_out["last_encoder_features"])

        max_len = max(emo_seq.shape[1], ah_seq.shape[1])
        emo_seq = self._pad(emo_seq, max_len)
        ah_seq  = self._pad(ah_seq,  max_len)

        for layer in self.emotion_to_ah_attn:
            emo_seq = emo_seq + layer(emo_seq, ah_seq,  ah_seq)
        for layer in self.ah_to_emotion_attn:
            ah_seq  = ah_seq  + layer(ah_seq,  emo_seq, emo_seq)

        pooled = torch.cat([emo_seq, ah_seq], dim=-1).mean(dim=1)

        fusion_emo = self.emotion_fc_out(pooled)
        fusion_ah  = self.ah_fc_out(pooled)

        final_emo = (fusion_emo + emo_out["emotion_logits"]) / 2
        final_ah  = (fusion_ah  + ah_out["ah_scores"])       / 2

        if return_features:
            return {
                "emotion_logits":    final_emo,
                "ah_logits":         final_ah,
                "last_emo_features": emo_seq,
                "last_ah_features":  ah_seq,
            }
        return {"emotion_logits": final_emo, "ah_logits": final_ah}
