import torch.nn as nn
from .blocks import TransformerModelWithAttention


class EmotionTransformer(nn.Module):
    """Cell 8. Стадия 1 — unimodal emotion encoder."""
    def __init__(
        self,
        input_dim_emotion=384,
        hidden_dim=256,
        out_features=256,
        num_transformer_heads=4,
        tr_layer_number=3,
        dropout=0.0,
        num_emotions=7,
        positional_encoding=True,
        **kwargs,                  # поглощает лишние аргументы из старых вызовов
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.emo_proj = nn.Sequential(
            nn.Linear(input_dim_emotion, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.emotion_encoder = TransformerModelWithAttention(
            hidden_dim=hidden_dim,
            num_heads=num_transformer_heads,
            num_layers=tr_layer_number,
            dropout=dropout,
        )
        self.emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions),
        )

    def forward(self, emotion_input=None, return_features=False, **kwargs):
        emo = self.emo_proj(emotion_input)
        emo = self.emotion_encoder(emo)
        out_emo = self.emotion_fc_out(emo.mean(dim=1))

        if return_features:
            return {"emotion_logits": out_emo, "last_encoder_features": emo}
        return {"emotion_logits": out_emo}
