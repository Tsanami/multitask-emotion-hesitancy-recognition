import torch.nn as nn
from .blocks import CustomMambaBlock


class EmotionMamba(nn.Module):
    """Альтернативный backbone на Mamba-блоках."""
    def __init__(
        self,
        input_dim_emotion=384,
        hidden_dim=256,
        out_features=256,
        mamba_layer_number=4,
        mamba_d_model=256,
        dropout=0.2,
        num_emotions=7,
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.emo_proj = nn.Sequential(
            nn.Linear(input_dim_emotion, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.emotion_encoder = nn.ModuleList([
            CustomMambaBlock(hidden_dim, mamba_d_model, dropout=dropout)
            for _ in range(mamba_layer_number)
        ])
        self.emotion_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_emotions),
        )

    def forward(self, emotion_input=None, return_features=False, **kwargs):
        emo = self.emo_proj(emotion_input)
        for layer in self.emotion_encoder:
            emo = layer(emo)
        out_emo = self.emotion_fc_out(emo.mean(dim=1))

        if return_features:
            return {"emotion_logits": out_emo, "last_encoder_features": emo}
        return {"emotion_logits": out_emo}
