import torch.nn as nn
from .blocks import TransformerEncoderLayer


class AHTransformer(nn.Module):
    """Стадия 1 — unimodal AH encoder."""
    def __init__(
        self,
        input_dim_ah=384,
        hidden_dim=512,
        out_features=128,
        num_transformer_heads=8,
        tr_layer_number=1,
        dropout=0.2,
        num_ah_classes=2,
        positional_encoding=True,
        attn_type="default",
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.attn_type = attn_type

        self.per_proj = nn.Sequential(
            nn.Linear(input_dim_ah, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        # default → штатный стек TransformerEncoderLayer (поведение не меняется);
        # иначе — сменный механизм внимания из models/attention.py
        if attn_type in (None, "default"):
            self.ah_encoder = nn.ModuleList([
                TransformerEncoderLayer(
                    input_dim=hidden_dim,
                    num_heads=num_transformer_heads,
                    dropout=dropout,
                    positional_encoding=positional_encoding,
                )
                for _ in range(tr_layer_number)
            ])
        else:
            from .attention import CustomAttentionEncoder
            self.ah_encoder = CustomAttentionEncoder(
                hidden_dim, num_transformer_heads, tr_layer_number,
                attn_type=attn_type, dropout=dropout,
            )
        self.ah_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_ah_classes),
        )

    def forward(self, ah_input=None, return_features=False, **kwargs):
        per = self.per_proj(ah_input)
        if self.attn_type in (None, "default"):
            for layer in self.ah_encoder:
                per = per + layer(per, per, per)
        else:
            per = self.ah_encoder(per)   # CustomAttentionEncoder: residual внутри
        out_per = self.ah_fc_out(per.mean(dim=1))

        if return_features:
            return {"ah_scores": out_per, "last_encoder_features": per}
        return {"ah_scores": out_per}
