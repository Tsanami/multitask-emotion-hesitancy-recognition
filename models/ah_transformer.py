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
        **kwargs,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.ah_proj = nn.Sequential(
            nn.Linear(input_dim_ah, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        self.ah_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                input_dim=hidden_dim,
                num_heads=num_transformer_heads,
                dropout=dropout,
                positional_encoding=positional_encoding,
            )
            for _ in range(tr_layer_number)
        ])
        self.ah_fc_out = nn.Sequential(
            nn.Linear(hidden_dim, out_features),
            nn.LayerNorm(out_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(out_features, num_ah_classes),
        )

    def forward(self, ah_input=None, return_features=False, **kwargs):
        ah = self.ah_proj(ah_input)
        for layer in self.ah_encoder:
            ah = ah + layer(ah, ah, ah)
        out_ah = self.ah_fc_out(ah.mean(dim=1))

        if return_features:
            return {"ah_scores": out_ah, "last_encoder_features": ah}
        return {"ah_scores": out_ah}
