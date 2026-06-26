"""
Проверка сменных механизмов внимания без данных/чекпойнтов.

    python -m tests.check_attention

Проверяет: (1) все 4 механизма форвардят на формах проекта без NaN/Inf и
сохраняют форму [B,T,D]; (2) ELSA численно совпадает с softmax при общих весах
(блочный online-softmax эквивалентен полному); (3) EmotionTransformer/AHTransformer
строятся и работают для каждого attn_type, а default-путь не затронут.
"""
import torch

from models.attention import (
    StandardSelfAttention, ELSAAttention, make_attention, ATTENTION_TYPES,
)
from models import EmotionTransformer, AHTransformer


def check_shapes_and_finiteness():
    torch.manual_seed(0)
    B, T = 4, 37  # нечётная длина — проверяем хвосты MHLA/ELSA
    for dim, heads in [(256, 4), (512, 8)]:
        x = torch.randn(B, T, dim)
        for at in ATTENTION_TYPES:
            y = make_attention(at, dim, num_heads=heads).eval()(x)
            assert y.shape == x.shape, (at, dim, y.shape)
            assert torch.isfinite(y).all(), f"{at} dim={dim}: NaN/Inf"
    print("[1] формы и конечность — OK")


@torch.no_grad()
def check_elsa_equals_softmax(dim=256, heads=4, N=137, block=37, tol=1e-4):
    ref = StandardSelfAttention(dim, heads).eval()
    elsa = ELSAAttention(dim, heads, block_size=block).eval()
    elsa.load_state_dict(ref.state_dict())
    x = torch.randn(2, N, dim)
    err = (ref(x) - elsa(x)).abs().max().item()
    assert err < tol, f"ELSA расходится с softmax: {err:.2e}"
    print(f"[2] ELSA≡softmax — OK (макс|Δ|={err:.2e})")


@torch.no_grad()
def check_encoders_build():
    emb = torch.randn(4, 40, 384)
    for attn in ("default",) + ATTENTION_TYPES:
        eo = EmotionTransformer(attn_type=attn).eval()(emotion_input=emb, return_features=True)
        ao = AHTransformer(attn_type=attn).eval()(ah_input=emb, return_features=True)
        assert eo["emotion_logits"].shape == (4, 7)
        assert ao["ah_scores"].shape == (4, 2)
    print("[3] EMO/AH энкодеры строятся для всех attn_type — OK")


if __name__ == "__main__":
    check_shapes_and_finiteness()
    check_elsa_equals_softmax()
    check_encoders_build()
    print("\nВсе проверки пройдены.")
