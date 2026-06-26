"""
Сменные механизмы внимания для stage-1 энкодеров (ER / AHR).

Портировано из домашних работ курса (HW6/HW7/HW8) с адаптацией под 1D
последовательности токенов [B, T, D], которые подаёт BGE-эмбеддер этого проекта.

Варианты (3 — из статей papers/, + softmax-baseline):
    softmax  — StandardSelfAttention: классическое scaled-dot-product, O(T²·D)
    zeros    — ZeroSAttention: Zero-Sum Linear Attention (zeros.pdf). Из softmax-весов
               убираются 0-й (1/N) и 1-й (δ/N) порядки, остаток ε гейтится обучаемыми
               σ; веса суммируются в 0 → допускают ОТРИЦАТЕЛЬНЫЕ значения (контрастность).
    mhla     — MHLA1D: Multi-Head Linear Attention (MHLA.pdf). Сплит токенов на M блоков,
               локальные KV-сводки + multi-head mixing → восстанавливает query-зависимость.
    elsa     — ELSAAttention: Exact Linear-Scan Attention (ELSA.pdf). Блочный online-softmax;
               численно == softmax, но O(block) памяти на скоры (не материализует T×T).

`make_attention(attn_type, dim, num_heads)` — фабрика модулей.
`AttentionEncoderLayer` — pre-norm трансформер-слой, в который вставляется любой из них.

ВАЖНО: модули self-attention (query=key=value=x). Default-путь проекта
(nn.MultiheadAttention / nn.TransformerEncoderLayer) НЕ затрагивается — он
используется при attn_type='default'. Эти варианты активируются только при явном
выборе одного из {softmax, linear, mhla, elsa}.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ════════════════════════════════════════════════════════════════════════════
# 1. Softmax (полный self-attention) — baseline для attention-ablation
# ════════════════════════════════════════════════════════════════════════════
class StandardSelfAttention(nn.Module):
    """Классический multi-head scaled-dot-product. Сложность O(T²·D)."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} не делится на num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv
        )
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)


# ════════════════════════════════════════════════════════════════════════════
# 2. ZeroS — Zero-Sum Linear Attention (zeros.pdf, §3.2 Reweighted Zero-sum Softmax)
# ════════════════════════════════════════════════════════════════════════════
class ZeroSAttention(nn.Module):
    """Zero-Sum Linear Attention (ZeroS, Lu et al., zeros.pdf).

    Идея статьи: разложить softmax на порядки и убрать 0-й (1/N, даёт лишь
    усреднение) и оставить 1-й (δ/N) + высшие (остаток ε) с обучаемыми гейтами.
    Для логитов s_{i,j} (query i по ключам j), s̄_i = mean_j s_{i,j}, δ = s − s̄:
        ε_{i,j} = softmax(s)_{i,j} − 1/N − δ_{i,j}/N          (Σ_j ε = 0)
        w_{i,j} = σ1·δ_{i,j}/N + σh·ε_{i,j},   Σ_j w_{i,j} = 0
        out_i   = Σ_j w_{i,j} v_j
    σ1=sigmoid(g1), σh=sigmoid(gh) — обучаемые скаляры на голову. Веса zero-sum →
    допускают отрицательные значения (контрастные/дифференциальные операции, чего
    выпуклая softmax-комбинация и vanilla-linear не могут — Prop. 3.1 статьи).

    keep_zero_order=True возвращает 0-й член (σ0·1/N, σ0=tanh(g0)) — в статье его
    советуют держать в ПЕРВОМ слое, чтобы не терять направление среднего (§3.1).

    Прим.: O(N²) по флопам (плотный softmax). Линейно-временна́я prefix-sum форма
    статьи нужна для каузальных LM; на коротком двунаправленном энкодере (T<60)
    считаем плотно — это та же «reweighted zero-sum softmax», ядро метода.
    """

    def __init__(self, dim, num_heads=4, keep_zero_order=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} не делится на num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        # гейты на голову: g1 — 1-й порядок (δ/N), gh — высшие (остаток ε)
        self.g1 = nn.Parameter(torch.zeros(num_heads))
        self.gh = nn.Parameter(torch.zeros(num_heads))
        self.keep_zero_order = keep_zero_order
        if keep_zero_order:
            self.g0 = nn.Parameter(torch.zeros(num_heads))

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv
        )
        s = (q @ k.transpose(-2, -1)) * self.scale         # логиты [B,H,N,N]
        sm = s.softmax(dim=-1)
        s_bar = s.mean(dim=-1, keepdim=True)               # s̄_i  [B,H,N,1]
        delta = s - s_bar                                  # δ
        eps = sm - 1.0 / N - delta / N                     # zero-sum остаток (Σ_j=0)

        sig1 = torch.sigmoid(self.g1).view(1, self.num_heads, 1, 1)
        sigh = torch.sigmoid(self.gh).view(1, self.num_heads, 1, 1)
        w = sig1 * (delta / N) + sigh * eps                # zero-sum веса
        if self.keep_zero_order:
            w = w + torch.tanh(self.g0).view(1, self.num_heads, 1, 1) * (1.0 / N)

        out = (w @ v).transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)


# ════════════════════════════════════════════════════════════════════════════
# 3. MHLA1D — Multi-Head Linear Attention, адаптация на 1D (чанки + mixing)
# ════════════════════════════════════════════════════════════════════════════
class MHLA1D(nn.Module):
    """Разбивает последовательность на M чанков, считает локальные kv-сводки,
    смешивает их обучаемой матрицей coeff (softmax). Падает на обычное LA, если
    T < M. Хвост (T % M) обрабатывается обычным LA."""

    def __init__(self, dim, num_heads=4, num_heads_mhla=4, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.M = num_heads_mhla
        self.eps = eps
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)
        self.to_out = nn.Linear(dim, dim, bias=False)
        self.mixing_coeff = nn.Parameter(torch.randn(self.M, self.M) * 0.01)

    def _phi(self, x):
        return F.relu(x) + self.eps

    def _plain_la(self, q, k, v):
        kv = torch.einsum('b n d, b n e -> b d e', k, v)
        k_sum = k.sum(dim=1)
        num = torch.einsum('b n d, b d e -> b n e', q, kv)
        den = torch.einsum('b n d, b d -> b n', q, k_sum).unsqueeze(-1).clamp(min=self.eps)
        return num / den

    def forward(self, x):
        B, N, D = x.shape
        q = self._phi(self.to_q(x))
        k = self._phi(self.to_k(x))
        v = self.to_v(x)

        chunk_size = N // self.M
        if chunk_size == 0:                      # последовательность короче M
            return self.to_out(self._plain_la(q, k, v))

        L = chunk_size * self.M
        q_r = q[:, :L, :].view(B, self.M, chunk_size, D)
        k_r = k[:, :L, :].view(B, self.M, chunk_size, D)
        v_r = v[:, :L, :].view(B, self.M, chunk_size, D)

        kv_summary = torch.einsum('b m n d, b m n e -> b m d e', k_r, v_r)
        z_summary = k_r.sum(dim=2)

        coeff = F.softmax(self.mixing_coeff, dim=-1)
        mixed_kv = torch.einsum('i m, b m d e -> b i d e', coeff, kv_summary)
        mixed_z = torch.einsum('i m, b m d -> b i d', coeff, z_summary)

        numerator = torch.einsum('b i n d, b i d e -> b i n e', q_r, mixed_kv)
        denominator = torch.einsum('b i n d, b i d -> b i n', q_r, mixed_z).unsqueeze(-1).clamp(min=self.eps)
        out_r = (numerator / denominator).view(B, L, D)

        out = torch.zeros_like(x)
        out[:, :L, :] = out_r
        if N % self.M != 0:                      # хвост → обычное LA
            out[:, L:, :] = self._plain_la(q[:, L:, :], k, v)
        return self.to_out(out)


# ════════════════════════════════════════════════════════════════════════════
# 4. ELSA — блочный online-softmax (Flash-стиль). Численно == softmax.
# ════════════════════════════════════════════════════════════════════════════
class ELSAAttention(nn.Module):
    """Тот же softmax, но посчитанный потоково по блокам ключей (running max +
    нормировка). O(T²) по флопам, но O(block_size) по пиковой памяти на скоры —
    не материализует полную матрицу T×T. Результат идентичен StandardSelfAttention
    при тех же весах (см. tests/check_elsa)."""

    def __init__(self, dim, num_heads=4, block_size=128):
        super().__init__()
        assert dim % num_heads == 0, f"dim={dim} не делится на num_heads={num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.block_size = block_size
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.view(B, N, self.num_heads, self.head_dim).transpose(1, 2), qkv
        )

        m_i = torch.full((B, self.num_heads, N, 1), float('-inf'), device=x.device, dtype=q.dtype)
        l_i = torch.zeros((B, self.num_heads, N, 1), device=x.device, dtype=q.dtype)
        acc = torch.zeros((B, self.num_heads, N, self.head_dim), device=x.device, dtype=q.dtype)

        for start in range(0, N, self.block_size):
            end = min(start + self.block_size, N)
            k_blk = k[:, :, start:end, :]
            v_blk = v[:, :, start:end, :]
            s = torch.einsum('bhnd,bhkd->bhnk', q, k_blk) * self.scale
            m_blk = s.max(dim=-1, keepdim=True).values
            m_new = torch.maximum(m_i, m_blk)
            alpha = torch.exp(m_i - m_new)
            p = torch.exp(s - m_new)
            l_i = l_i * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.einsum('bhnk,bhkd->bhnd', p, v_blk)
            m_i = m_new

        out = acc / l_i.clamp(min=1e-20)
        out = out.transpose(1, 2).reshape(B, N, D)
        return self.to_out(out)


# ════════════════════════════════════════════════════════════════════════════
# Фабрика + сменный pre-norm слой
# ════════════════════════════════════════════════════════════════════════════
ATTENTION_TYPES = ("softmax", "zeros", "mhla", "elsa")


def make_attention(attn_type, dim, num_heads=4, **kwargs):
    """Возвращает модуль self-attention по строковому ключу."""
    if attn_type == "softmax":
        return StandardSelfAttention(dim, num_heads=num_heads)
    if attn_type == "zeros":
        return ZeroSAttention(dim, num_heads=num_heads,
                              keep_zero_order=kwargs.get("keep_zero_order", False))
    if attn_type == "mhla":
        return MHLA1D(dim, num_heads=num_heads,
                      num_heads_mhla=kwargs.get("num_heads_mhla", num_heads))
    if attn_type == "elsa":
        return ELSAAttention(dim, num_heads=num_heads,
                             block_size=kwargs.get("block_size", 128))
    raise ValueError(f"Unknown attn_type={attn_type!r}; ожидается один из {ATTENTION_TYPES}")


class AttentionEncoderLayer(nn.Module):
    """Pre-norm трансформер-слой (как в HW7 CustomTransformerLayer):
        x = x + attn(norm1(x))
        x = x + ff(norm2(x))
    В attn вставляется любой модуль из make_attention. FF — Linear→GELU→Linear.
    """

    def __init__(self, dim, attn_module, ff_mult=1, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = attn_module
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * ff_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class CustomAttentionEncoder(nn.Module):
    """Стек AttentionEncoderLayer + обучаемое позиционное кодирование.
    Drop-in замена TransformerModelWithAttention при attn_type != 'default'.
    Сигнатура forward совместима: [B, T, D] → [B, T, D]."""

    def __init__(self, hidden_dim, num_heads, num_layers, attn_type="softmax",
                 dropout=0.1, max_len=1000, **attn_kwargs):
        super().__init__()
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_len, hidden_dim))
        self.layers = nn.ModuleList([
            AttentionEncoderLayer(
                hidden_dim,
                make_attention(attn_type, hidden_dim, num_heads=num_heads, **attn_kwargs),
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

    def forward(self, x):
        T = x.size(1)
        x = x + self.positional_encoding[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        return x
