"""
Streamlit-демо SSL-MEHR: интерактивная проверка лучших fusion-моделей.

Вводишь текст → BGE-small считает эмбеддинги токенов [1,T,384] → выбранная
fusion-модель (стадия 2) выдаёт одновременно:
    • эмоции (ER, 7 классов CMU-MOSEI) — softmax + официальная мульти-лейбл
      бинаризация (transform_matrix: Neutral, либо набор эмоций с prob ≥ 1/7);
    • амбивалентность/хеджирование (AHR, бинарно на BAH) — P(хеджирование).

Тот же текст подаётся в обе ветки энкодеров (emotion_input == ah_input),
поэтому одна реплика даёт оба предсказания.

Запуск:
    ./.venv/bin/python -m streamlit run mehr/app.py
    # или из каталога mehr/:  streamlit run app.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st

# ── пути (от расположения файла, независимо от cwd) ────────────────────────────
PKG_DIR = Path(__file__).resolve().parent
if str(PKG_DIR) not in sys.path:
    sys.path.insert(0, str(PKG_DIR))

from models import EmotionTransformer, AHTransformer, FusionTransformer  # noqa: E402
from training.metrics import transform_matrix  # noqa: E402
from data.datasets import SUPPORTED_MODELS  # noqa: E402
from transformers import AutoTokenizer, AutoModel  # noqa: E402

RESULTS = PKG_DIR / "results"

EMO_LABELS = ["Neutral", "Anger", "Disgust", "Fear",
              "Happiness", "Sadness", "Surprise"]

# Лучшие модели стадии 2 (seed 42, веса классов эмоций включены).
# Метрики — средние тестовые по сидам из README (для подписи в UI).
MODEL_REGISTRY = {
    "E4 · fusion + SSL (лучший overall)": {
        "exp": "E4_fusion_ssl_emo_w",
        "note": "overall_f1 0.6374 · emo_mf1 0.5991 · ah_mf1 0.6758 — лучший общий результат",
    },
    "E6 · fusion + SSL + разморозка (лучший AH)": {
        "exp": "E6_fusion_ssl_unfreeze_emo_w",
        "note": "ah_mf1 0.6760 · ah_wf1 0.6847 · overall_f1 0.6372 — лучший по AH",
    },
    "E3 · fusion без SSL (baseline)": {
        "exp": "E3_fusion_no_ssl_emo_w",
        "note": "overall_f1 0.6332 — baseline стадии 2 (без псевдоразметки)",
    },
    "E5 · fusion + GradNorm": {
        "exp": "E5_fusion_gradnorm_emo_w",
        "note": "overall_f1 0.6271 — динамическая балансировка задач",
    },
}
SEED = 42

EXAMPLES = [
    "I absolutely loved this movie, it was the best thing I've seen all year!",
    "Well, I mean, maybe it's good? I'm not really sure, it kind of depends.",
    "I am so furious right now, this is completely unacceptable.",
    "It might be okay, or perhaps not — honestly I can't quite tell.",
    "That was terrifying, I couldn't stop shaking the whole time.",
]


# ── загрузка (кешируется на сессию) ───────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@st.cache_resource(show_spinner="Загрузка BGE-small энкодера…")
def load_encoder():
    name = SUPPORTED_MODELS["bge-small"]
    device = get_device()
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModel.from_pretrained(name).to(device).eval()
    return tok, mdl


@st.cache_resource(show_spinner="Загрузка fusion-модели…")
def load_fusion(exp: str):
    device = get_device()
    cfg_path = RESULTS / f"{exp}_seed{SEED}.config.json"
    pt_path = RESULTS / f"{exp}_seed{SEED}.pt"
    with open(cfg_path) as f:
        cfg = json.load(f)

    emo_model = EmotionTransformer(
        input_dim_emotion=384, hidden_dim=256, out_features=256,
        num_transformer_heads=4, tr_layer_number=3, dropout=0.0,
    ).to(device)
    ah_model = AHTransformer(
        input_dim_ah=384, hidden_dim=512, out_features=128,
        num_transformer_heads=8, tr_layer_number=1, dropout=0.2,
    ).to(device)

    model = FusionTransformer(
        emo_model=emo_model, ah_model=ah_model,
        hidden_dim=cfg["hidden_dim"], out_features=cfg["out_features"],
        num_transformer_heads=cfg["num_transformer_heads"],
        tr_layer_number=cfg["tr_layer_number"], dropout=cfg["dropout"],
        unfreeze_encoders=False,
    ).to(device)
    # Полный fusion-чекпойнт содержит и веса энкодеров (важно для E6-разморозки).
    model.load_state_dict(torch.load(pt_path, map_location=device))
    model.eval()
    return model


@torch.no_grad()
def embed(text: str):
    tok, mdl = load_encoder()
    device = get_device()
    enc = tok(text, padding=True, truncation=True, max_length=128,
              return_tensors="pt").to(device)
    return mdl(**enc).last_hidden_state  # [1, T, 384]


@torch.no_grad()
def predict(exp: str, text: str):
    model = load_fusion(exp)
    feat = embed(text)
    out = model(emotion_input=feat, ah_input=feat)
    emo_probs = F.softmax(out["emotion_logits"], dim=1)[0].cpu().numpy()
    ah_probs = F.softmax(out["ah_logits"], dim=1)[0].cpu().numpy()
    # Официальное мульти-лейбл решение проекта (Neutral либо набор эмоций).
    multilabel = transform_matrix(emo_probs[None, :])[0]  # [6] для 6 эмоций
    is_neutral = bool(multilabel.sum() == 0)
    return emo_probs, ah_probs, multilabel, is_neutral


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SSL-MEHR демо", page_icon="🎭", layout="wide")
st.title("🎭 SSL-MEHR — распознавание эмоций и хеджирования")
st.caption(
    "Текстовые fusion-модели (стадия 2): эмоции (CMU-MOSEI) + "
    "амбивалентность/хеджирование (BAH). Один текст → оба предсказания."
)

with st.sidebar:
    st.header("Модель")
    choice = st.radio("Чекпойнт (seed 42, +веса эмоций)",
                      list(MODEL_REGISTRY), index=0)
    st.info(MODEL_REGISTRY[choice]["note"])
    st.markdown("---")
    st.caption(f"Устройство: `{get_device()}`")
    st.caption("Эмоции бинаризуются правилом из статьи: если P(Neutral) ≥ 6/7 → "
               "Neutral; иначе эмоция засчитывается при P ≥ 1/7.")

# Примеры
st.write("**Примеры** (нажми, чтобы подставить):")
ex_cols = st.columns(len(EXAMPLES))
if "text" not in st.session_state:
    st.session_state.text = EXAMPLES[0]
for i, (col, ex) in enumerate(zip(ex_cols, EXAMPLES)):
    if col.button(f"#{i + 1}", help=ex, use_container_width=True):
        st.session_state.text = ex

text = st.text_area("Текст для анализа (English)", key="text", height=120)
go = st.button("Анализировать", type="primary", use_container_width=True)

if go and text.strip():
    exp = MODEL_REGISTRY[choice]["exp"]
    emo_probs, ah_probs, multilabel, is_neutral = predict(exp, text)

    left, right = st.columns(2)

    # ── Эмоции ────────────────────────────────────────────────────────────────
    with left:
        st.subheader("Эмоции (ER)")
        present = ["Neutral"] if is_neutral else [
            EMO_LABELS[1 + i] for i, v in enumerate(multilabel) if v > 0
        ]
        st.markdown("**Предсказание:** " +
                    (", ".join(present) if present else "—"))
        emo_table = {
            "эмоция": EMO_LABELS,
            "softmax": [float(p) for p in emo_probs],
        }
        st.bar_chart(emo_table, x="эмоция", y="softmax", height=300)

    # ── Хеджирование ─────────────────────────────────────────────────────────
    with right:
        st.subheader("Амбивалентность / хеджирование (AHR)")
        p_hedge = float(ah_probs[1])
        verdict = "🟠 Хеджирование / неоднозначность" if p_hedge >= 0.5 \
            else "🟢 Уверенное высказывание"
        st.markdown(f"**Предсказание:** {verdict}")
        st.metric("P(хеджирование)", f"{p_hedge:.1%}")
        st.progress(p_hedge)
        st.bar_chart(
            {"класс": ["уверенно (0)", "хеджирование (1)"],
             "softmax": [float(ah_probs[0]), float(ah_probs[1])]},
            x="класс", y="softmax", height=220,
        )

    with st.expander("Сырые softmax-вероятности"):
        st.json({
            "emotion": {l: round(float(p), 4)
                        for l, p in zip(EMO_LABELS, emo_probs)},
            "ah": {"confident_0": round(float(ah_probs[0]), 4),
                   "hedging_1": round(float(ah_probs[1]), 4)},
        })
elif go:
    st.warning("Введите текст.")
