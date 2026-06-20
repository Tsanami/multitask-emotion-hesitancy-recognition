import os
from dataclasses import dataclass, field

# ── Корневые директории ────────────────────────────────────────────────────────
# Привязаны к расположению пакета mehr/, а не к cwd — работает из любой папки.
PKG_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../mehr
DATA_DIR    = os.path.join(PKG_DIR, "data")
MOSEI_DIR   = os.path.join(DATA_DIR, "raw", "EAAI", "CMU-MOSEI")  # новый источник MOSEI (EAAI)
BAH_DIR     = os.path.join(DATA_DIR, "raw", "bah_data", "split")  # BAH split (train/val/test.txt)
CACHE_DIR   = os.path.join(DATA_DIR, "embeddings_cache")
RESULTS_DIR = os.path.join(PKG_DIR, "results")                    # чекпойнты/истории/трекинг


@dataclass
class BaseConfig:
    # Данные
    mosei_path:   str = MOSEI_DIR
    bah_path:     str = BAH_DIR
    output_dir:   str = RESULTS_DIR
    encoder_model: str = "bge-small"       # ключ из SUPPORTED_MODELS

    # Пути к кэшу эмбеддингов. Файл есть → загрузить; нет → вычислить и сохранить сюда.
    # MOSEI помечен суффиксом eaai, чтобы не подтянуть устаревший кэш старого mosei_data.
    mosei_train_emb:      str = f"{CACHE_DIR}/CMU-MOSEI_train_eaai_bge-small_embeddings.pkl"
    mosei_validation_emb: str = f"{CACHE_DIR}/CMU-MOSEI_validation_eaai_bge-small_embeddings.pkl"
    mosei_test_emb:       str = f"{CACHE_DIR}/CMU-MOSEI_test_eaai_bge-small_embeddings.pkl"
    bah_train_emb:        str = f"{CACHE_DIR}/BAH_train_bge-small_embeddings.pkl"
    bah_val_emb:          str = f"{CACHE_DIR}/BAH_val_bge-small_embeddings.pkl"
    bah_test_emb:         str = f"{CACHE_DIR}/BAH_test_bge-small_embeddings.pkl"
    # Обучение.
    batch_size:   int   = 32
    epochs:       int   = 100
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    max_patience: int   = 15

    # Трекинг экспериментов (MLflow). run_name="" → авто из имени стадии/сида.
    use_mlflow: bool = True
    run_name:   str  = ""

     # Модели стадии 1
    emo_model_path: str = f"{RESULTS_DIR}/Transformer_bge-small_emotion.pt"
    ah_model_path:  str = f"{RESULTS_DIR}/Transformer_bge-small_ah.pt"

    # Fusion архитектура
    hidden_dim:            int   = 256
    out_features:          int   = 512
    num_transformer_heads: int   = 4
    tr_layer_number:       int   = 1
    dropout:               float = 0.1

    # Лосс
    flag_emo_weight: bool = False    # веса w_c=(K-k_c)/k_c из статьи (Figure 4)
    flag_ah_weight:  bool = True



@dataclass
class Stage2NoSSLConfig(BaseConfig):
    use_ssl:     bool = False
    output_path: str  = f"{RESULTS_DIR}/fusion_no_ssl.pt"
    history_path: str = f"{RESULTS_DIR}/fusion_no_ssl_history.json"

    # Заглушки — не используются
    ssl_warmup_epochs: int   = 0
    ssl_conf_thr_emo:  float = 0.6
    ssl_conf_thr_ah:   float = 0.6
    lambda_ssl:        float = 0.2


@dataclass
class Stage2SSLConfig(BaseConfig):
    use_ssl:           bool  = True
    output_path:       str   = f"{RESULTS_DIR}/fusion_ssl.pt"
    history_path:      str   = f"{RESULTS_DIR}/fusion_ssl_history.json"

    ssl_warmup_epochs: int   = 2
    ssl_conf_thr_emo:  float = 0.6    # из config.toml статьи
    ssl_conf_thr_ah:   float = 0.6
    lambda_ssl:        float = 0.2


@dataclass
class Stage1Config(BaseConfig):
    emo_output_path: str = f"{RESULTS_DIR}/Transformer_bge-small_emotion.pt"
    ah_output_path:  str = f"{RESULTS_DIR}/Transformer_bge-small_ah.pt"


@dataclass
class Stage2GradNormConfig(BaseConfig):
    use_ssl:           bool  = True
    use_gradnorm:      bool  = True
    output_path:       str   = f"{RESULTS_DIR}/fusion_gradnorm.pt"
    history_path:      str   = f"{RESULTS_DIR}/fusion_gradnorm_history.json"

    ssl_warmup_epochs: int   = 2
    ssl_conf_thr_emo:  float = 0.6
    ssl_conf_thr_ah:   float = 0.6
    lambda_ssl:        float = 0.2

    # GradNorm гиперпараметры (из config.toml статьи)
    alpha_sup:  float = 1.0
    w_lr_sup:   float = 0.005
    alpha_ssl:  float = 1.5
    w_lr_ssl:   float = 0.01
    w_floor:    float = 1e-3
