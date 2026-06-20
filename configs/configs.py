from dataclasses import dataclass, field


@dataclass
class BaseConfig:
    # Данные
    mosei_path:   str = "mosei_data"
    bah_path:     str = "data/split"
    output_dir:   str = "best_models_emo_weights_off_20_06"
    encoder_model: str = "bge-small"       # ключ из SUPPORTED_MODELS

    # Пути к предвычисленным эмбеддингам (None = вычислить на лету)
    mosei_train_emb:      str = "mehr/data/embeddings_cache/CMU-MOSEI_train_bge-small_embeddings.pkl"
    mosei_validation_emb: str = "mehr/data/embeddings_cache/CMU-MOSEI_validation_bge-small_embeddings.pkl"
    mosei_test_emb:       str = "mehr/data/embeddings_cache/CMU-MOSEI_test_bge-small_embeddings.pkl"
    bah_train_emb:        str = "mehr/data/embeddings_cache/BAH_train_bge-small_embeddings.pkl"
    bah_val_emb:          str = "mehr/data/embeddings_cache/BAH_val_bge-small_embeddings.pkl"
    bah_test_emb:         str = "mehr/data/embeddings_cache/BAH_test_bge-small_embeddings.pkl"
    # Обучение. 
    batch_size:   int   = 32 
    epochs:       int   = 100
    lr:           float = 1e-4
    weight_decay: float = 1e-5
    max_patience: int   = 15

     # Модели стадии 1
    emo_model_path: str = "best_models/Transformer_bge-small_emotion.pt"
    ah_model_path:  str = "best_models/Transformer_bge-small_ah.pt"

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
    output_path: str  = "best_models/fusion_no_ssl.pt"
    history_path: str = "best_models/fusion_no_ssl_history.json"

    # Заглушки — не используются
    ssl_warmup_epochs: int   = 0
    ssl_conf_thr_emo:  float = 0.6
    ssl_conf_thr_ah:   float = 0.6
    lambda_ssl:        float = 0.2


@dataclass
class Stage2SSLConfig(BaseConfig):
    use_ssl:           bool  = True
    output_path:       str   = "best_models/fusion_ssl.pt"
    history_path:      str   = "best_models/fusion_ssl_history.json"

    ssl_warmup_epochs: int   = 2
    ssl_conf_thr_emo:  float = 0.6    # из config.toml статьи
    ssl_conf_thr_ah:   float = 0.6
    lambda_ssl:        float = 0.2


@dataclass
class Stage1Config(BaseConfig):
    emo_output_path: str = "best_models/Transformer_bge-small_emotion.pt"
    ah_output_path:  str = "best_models/Transformer_bge-small_ah.pt"


@dataclass
class Stage2GradNormConfig(BaseConfig):
    use_ssl:           bool  = True
    use_gradnorm:      bool  = True
    output_path:       str   = "best_models/fusion_gradnorm.pt"
    history_path:      str   = "best_models/fusion_gradnorm_history.json"

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
