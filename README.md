# SSL-MEPR (текстовая модальность): ER + AHR

Полуконтролируемое многозадачное кросс-доменное обучение для **текстового**
распознавания эмоций (ER, CMU-MOSEI) и амбивалентности/хеджирования (AHR, BAH).

Воспроизведение текстовой части [SSL-MEPR](https://github.com/LEYA-HSE/SSL-MEPR) с
заменой задачи personality (FIv2) на AHR (BAH). **M** в MEPR = multitask.

**Исследовательский вопрос:** даёт ли кросс-доменная псевдоразметка (SSL) прирост
mF1 на обеих задачах относительно multi-task без SSL?

---

## Структура

```
mehr/
├── configs/
│   └── configs.py          # пути (от PKG_DIR), BaseConfig + Stage1/Stage2/GradNorm,
│                           #   unfreeze_encoders, encoder_lr, use_mlflow, run_name
├── data/
│   ├── preprocessing.py    # get_cmu_mosei_data (EAAI), get_bah_data
│   ├── datasets.py         # DatasetEmotionAHFusion, custom_collate_fn, кэш эмбеддингов
│   ├── loaders.py          # get_stage1_loaders, get_stage2_loaders
│   └── raw/
│       ├── EAAI/CMU-MOSEI/  # train_full.csv, dev_full.csv, test_full.csv
│       └── bah_data/split/  # train.txt, val.txt, test.txt
├── models/
│   ├── blocks.py           # TransformerEncoderLayer, Mamba, позиц. кодирование
│   ├── emotion_transformer.py / emotion_mamba.py (альт.) / ah_transformer.py
│   └── fusion_transformer.py  # cross-attention fusion (+ unfreeze_encoders)
├── training/
│   ├── measures.py         # mf1, uar, mwacc, wf1, acc_func, ccc
│   ├── metrics.py          # predict_emotions (transform_matrix) + реэкспорт measures
│   ├── losses.py           # build_criteria (веса классов)
│   ├── stage1_epochs.py    # train/eval single-task
│   ├── epochs.py           # train/eval fusion + SSL (+ coverage/pseudo-label трекинг)
│   ├── epochs_gradnorm.py  # train fusion + GradNorm (с тем же трекингом)
│   ├── gradnorm.py         # GradNormMultiTaskLoss (wallet-вариант)
│   └── significance.py     # парный тест значимости по test_* из MLflow
├── utils/
│   ├── seed.py history.py checkpointing.py
│   ├── tracking.py         # обёртка MLflow (SQLite-бэкенд); no-op если выключено
│   └── plotting.py         # графики из JSON-истории ИЛИ из MLflow-run
├── notebooks/              # EDA_ER, results_analysis, mosei_comparison
├── train_stage1.py         # точка входа: emotion / ah энкодер
├── train_stage2.py         # точка входа: fusion (с/без SSL, ±разморозка, ±GradNorm)
└── run_experiments.py      # ablation на сидах (mean ± std), послед./параллельно
```

---

## Установка

```bash
pip install -r requirements.txt   # torch, transformers, mlflow, scipy, ...
```

Backbone `BAAI/bge-small-en-v1.5` (384d) скачивается из HuggingFace при первом
вычислении эмбеддингов.

---

## Данные

Все пути строятся от расположения пакета (`configs.configs.PKG_DIR`) — скрипты
работают из любого cwd.

- **MOSEI** (ER, EAAI-версия): `data/raw/EAAI/CMU-MOSEI/{train_full,dev_full,test_full}.csv`.
  Внутренние имена частей `train/validation/test` → файлы через `MOSEI_PART_FILES`.
  Колонки: `video_name, text, Neutral, Anger, Disgust, Fear, Happiness, Sadness,
  Surprise, Other` — берутся 7 эмоций (Neutral на индексе 0), `Other` игнорируется.
  Размеры: **train 16274 / dev 1861 / test 4653**.
- **BAH** (AHR): `data/raw/bah_data/split/{train,val,test}.txt`, формат `id,label,text`
  (label 0/1). Размеры: **778 / 124 / 525**. Пайплайн текстовый — `id` это номер
  строки, не указатель на медиа.

**Эмбеддинги** кешируются в `data/embeddings_cache/`. Логика `path_to_emb`: файл
есть → загрузить (с проверкой, что длина кэша == числу текстов), нет → вычислить и
сохранить туда же. MOSEI-кэш помечен суффиксом `eaai`, чтобы не подтянуть
устаревший кэш старого `mosei_data`.

---

## Пайплайн

### Стадия 1 — unimodal энкодеры
```bash
python train_stage1.py --task emotion --seed 42
python train_stage1.py --task ah      --seed 42
```
Чекпойнты → `results/Transformer_bge-small_{emotion,ah}.pt` (их грузит стадия 2).

### Стадия 2 — cross-domain fusion
```bash
python train_stage2.py                       # baseline без SSL
python train_stage2.py --ssl                 # + SSL (воспроизведение статьи)
python train_stage2.py --gradnorm            # SSL + GradNorm
python train_stage2.py --ssl --unfreeze      # SSL + разморозка stage-1 энкодеров
python train_stage2.py --ssl --unfreeze --encoder_lr 5e-6   # мягче, если оверфит

# переопределение гиперпараметров SSL
python train_stage2.py --ssl --ssl_conf_thr_emo 0.8 --lambda_ssl 0.3
```
Разморозка (`--unfreeze`) размораживает **stage-1 трансформеры** emo_model/ah_model
(НЕ сам BGE — его в графе стадии 2 нет) и обучает их через дискриминативный LR
(`encoder_lr` < lr головы). Frozen-путь при выключенном флаге не меняется.

### Ablation study
```bash
# последовательно (одна GPU/CPU)
python run_experiments.py

# параллельно: очередь (эксперимент × seed), N процессов на карту
python run_experiments.py --parallel --gpus 0,1,2,3,4,5,6,7 --per-gpu 8
python run_experiments.py --parallel --gpus 0 --per-gpu 8     # одна карта тоже ок
```
Прогоняет эксперименты на `SEEDS` (по умолчанию 10 сидов), печатает таблицу
mean ± std и дельту E4−E3 (вклад SSL). Результаты → `results/ablation_results.json`,
логи параллельных заданий → `results/_jobout/`.

| ID | Конфиг | Описание |
|----|--------|----------|
| E3 | `fusion_no_ssl`        | Fusion без SSL — baseline стадии 2 |
| E4 | `fusion_ssl`           | SSL thr=0.6, λ=0.2 — репликация статьи |
| E5 | `fusion_gradnorm`      | SSL + GradNorm (динамическая балансировка) |
| E6 | `fusion_ssl_unfreeze`  | SSL + разморозка stage-1 энкодеров |

Ключевые сравнения: **E3 vs E4** (вклад SSL), **E4 vs E6** (вклад разморозки).
Набор экспериментов задаётся словарём `EXPERIMENTS` в `run_experiments.py` —
новый эксперимент = одна лямбда-фабрика конфига, без отдельного скрипта.

---

## Метрики

Считаются в `training/measures.py` (per-column macro avg → mean — как в статье,
**не** `sklearn.f1_score(average='macro')`).

| Метрика | Задача | Что это |
|---------|--------|---------|
| `emo_mf1`, `ah_mf1` | ER / AHR | mean Macro-F1 (per-column → mean) |
| `emo_uar`, `ah_uar` | ER / AHR | mean UAR (macro-avg recall) |
| `emo_mwacc` | **только ER** | mean weighted accuracy: `1/C·Σ ½(TP/(TP+FN)+TN/(TN+FP))`, C=6 эмоций, balanced accuracy на колонку → среднее |
| `ah_wf1` | **только AHR** | weighted-average F1 (взвешенный по поддержке классов) |
| `overall_f1` | обе | `(emo_mf1 + ah_mf1) / 2` — метрика выбора лучшей модели |

Эмоции — multi-label предсказание: softmax → `transform_matrix` (если
`prob[Neutral] ≥ 6/7` → всё 0; иначе эмоция=1 при `prob ≥ 1/7`).

---

## Трекинг экспериментов (MLflow)

Логирование встроено в `train_stage1/2` и `run_experiments` через
`utils/tracking.py`: параметры конфига + seed, метрики по эпохам (`train_*`,
`val_*`, `lr`), финальные `test_*`, артефакты (чекпойнт, `*.config.json`,
история). Бэкенд — SQLite (`results/mlflow.db`), артефакты — `results/mlartifacts/`.
Отключить: `cfg.use_mlflow=False` (или если mlflow не установлен — тихий no-op).

```bash
mlflow ui --backend-store-uri sqlite:///results/mlflow.db   # http://127.0.0.1:5000
```

### Графики (utils/plotting.py)
```bash
python -m utils.plotting --list                       # все runs
python -m utils.plotting --run E4_fusion_ssl_seed42   # из MLflow по имени
python -m utils.plotting --run-id b7e4c1a9            # по run_id (можно префикс)
python -m utils.plotting --compare E3_fusion_no_ssl_seed42 E4_fusion_ssl_seed42
python -m utils.plotting results/E4_fusion_ssl_seed42_history.json   # из JSON
```
Авто-режим: стадия 1 (кривые mF1/UAR) vs стадия 2 (EMO+AH mF1, SSL coverage,
гистограмма псевдо-меток).

### Значимость (training/significance.py)
```bash
python -m training.significance --list                # какие runs видны
python -m training.significance \
    --pair E3_fusion_no_ssl E4_fusion_ssl \
    --pair E4_fusion_ssl    E6_fusion_ssl_unfreeze \
    --metrics overall_f1 emo_mf1 ah_mf1 emo_mwacc ah_wf1
```
Парный анализ по сидам (один сид = одна пара, общий сплит → шум сида вычитается):
mean Δ, bootstrap 95% CI (10k), Wilcoxon, парный t, Cohen's d_z. Тянет `test_*` из
того же MLflow. На n<8 ориентир — CI, а не p-value.

---

## Ключевые детали воспроизведения

- **Backbone:** `BAAI/bge-small-en-v1.5` (384d), вход — вся последовательность
  токенов `[B,T,384]`. В графе стадии 2 BGE отсутствует (эмбеддинги предвычислены).
- **Параметры моделей:** EmotionTransformer `hidden=256, out=256, heads=4, layers=3`;
  AHTransformer `hidden=512, out=128, heads=8, layers=1, dropout=0.2`;
  FusionTransformer `hidden=256, out=512, heads=4, layers=1, dropout=0.1`.
- **ER лосс:** CE с dominant-меткой (`argmax(emo_label)`). Веса (если включены)
  `w_c=(K−k_c)/k_c`, `k_c` по multi-label (`intensity>0`), не по argmax.
- **AHR лосс:** CE с весами классов `len/(2·counts)`.
- **SSL (стадия 2):** псевдо-метки softmax→argmax→CE при `conf > порога`, отдельно
  для каждой задачи. Параметры статьи: `ssl_conf_thr=0.6`, `lambda_ssl=0.2`,
  warmup 2 эпохи.
- **Воспроизводимость:** `set_seed` фиксирует random/numpy/torch/cudnn; один и тот
  же `SEEDS` для всех экспериментов (иначе парный тест значимости невозможен).
- **Чекпойнты:** `save_checkpoint_with_config` → `model.pt` + `model.config.json`;
  выбор лучшей модели по val `overall_f1`.
