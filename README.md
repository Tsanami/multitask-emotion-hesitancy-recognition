# SSL-MEPR (текстовая модальность): ER + AHR

Полуконтролируемая многозадачная кросс-доменная структура обучения для текстового
распознавания эмоций (ER, CMU-MOSEI) и амбивалентности личности (AHR, BAH).

Воспроизведение текстовой части SSL-MEPR с заменой задачи personality на AHR.

## Структура

```
ssl_mepr/
├── configs/
│   └── configs.py          # BaseConfig, Stage1Config, Stage2NoSSLConfig, Stage2SSLConfig
├── data/
│   ├── preprocessing.py    # get_cmu_mosei_data, get_bah_data
│   ├── datasets.py         # DatasetEmotionAHFusion, custom_collate_fn
│   └── loaders.py          # get_stage1_loaders, get_stage2_loaders
├── models/
│   ├── blocks.py           # TransformerEncoderLayer, Mamba-блоки, позиц. кодирование
│   ├── emotion_transformer.py
│   ├── emotion_mamba.py    # альтернативный backbone
│   ├── ah_transformer.py
│   └── fusion_transformer.py
├── training/
│   ├── metrics.py          # predict_emotions + импорт mf1/uar из measures.py
│   ├── measures.py         # метрики статьи (mf1, uar, acc_func, ccc)
│   ├── losses.py           # build_criteria (веса классов)
│   ├── stage1_epochs.py    # train/eval для single-task
│   └── epochs.py           # train/eval для fusion + SSL
├── utils/
│   ├── seed.py             # set_seed для воспроизводимости
│   └── history.py          # History → JSON
├── train_stage1.py         # точка входа: обучить emotion или ah энкодер
├── train_stage2.py         # точка входа: обучить fusion (с/без SSL)
└── run_experiments.py      # ablation на нескольких сидах (mean ± std)
```

## Пайплайн

### Стадия 1 — unimodal энкодеры
```bash
python train_stage1.py --task emotion --seed 42 --mosei_path data/mosei
python train_stage1.py --task ah      --seed 42 --bah_path   data/bah
```

### Стадия 2 — cross-domain fusion
```bash
# без SSL (baseline E3)
python train_stage2.py --seed 42

# с SSL (E4 — воспроизведение статьи)
python train_stage2.py --ssl --seed 42

# с переопределением гиперпараметров
python train_stage2.py --ssl --ssl_conf_thr_emo 0.8 --lambda_ssl 0.3
```

### Ablation study (для НИРа)
```bash
python run_experiments.py
```
Прогоняет E3/E4/E5 на сидах [42, 0, 123], усредняет, печатает таблицу
mean ± std и дельту E4−E3 (вклад SSL).

## Данные

- **MOSEI** (ER): CSV с колонками `text, Neutral, Anger, Disgust, Fear, Happiness, Sadness, Surprise`.
  Файлы: `train.csv`, `validation.csv`, `test.csv`.
- **BAH** (AHR): txt-файлы `train.txt`, `val.txt`, `test.txt` формата `id,label,text`.

При первом запуске эмбеддинги BGE-small вычисляются и кешируются в
`<path>/embeddings_cache/`. Дальше передавай пути к `.pkl` через поля конфига
(`mosei_train_emb`, `bah_train_emb`, ...) чтобы не пересчитывать.

## Ключевые детали воспроизведения

- Backbone: `BAAI/bge-small-en-v1.5` (384d), вход — вся последовательность токенов.
- ER: 7 классов (Neutral на индексе 0), CE-лосс с dominant-меткой (argmax).
- Предсказание эмоций: softmax → пороги 1/7 и 6/7 (`transform_matrix`).
- AHR: бинарная классификация, CE с весами классов.
- SSL: псевдо-метки по порогу уверенности (softmax + argmax), отдельно для каждой задачи.
- SSL confidence threshold = 0.6 (из config.toml статьи), warmup 2 эпохи.
