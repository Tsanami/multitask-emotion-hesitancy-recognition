"""
Построение графиков для НИРа. Поддерживает два источника данных:

  1. JSON history (обратная совместимость):
       python -m utils.plotting mehr/results/E4_fusion_ssl_seed42_history.json

  2. MLflow run по имени или run_id:
       python -m utils.plotting --run  E4_fusion_ssl_seed42
       python -m utils.plotting --run-id b5fb589368c5...

  3. Сравнение нескольких runs на одном графике:
       python -m utils.plotting --compare E3_fusion_no_ssl_seed42 E4_fusion_ssl_seed42

Строит для каждого run:
  - кривые train/val mF1 по эпохам (EMO + AH)
  - кривую val overall F1 по эпохам
  - SSL coverage по эпохам (если есть)
  - распределение псевдо-меток эмоций по классам, последняя эпоха (если есть)

При --compare строит только кривую val overall F1 — overlay всех runs.
"""
import sys
import json
import os
import tempfile
import numpy as np

EMO_NAMES = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]


# ── Загрузчики ─────────────────────────────────────────────────────────────────

def _load_from_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _mlflow_client():
    from utils.tracking import TRACKING_URI
    from mlflow.tracking import MlflowClient
    return MlflowClient(TRACKING_URI)


def _all_runs():
    """Возвращает DataFrame всех runs эксперимента (отсортировано по времени)."""
    import mlflow
    from utils.tracking import TRACKING_URI, EXPERIMENT_NAME
    mlflow.set_tracking_uri(TRACKING_URI)
    df = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        order_by=["start_time DESC"],
    )
    return df


def list_runs():
    """Печатает таблицу доступных runs (для --list)."""
    df = _all_runs()
    if df.empty:
        print("Нет runs в эксперименте 'ssl_mepr'.")
        return
    cols = ["run_id", "tags.mlflow.runName", "status",
            "metrics.test_emo_mf1", "metrics.test_ah_mf1", "metrics.test_overall_f1"]
    cols = [c for c in cols if c in df.columns]
    out = df[cols].copy()
    out["run_id"] = out["run_id"].str[:12]          # укорочённый ID для читаемости
    out.columns = [c.replace("tags.mlflow.", "").replace("metrics.", "") for c in cols]
    print(out.to_string(index=False))


def _get_run_id(client, run_name: str = None, run_id: str = None) -> str:
    """
    Возвращает полный run_id.
    - run_name: ищет по тегу tags.mlflow.runName (регистронезависимо).
    - run_id: поддерживает как полный UUID, так и prefix (≥4 символа).
    """
    df = _all_runs()
    if df.empty:
        raise ValueError("Нет runs в эксперименте 'ssl_mepr'. Сначала запусти обучение.")

    name_col = "tags.mlflow.runName"

    if run_id:
        # Полный ID или prefix-match
        if run_id in df["run_id"].values:
            return run_id
        matches = df[df["run_id"].str.startswith(run_id)]
        if matches.empty:
            _print_available(df, name_col)
            raise ValueError(f"run_id с префиксом '{run_id}' не найден.")
        if len(matches) > 1:
            print(f"[plotting] Префикс '{run_id}' неоднозначен ({len(matches)} совпадений), берём последний")
        return matches.iloc[0]["run_id"]

    # Поиск по run_name через тег (MLflow хранит имя в tags.mlflow.runName)
    if name_col not in df.columns:
        _print_available(df, None)
        raise ValueError("Колонка run_name не найдена. Укажи --run-id.")
    matches = df[df[name_col] == run_name]
    if matches.empty:
        _print_available(df, name_col)
        raise ValueError(f"Run '{run_name}' не найден.")
    if len(matches) > 1:
        print(f"[plotting] Найдено {len(matches)} runs с именем '{run_name}', берём последний")
    return matches.iloc[0]["run_id"]


def _print_available(df, name_col):
    print("\nДоступные runs:")
    rows = []
    for _, r in df.iterrows():
        name = r.get(name_col, "") if name_col else ""
        rows.append(f"  {r['run_id'][:12]}  {name}")
    print("\n".join(rows[:20]))
    if len(rows) > 20:
        print(f"  ... и ещё {len(rows)-20}. Используй --list для полного списка.")


def _metric_history(client, rid: str, key: str) -> list:
    """Возвращает список float-значений по шагам (отсортировано)."""
    entries = client.get_metric_history(rid, key)
    if not entries:
        return []
    return [m.value for m in sorted(entries, key=lambda m: m.step)]


def _load_from_mlflow(run_name: str = None, run_id: str = None) -> dict:
    """
    Загружает метрики run из MLflow в формат, совместимый с JSON history.
    pseudo_emo_hist берётся из артефакта (history.json), если залогирован.
    """
    client = _mlflow_client()
    rid = _get_run_id(client, run_name, run_id)

    h = {}
    metric_map = {
        "train_emo_mf1": "train_emo_mf1",
        "train_ah_mf1":  "train_ah_mf1",
        "val_emo_mf1":   "val_emo_mf1",
        "val_emo_uar":   "val_emo_uar",
        "val_ah_mf1":    "val_ah_mf1",
        "val_ah_uar":    "val_ah_uar",
        "val_overall_f1": "val_overall_f1",
        "train_cov_ssl_emo": "train_cov_ssl_emo",
        "train_cov_ssl_ah":  "train_cov_ssl_ah",
        "train_n_ssl_emo":   "train_n_ssl_emo",
        "train_n_ssl_ah":    "train_n_ssl_ah",
        "train_loss":    "train_loss",
        "lr":            "lr",
    }
    for dst, src in metric_map.items():
        vals = _metric_history(client, rid, src)
        if vals:
            h[dst] = vals

    if "train_emo_mf1" not in h:
        raise ValueError(f"Run {rid}: метрики не найдены. Убедись, что run завершён.")

    # pseudo_emo_hist: список списков (не числовой → в MLflow нет)
    # пробуем достать из артефакта history.json
    h["train_pseudo_emo_hist"] = _try_load_pseudo_hist(client, rid)

    return h


def _try_load_pseudo_hist(client, rid: str) -> list:
    """Скачивает залогированный history-артефакт и возвращает pseudo_emo_hist."""
    try:
        artifacts = client.list_artifacts(rid)
        history_art = next(
            (a for a in artifacts if a.path.endswith("_history.json")), None
        )
        if history_art is None:
            return []
        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(rid, history_art.path, tmp)
            with open(local) as f:
                data = json.load(f)
            return data.get("train_pseudo_emo_hist", [])
    except Exception:
        return []


# ── Графики ────────────────────────────────────────────────────────────────────

def _savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"saved {path}")


def plot_single(h: dict, title: str = "", out_prefix: str = "plot"):
    """Строит все 4 графика для одного run (из dict в формате history)."""
    import matplotlib.pyplot as plt

    epochs = range(1, len(h.get("train_emo_mf1", h.get("val_emo_mf1", []))) + 1)

    # 1. Кривые mF1 ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    if "train_emo_mf1" in h:
        ax.plot(epochs, h["train_emo_mf1"], label="train EMO mF1", linestyle="--")
    ax.plot(epochs, h["val_emo_mf1"],   label="val EMO mF1")
    if "train_ah_mf1" in h:
        ax.plot(epochs, h["train_ah_mf1"],  label="train AH mF1", linestyle="--")
    ax.plot(epochs, h["val_ah_mf1"],    label="val AH mF1")
    if "val_overall_f1" in h:
        ax.plot(epochs, h["val_overall_f1"], label="val overall F1", linewidth=2, color="black")
    ax.set_xlabel("Эпоха"); ax.set_ylabel("mF1 / F1")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title(f"Кривые обучения{' — ' + title if title else ''}")
    _savefig(fig, f"{out_prefix}_mf1.png")

    # 2. SSL coverage ────────────────────────────────────────────────────────
    cov_emo = h.get("train_cov_ssl_emo", [])
    cov_ah  = h.get("train_cov_ssl_ah", [])
    if cov_emo and any(v > 0 for v in cov_emo):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(epochs, cov_emo, label="EMO pseudo-label coverage")
        if cov_ah:
            ax.plot(epochs, cov_ah, label="AH pseudo-label coverage")
        ax.set_xlabel("Эпоха"); ax.set_ylabel("Доля уверенных псевдо-меток")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
        ax.set_title(f"SSL coverage{' — ' + title if title else ''}")
        _savefig(fig, f"{out_prefix}_ssl_coverage.png")

    # 3. Распределение псевдо-меток (последняя эпоха) ──────────────────────
    pseudo_hist = h.get("train_pseudo_emo_hist", [])
    if pseudo_hist:
        last = np.array(pseudo_hist[-1])
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.bar(EMO_NAMES, last)
        ax.set_ylabel("Кол-во псевдо-меток")
        ax.set_title(f"Псевдо-метки EMO, последняя эпоха{' — ' + title if title else ''}")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        _savefig(fig, f"{out_prefix}_pseudo_hist.png")


def plot_compare(runs: list[tuple[str, dict]], out_prefix: str = "compare"):
    """
    Накладывает кривые val overall F1 нескольких runs.
    runs: [(label, history_dict), ...]
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, h in runs:
        vals = h.get("val_overall_f1", [])
        if vals:
            ax.plot(range(1, len(vals) + 1), vals, label=label)
    ax.set_xlabel("Эпоха"); ax.set_ylabel("val overall F1")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    ax.set_title("Сравнение runs — val overall F1")
    _savefig(fig, f"{out_prefix}_compare.png")


# ── Обратная совместимость ─────────────────────────────────────────────────────

def plot_history(history_path: str, out_prefix: str = None):
    """Старый API: принимает путь к JSON и строит графики."""
    h = _load_from_json(history_path)
    prefix = out_prefix or history_path.replace(".json", "")
    title  = os.path.basename(history_path).replace("_history.json", "")
    plot_single(h, title=title, out_prefix=prefix)


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Графики обучения из JSON или MLflow"
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("json", nargs="?", help="Путь к history JSON (старый способ)")
    src.add_argument("--run",     metavar="RUN_NAME", help="MLflow run_name")
    src.add_argument("--run-id",  metavar="RUN_ID",   help="MLflow run_id (полный или префикс)")
    src.add_argument("--compare", nargs="+", metavar="RUN_NAME",
                     help="Несколько run_name для overlay val overall F1")
    src.add_argument("--list", action="store_true",
                     help="Показать все доступные runs и выйти")
    parser.add_argument("--out", metavar="PREFIX",
                        help="Префикс выходных PNG (по умолчанию — имя run / JSON)")

    args = parser.parse_args()

    if args.list:
        list_runs()
        sys.exit(0)

    if args.json:
        plot_history(args.json, args.out)

    elif args.run or args.run_id:
        run_kw = {"run_name": args.run} if args.run else {"run_id": args.run_id}
        title  = args.run or args.run_id
        prefix = args.out or title.replace("/", "_")
        h = _load_from_mlflow(**run_kw)
        plot_single(h, title=title, out_prefix=prefix)

    else:  # --compare
        loaded = []
        for name in args.compare:
            print(f"Loading {name}...")
            try:
                h = _load_from_mlflow(run_name=name)
                loaded.append((name, h))
            except ValueError as e:
                print(f"  WARNING: {e}")
        if not loaded:
            print("Нет данных для сравнения"); sys.exit(1)
        prefix = args.out or "compare"
        plot_compare(loaded, out_prefix=prefix)
