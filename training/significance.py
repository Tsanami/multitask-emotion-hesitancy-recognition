"""
Парный анализ значимости для ablation НИРа (E3 vs E4, frozen vs unfreeze, ...).

ЗАЧЕМ
-----
Дельты в этом проекте крошечные (~0.002-0.012) и сопоставимы с разбросом по сидам.
"mean ± std по 3 сидам" не отвечает на вопрос "это сигнал или шум". Этот скрипт
отвечает на него корректно: ПАРНЫМ тестом (один и тот же seed = одна пара), потому
что обе модели в паре видят один и тот же train/val/test-сплит — парный тест на
порядок мощнее непарного на таких маленьких эффектах.

ИСТОЧНИК ДАННЫХ
---------------
Берём ЧЕСТНЫЕ test-метрики из MLflow (sqlite-бэкенд, как настроен в utils/tracking.py).
Каждый run матчится по run-name '<exp>_seed<seed>'. Тянем test_* метрики.
Никаких val-прокси: значимость считается на том же test-сете, что и финальные числа.

ЧТО СЧИТАЕМ
----------
Для каждой метрики (overall_f1, emo_mf1, ah_mf1, ...) и каждой пары экспериментов:
  - парные разности d_i = metric(B, seed_i) - metric(A, seed_i)
  - Wilcoxon signed-rank (непараметрический, по умолчанию)
  - парный bootstrap 95% CI на средней разности (надёжнее t при малом n)
  - парный t-тест (для справки; на n<8 ему не верим, но печатаем)
  - Cohen's d_z (величина эффекта для парных данных)

Печатает таблицу: mean Δ, 95% CI, p-value, вердикт significant/ns.

ИСПОЛЬЗОВАНИЕ
-------------
    python -m training.significance \
        --pair E3_fusion_no_ssl E4_fusion_ssl \
        --pair E4_fusion_ssl   E9_fusion_ssl_unfreeze \
        --metrics overall_f1 emo_mf1 ah_mf1

Если --pair не задан, по умолчанию сравнивает E3 vs E4 (вклад SSL).
Список доступных run-ов:  python -m training.significance --list
"""
import argparse
import sys
import numpy as np

try:
    import mlflow
    from mlflow.tracking import MlflowClient
except ImportError:
    sys.exit("[significance] mlflow не установлен: pip install mlflow")

try:
    from scipy import stats as _scipy_stats
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from utils.tracking import TRACKING_URI, EXPERIMENT_NAME

METRICS_DEFAULT = ["overall_f1", "emo_mf1", "emo_uar", "ah_mf1", "ah_uar"]
ALPHA = 0.05
N_BOOT = 10000
RNG_SEED = 0


# ── загрузка test-метрик из MLflow ────────────────────────────────────────────
def _client():
    mlflow.set_tracking_uri(TRACKING_URI)
    return MlflowClient()


def load_runs(client):
    """
    Возвращает: dict[run_name] -> dict с 'seed' и test-метриками (без префикса test_).
    Если у эксперимента несколько run-ов с одним именем, берём последний по времени.
    """
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        sys.exit(f"[significance] эксперимент '{EXPERIMENT_NAME}' не найден в {TRACKING_URI}")

    runs = client.search_runs([exp.experiment_id], order_by=["attributes.start_time ASC"])
    out = {}
    for r in runs:
        name = r.data.tags.get("mlflow.runName") or r.info.run_name
        if not name:
            continue
        metrics = {k[len("test_"):]: v for k, v in r.data.metrics.items()
                   if k.startswith("test_")}
        if not metrics:
            continue  # run без test-метрик (например упавший) — пропускаем
        seed = r.data.params.get("seed")
        out[name] = {"seed": (int(seed) if seed is not None else None),
                     "metrics": metrics, "run_id": r.info.run_id}
    return out


def _seed_of(run_name, info):
    """seed из param-а; если его нет — пытаемся выдрать из имени '..._seed<N>'."""
    if info["seed"] is not None:
        return info["seed"]
    if "_seed" in run_name:
        tail = run_name.split("_seed")[-1]
        digits = "".join(c for c in tail if c.isdigit())
        if digits:
            return int(digits)
    return None


def collect_pair(runs, exp_a, exp_b, metric):
    """
    Сопоставляет run-ы A и B по seed. Возвращает (seeds, a_vals, b_vals) только по
    общим сидам, где у обоих есть данная метрика.
    """
    def by_seed(prefix):
        d = {}
        for name, info in runs.items():
            # точное совпадение префикса до _seed
            if name == prefix or name.startswith(prefix + "_seed"):
                s = _seed_of(name, info)
                if s is not None and metric in info["metrics"]:
                    d[s] = info["metrics"][metric]
        return d

    a, b = by_seed(exp_a), by_seed(exp_b)
    common = sorted(set(a) & set(b))
    return common, np.array([a[s] for s in common]), np.array([b[s] for s in common])


# ── статистика ────────────────────────────────────────────────────────────────
def paired_bootstrap_ci(d, n_boot=N_BOOT, alpha=ALPHA, seed=RNG_SEED):
    """95% CI на среднюю парную разность через bootstrap по парам."""
    rng = np.random.default_rng(seed)
    n = len(d)
    if n == 0:
        return (np.nan, np.nan)
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def cohen_dz(d):
    """Величина эффекта для парных данных: mean(d)/std(d)."""
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    return float(d.mean() / sd) if sd > 0 else np.nan


def analyse(d):
    """Все статистики на векторе парных разностей d = B - A."""
    n = len(d)
    res = {
        "n": n,
        "mean_delta": float(d.mean()) if n else np.nan,
        "ci": paired_bootstrap_ci(d),
        "dz": cohen_dz(d),
        "p_wilcoxon": np.nan,
        "p_ttest": np.nan,
    }
    if n >= 2 and _HAS_SCIPY:
        # Wilcoxon падает если все разности нулевые — обрабатываем
        if np.allclose(d, 0):
            res["p_wilcoxon"] = 1.0
        else:
            try:
                res["p_wilcoxon"] = float(_scipy_stats.wilcoxon(d).pvalue)
            except ValueError:
                res["p_wilcoxon"] = np.nan
        res["p_ttest"] = float(_scipy_stats.ttest_rel(
            d, np.zeros_like(d)).pvalue) if d.std() > 0 else 1.0
    return res


# ── вывод ─────────────────────────────────────────────────────────────────────
def verdict(res):
    p = res["p_wilcoxon"]
    lo, hi = res["ci"]
    ci_excludes_0 = not (lo <= 0 <= hi) if not np.isnan(lo) else False
    if res["n"] < 3:
        return "n<3 (нет мощности)"
    if not np.isnan(p) and p < ALPHA and ci_excludes_0:
        return "ЗНАЧИМО"
    if ci_excludes_0:
        return "CI без 0 (гранично)"
    return "не значимо (ns)"


def print_pair(exp_a, exp_b, metric, seeds, a, b, res):
    d = b - a
    direction = exp_b if res["mean_delta"] >= 0 else exp_a
    print(f"\n  {metric}   ({exp_b} − {exp_a})")
    print(f"    seeds:     {seeds}  (n={res['n']})")
    print(f"    {exp_a:<26} mean={a.mean():.4f}")
    print(f"    {exp_b:<26} mean={b.mean():.4f}")
    print(f"    Δ (B−A):   {res['mean_delta']:+.4f}   в пользу: {direction}")
    lo, hi = res["ci"]
    print(f"    95% CI:    [{lo:+.4f}, {hi:+.4f}]   (bootstrap, {N_BOOT})")
    pw = res["p_wilcoxon"]; pt = res["p_ttest"]
    print(f"    Wilcoxon p={pw:.4f}" if not np.isnan(pw) else "    Wilcoxon p=n/a",
          f"| t-test p={pt:.4f}" if not np.isnan(pt) else "| t-test p=n/a",
          f"| d_z={res['dz']:.2f}" if not np.isnan(res['dz']) else "| d_z=n/a")
    print(f"    ВЕРДИКТ:   {verdict(res)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"),
                    help="Пара экспериментов A B (B сравнивается с A). Можно несколько раз.")
    ap.add_argument("--metrics", nargs="+", default=METRICS_DEFAULT)
    ap.add_argument("--list", action="store_true", help="Показать доступные run-ы и выйти")
    args = ap.parse_args()

    client = _client()
    runs = load_runs(client)

    if args.list:
        print(f"Найдено {len(runs)} run-ов с test-метриками в '{EXPERIMENT_NAME}':")
        for name in sorted(runs):
            s = runs[name]["seed"]
            ms = ", ".join(sorted(runs[name]["metrics"]))
            print(f"  {name:<34} seed={s}   [{ms}]")
        return

    pairs = args.pair or [["E3_fusion_no_ssl", "E4_fusion_ssl"]]

    if not _HAS_SCIPY:
        print("[!] scipy не установлен — p-values недоступны, печатаю только CI/bootstrap.\n"
              "    pip install scipy для полного отчёта.")

    print("=" * 72)
    print("ПАРНЫЙ АНАЛИЗ ЗНАЧИМОСТИ  (test-метрики из MLflow)")
    print(f"alpha={ALPHA} | bootstrap={N_BOOT} | парность по seed")
    print("=" * 72)

    any_output = False
    for exp_a, exp_b in pairs:
        print(f"\n{'━'*72}\nПАРА:  {exp_b}  vs  {exp_a}\n{'━'*72}")
        for metric in args.metrics:
            seeds, a, b = collect_pair(runs, exp_a, exp_b, metric)
            if len(seeds) == 0:
                print(f"\n  {metric}: нет общих сидов с этой метрикой — пропуск")
                continue
            any_output = True
            res = analyse(b - a)
            print_pair(exp_a, exp_b, metric, seeds, a, b, res)

    if not any_output:
        print("\n[!] Ни одной валидной пары не собрано. Проверь имена: --list")
    else:
        print("\n" + "=" * 72)
        print("Как читать: смотри в первую очередь на 95% CI. Если он включает 0 —")
        print("эффект неотличим от шума, сколько бы ни был мал p. На n<8 Wilcoxon и")
        print("t-тест слабомощны: широкий CI при малом n = 'данных мало', а не 'нет эффекта'.")
        print("=" * 72)


if __name__ == "__main__":
    main()
