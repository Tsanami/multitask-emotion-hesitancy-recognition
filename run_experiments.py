"""
Запуск серии экспериментов на нескольких сидах для ablation study НИРа.

ПОСЛЕДОВАТЕЛЬНО (как раньше, одна GPU):
    python run_experiments.py

ПАРАЛЛЕЛЬНО ПО КАРТАМ (8×H100): каждый (эксперимент, seed) — отдельный
процесс, привязанный к своей GPU через CUDA_VISIBLE_DEVICES. Карта, освободившись,
берёт следующее задание из общей очереди (авто-балансировка нагрузки):
    python run_experiments.py --parallel --gpus 0,1,2,3,4,5,6,7

Каждый эксперимент прогоняется на SEEDS, результаты усредняются (mean ± std).
Это нужно чтобы показать что разница E3 vs E4 статистически (не)значима.
Полный парный анализ значимости — training/significance.py (тянет test_* из MLflow).

ВНУТРЕННИЙ РЕЖИМ (не вызывать вручную):
    python run_experiments.py --run-one <EXP_NAME> <SEED>
выполняет ровно одно задание и пишет его test-метрики в JSON. Используется
параллельным диспетчером, чтобы конфиг строился из того же EXPERIMENTS-словаря,
что и в последовательном режиме (никакого расхождения настроек).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np

from configs.configs import (
    Stage2NoSSLConfig, Stage2SSLConfig, Stage2GradNormConfig, RESULTS_DIR,
)
from train_stage2 import run

# 10 сидов — нужно для мощности парного теста (на 3 сидах Wilcoxon не опускается
# ниже p=0.25 чисто из-за числа перестановок).
SEEDS = [42, 0, 123, 1, 2, 7, 13, 21, 100, 2024]

EXPERIMENTS = {
    "E3_fusion_no_ssl":        lambda: Stage2NoSSLConfig(),
    "E4_fusion_ssl":           lambda: Stage2SSLConfig(),
    "E5_fusion_gradnorm":      lambda: Stage2GradNormConfig(),
    # E9: end-to-end разморозка stage-1 энкодеров (SSL + unfreeze) — лучший ран на seed 42
    "E6_fusion_ssl_unfreeze":  lambda: Stage2SSLConfig(unfreeze_encoders=True),
}

METRICS = ["emo_mf1", "emo_uar", "emo_mwacc", "ah_mf1", "ah_uar", "ah_wf1", "overall_f1"]

# куда воркер кладёт результат одного задания (подхватывается диспетчером)
JOBOUT_DIR = os.path.join(RESULTS_DIR, "_jobout")


def _configure(name, seed):
    """Строит cfg для (name, seed) — единый источник для обоих режимов."""
    cfg = EXPERIMENTS[name]()
    cfg.output_path  = f"{RESULTS_DIR}/{name}_seed{seed}.pt"
    cfg.history_path = f"{RESULTS_DIR}/{name}_seed{seed}_history.json"
    cfg.run_name     = f"{name}_seed{seed}"
    return cfg


# ── ВНУТРЕННИЙ РЕЖИМ: одно задание ────────────────────────────────────────────
def run_one(name, seed):
    cfg = _configure(name, seed)
    print(f"\n{'#'*22} {name} | seed {seed} {'#'*22}")
    test_log = run(cfg, seed=seed)
    os.makedirs(JOBOUT_DIR, exist_ok=True)
    out = {m: float(test_log[m]) for m in METRICS}
    with open(os.path.join(JOBOUT_DIR, f"{name}_seed{seed}.json"), "w") as f:
        json.dump(out, f)
    return out


# ── ПОСЛЕДОВАТЕЛЬНЫЙ РЕЖИМ ─────────────────────────────────────────────────────
def run_multiseed_sequential(name):
    per_seed = {m: [] for m in METRICS}
    for seed in SEEDS:
        if "E3" in name and seed in [42, 0, 123, 1, 2, 7]:
            # E3_fusion_no_ssl — baseline, прогоняем только на 3 сидах, чтобы не тратить время
            continue
        out = run_one(name, seed)
        for m in METRICS:
            per_seed[m].append(out[m])
    return per_seed


# ── ПАРАЛЛЕЛЬНЫЙ РЕЖИМ: очередь заданий, N процессов на карту ─────────────────
def run_parallel(gpus, per_gpu=1):
    """
    gpus: список int. per_gpu: сколько процессов держать на каждой карте.

    Поднимает len(gpus)*per_gpu воркеров; каждый берёт задания из общей очереди и
    запускает 'python run_experiments.py --run-one <exp> <seed>' с
    CUDA_VISIBLE_DEVICES=<gpu>. Балансировка естественная: слот освободился — взял
    следующее задание.

    Зачем per_gpu>1: модель крошечная (~1 ГБ, ~7% util на H100), карта простаивает.
    Несколько процессов на карту уплотняют загрузку и сокращают общее время свипа.
    Память: per_gpu * ~1 ГБ на карту — для 80 ГБ H100 можно смело 8-16.
    """
    os.makedirs(JOBOUT_DIR, exist_ok=True)
    jobs = deque((name, seed) for name in EXPERIMENTS for seed in SEEDS)
    total = len(jobs)
    capacity = len(gpus) * per_gpu
    print(f"[parallel] заданий: {total} ({len(EXPERIMENTS)} эксп × {len(SEEDS)} сидов) "
          f"| {len(gpus)} карт(ы) × {per_gpu} проц = {capacity} параллельно: {gpus}")

    # Плоский список слотов: каждый слот привязан к конкретной карте.
    # slots[i] = None (свободен) либо (gpu, Popen, (name,seed), start_time)
    slot_gpu = [g for g in gpus for _ in range(per_gpu)]  # карта каждого слота
    slots = [None] * capacity
    done = 0
    failed = []
    t0 = time.time()

    def launch(gpu, job):
        name, seed = job
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        # дочерний процесс видит ровно одну карту → внутри это всегда cuda:0
        log_path = os.path.join(JOBOUT_DIR, f"{name}_seed{seed}.log")
        logf = open(log_path, "w")
        p = subprocess.Popen(
            [sys.executable, os.path.abspath(__file__), "--run-one", name, str(seed)],
            env=env, stdout=logf, stderr=subprocess.STDOUT,
        )
        p._logf = logf  # держим хэндл, закроем по завершении
        print(f"[gpu {gpu}] ▶ {name} seed {seed}  (лог: {log_path})")
        return p

    while done < total:
        # раздаём свободным слотам задания
        for i in range(capacity):
            if slots[i] is None and jobs:
                job = jobs.popleft()
                g = slot_gpu[i]
                slots[i] = (g, launch(g, job), job, time.time())

        # опрашиваем активные слоты
        time.sleep(2.0)
        for i in range(capacity):
            slot = slots[i]
            if slot is None:
                continue
            g, p, job, started = slot
            ret = p.poll()
            if ret is None:
                continue  # ещё работает
            # завершилось
            p._logf.close()
            name, seed = job
            dt = time.time() - started
            if ret == 0:
                done += 1
                print(f"[gpu {g}] ✓ {name} seed {seed}  ({dt:.0f}s)  [{done}/{total}]")
            else:
                done += 1
                failed.append(job)
                print(f"[gpu {g}] ✗ {name} seed {seed}  RET={ret}  "
                      f"(см. лог) [{done}/{total}]")
            slots[i] = None

    elapsed = time.time() - t0
    print(f"\n[parallel] готово за {elapsed/60:.1f} мин. "
          f"Успешно: {total-len(failed)}/{total}.")
    if failed:
        print(f"[parallel] УПАЛИ: {failed}")
        print(f"[parallel] логи в {JOBOUT_DIR}/<exp>_seed<seed>.log")

    # собираем результаты из jobout-файлов
    all_results = {}
    for name in EXPERIMENTS:
        per_seed = {m: [] for m in METRICS}
        for seed in SEEDS:
            fp = os.path.join(JOBOUT_DIR, f"{name}_seed{seed}.json")
            if not os.path.exists(fp):
                continue  # упавшее задание — пропускаем, не валим всю агрегацию
            with open(fp) as f:
                out = json.load(f)
            for m in METRICS:
                per_seed[m].append(out[m])
        all_results[name] = per_seed
    return all_results


# ── агрегация + печать ─────────────────────────────────────────────────────────
def aggregate_and_report(all_results):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n\n" + "=" * 78)
    print(f"ABLATION STUDY — mean ± std по сидам {SEEDS}")
    print("=" * 78)
    header = f"{'Experiment':<24}" + "".join(f"{m:>16}" for m in METRICS)
    print(header); print("-" * len(header))
    for name, res in all_results.items():
        row = f"{name:<24}"
        for m in METRICS:
            arr = np.array(res[m]) if res[m] else np.array([np.nan])
            row += f"{arr.mean():>8.4f}±{arr.std():.3f}"
        print(row)

    if "E3_fusion_no_ssl" in all_results and "E4_fusion_ssl" in all_results:
        print("\n" + "=" * 50)
        print("ВКЛАД SSL (E4 − E3), по средним:")
        print("=" * 50)
        for m in METRICS:
            e3 = np.array(all_results["E3_fusion_no_ssl"][m] or [np.nan]).mean()
            e4 = np.array(all_results["E4_fusion_ssl"][m] or [np.nan]).mean()
            delta = e4 - e3
            print(f"  {m:<14} {'+' if delta>=0 else ''}{delta:.4f}")

    print("\n[i] Это только средние. Парная значимость (рекомендуется):")
    print("    python -m training.significance --pair E3_fusion_no_ssl E4_fusion_ssl \\")
    print("                                    --pair E4_fusion_ssl E9_fusion_ssl_unfreeze")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", action="store_true",
                    help="Параллельный запуск по нескольким GPU")
    ap.add_argument("--gpus", type=str, default="0",
                    help="Список GPU через запятую, напр. 0,1,2,3,4,5,6,7")
    ap.add_argument("--per-gpu", type=int, default=1,
                    help="Сколько процессов на каждую карту (модель мелкая — "
                         "смело 8-16). Всего параллельно: len(gpus)*per_gpu")
    ap.add_argument("--run-one", nargs=2, metavar=("EXP", "SEED"),
                    help="ВНУТРЕННИЙ: выполнить одно задание и записать результат")
    args = ap.parse_args()

    # внутренний режим воркера
    if args.run_one:
        name, seed = args.run_one
        if name not in EXPERIMENTS:
            sys.exit(f"неизвестный эксперимент: {name} (есть: {list(EXPERIMENTS)})")
        run_one(name, int(seed))
        return

    if args.parallel:
        gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
        if not gpus:
            sys.exit("--parallel требует --gpus, напр. --gpus 0,1,2,3")
        all_results = run_parallel(gpus, per_gpu=max(1, args.per_gpu))
    else:
        all_results = {name: run_multiseed_sequential(name) for name in EXPERIMENTS}

    aggregate_and_report(all_results)


if __name__ == "__main__":
    main()
