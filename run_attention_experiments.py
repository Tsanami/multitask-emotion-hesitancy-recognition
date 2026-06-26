"""
Attention-ablation

Stage-1 энкодеры (ER на MOSEI, AHR на BAH) обучаются с разными механизмами
внимания из статей (models/attention.py):
    softmax — baseline (полный self-attention)
    zeros   — Zero-Sum Linear Attention (zeros.pdf)
    mhla    — Multi-Head Linear Attention (MHLA.pdf)
    elsa    — Exact Linear-Scan Attention (ELSA.pdf)

Плюс задача fusion_grad_norm: stage-2 fusion + GradNorm поверх энкодеров с
выбранным attn_type. САМ fusion использует дефолтное внимание — attn_type меняет
только stage-1 энкодеры под ним. Так меряется сквозной эффект механизма внимания.

Метрики (как в обычном run_experiments):
    emotion          → mf1, uar, mwacc
    ah               → mf1, uar, wf1
    fusion_grad_norm → emo_mf1, emo_uar, emo_mwacc, ah_mf1, ah_uar, ah_wf1, overall_f1
Веса классов эмоций включены (flag_emo_weight=True).

    python run_attention_experiments.py
    python run_attention_experiments.py --tasks emotion --attn softmax zeros mhla elsa

Затем — парная значимость по test_* из MLflow:
    python -m training.significance \\
        --pair attn_emotion_softmax attn_emotion_zeros --metrics emo_mf1 emo_uar
"""
import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque

import numpy as np

from configs.configs import Stage1Config, Stage2GradNormConfig, RESULTS_DIR
from train_stage1 import run as run_stage1
from train_stage2 import run as run_stage2

# Один сид — как и оставлено; для значимости на 10 сидах расширить список.
SEEDS = [42]

ATTN_TYPES = ["softmax", "zeros", "mhla", "elsa"]
TASKS = ["emotion", "ah", "fusion_grad_norm"]
STAGE1_TASKS = ("emotion", "ah")

FLAG_EMO_WEIGHT = True

METRICS = {
    "emotion":          ["mf1", "uar", "mwacc"],
    "ah":               ["mf1", "uar", "wf1"],
    "fusion_grad_norm": ["emo_mf1", "emo_uar", "emo_mwacc",
                         "ah_mf1", "ah_uar", "ah_wf1", "overall_f1"],
}
PRIMARY = {"emotion": "mf1", "ah": "mf1", "fusion_grad_norm": "overall_f1"}

JOBOUT_DIR = os.path.join(RESULTS_DIR, "_jobout_attn")


def _configure(task, attn, seed):
    """Единый источник конфига для обоих режимов."""
    if task in STAGE1_TASKS:
        cfg = Stage1Config()
        cfg.attn_type = attn
        cfg.flag_emo_weight = FLAG_EMO_WEIGHT
        # отдельные чекпойнты на (attn, seed), чтобы не перетирать stage-1 default
        cfg.emo_output_path = f"{RESULTS_DIR}/attn_{attn}_emotion_seed{seed}.pt"
        cfg.ah_output_path  = f"{RESULTS_DIR}/attn_{attn}_ah_seed{seed}.pt"
        cfg.run_name = f"attn_{task}_{attn}_seed{seed}"
        return cfg

    if task == "fusion_grad_norm":
        cfg = Stage2GradNormConfig()
        cfg.attn_type = attn                  # энкодеры под fusion — с этим вниманием
        cfg.flag_emo_weight = FLAG_EMO_WEIGHT
        # грузим именно attn-специфичные stage-1 чекпойнты
        cfg.emo_model_path = f"{RESULTS_DIR}/attn_{attn}_emotion_seed{seed}.pt"
        cfg.ah_model_path  = f"{RESULTS_DIR}/attn_{attn}_ah_seed{seed}.pt"
        cfg.output_path    = f"{RESULTS_DIR}/attn_{attn}_fusion_gradnorm_seed{seed}.pt"
        cfg.history_path   = f"{RESULTS_DIR}/attn_{attn}_fusion_gradnorm_seed{seed}_history.json"
        cfg.run_name = f"attn_fusion_grad_norm_{attn}_seed{seed}"
        return cfg

    raise ValueError(f"неизвестная задача: {task}")


# ── один прогон ───────────────────────────────────────────────────────────────
def run_one(task, attn, seed):
    cfg = _configure(task, attn, seed)
    print(f"\n{'#'*18} {task} | attn={attn} | seed {seed} {'#'*18}")

    if task in STAGE1_TASKS:
        test_log = run_stage1(cfg, task, seed)
    else:  # fusion_grad_norm
        for p in (cfg.emo_model_path, cfg.ah_model_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Нет stage-1 чекпойнта {p}. Сначала прогоните emotion и ah "
                    f"для attn={attn} (в параллельном режиме это фаза 1)."
                )
        test_log = run_stage2(cfg, seed=seed)

    os.makedirs(JOBOUT_DIR, exist_ok=True)
    out = {m: float(test_log[m]) for m in METRICS[task]}
    with open(os.path.join(JOBOUT_DIR, f"{task}_{attn}_seed{seed}.json"), "w") as f:
        json.dump(out, f)
    return out


# ── последовательный режим (attn-major: stage-1 раньше fusion для того же attn) ─
def run_sequential(tasks, attns):
    all_results = {}
    for attn in attns:
        for task in tasks:
            per_seed = {m: [] for m in METRICS[task]}
            for seed in SEEDS:
                out = run_one(task, attn, seed)
                for m in METRICS[task]:
                    per_seed[m].append(out[m])
            all_results[f"{task}_{attn}"] = per_seed
    return all_results



# ── агрегация + печать ────────────────────────────────────────────────────────
def aggregate_and_report(all_results, tasks, attns):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(f"{RESULTS_DIR}/attention_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n\n" + "=" * 70)
    print(f"ATTENTION ABLATION — mean ± std по сидам {SEEDS}")
    print("=" * 70)
    for task in tasks:
        metrics = METRICS[task]
        primary = PRIMARY[task]
        print(f"\n--- task: {task} ---")
        print(f"{'attn':<10}" + "".join(f"{m:>16}" for m in metrics))
        base = None
        for attn in attns:
            res = all_results.get(f"{task}_{attn}", {})
            row = f"{attn:<10}"
            prim_mean = None
            for m in metrics:
                arr = np.array(res[m]) if res.get(m) else np.array([np.nan])
                row += f"{arr.mean():>9.4f}±{arr.std():.3f}"
                if m == primary:
                    prim_mean = arr.mean()
            if attn == "softmax":
                base = prim_mean
            delta = (f"  Δ{primary}={prim_mean-base:+.4f}"
                     if (base is not None and prim_mean is not None) else "")
            print(row + delta)

    print("\n[i] Δ — относительно softmax-внутри-ablation (like-for-like).")
    print("    Парная значимость: python -m training.significance "
          "--pair attn_emotion_softmax attn_emotion_zeros --metrics emo_mf1")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    ap.add_argument("--attn", nargs="+", choices=ATTN_TYPES, default=ATTN_TYPES)
    ap.add_argument("--run-one", nargs=3, metavar=("TASK", "ATTN", "SEED"),
                    help="ВНУТРЕННИЙ: одно задание")
    args = ap.parse_args()

    if args.run_one:
        task, attn, seed = args.run_one
        run_one(task, attn, int(seed))
        return

    all_results = run_sequential(args.tasks, args.attn)

    aggregate_and_report(all_results, args.tasks, args.attn)


if __name__ == "__main__":
    main()
