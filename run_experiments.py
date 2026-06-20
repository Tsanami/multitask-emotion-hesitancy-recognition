"""
Запуск серии экспериментов на нескольких сидах для ablation study НИРа.

    python run_experiments.py

Каждый эксперимент прогоняется на SEEDS, результаты усредняются (mean ± std).
Это нужно чтобы показать что разница E3 vs E4 статистически значима.
"""
import json
import numpy as np

from configs.configs import Stage2NoSSLConfig, Stage2SSLConfig, Stage2GradNormConfig
from train_stage2 import run

SEEDS      = [42, 0, 123]
# MOSEI_PATH = "data/mosei"
# BAH_PATH   = "data/bah"

# EMB = dict(
#     mosei_train_emb      = "data/mosei/embeddings_cache/CMU-MOSEI_train_bge-small_embeddings.pkl",
#     mosei_validation_emb = "data/mosei/embeddings_cache/CMU-MOSEI_validation_bge-small_embeddings.pkl",
#     mosei_test_emb       = "data/mosei/embeddings_cache/CMU-MOSEI_test_bge-small_embeddings.pkl",
#     bah_train_emb        = "data/bah/embeddings_cache/BAH_train_bge-small_embeddings.pkl",
#     bah_val_emb          = "data/bah/embeddings_cache/BAH_val_bge-small_embeddings.pkl",
#     bah_test_emb         = "data/bah/embeddings_cache/BAH_test_bge-small_embeddings.pkl",
# )

EXPERIMENTS = {
    "E3_fusion_no_ssl": lambda: Stage2NoSSLConfig(),
    "E4_fusion_ssl": lambda: Stage2SSLConfig(),
    "E5_fusion_ssl_thr08": lambda: Stage2SSLConfig(ssl_conf_thr_emo=0.8, ssl_conf_thr_ah=0.8),
    # E6/E7: вариация силы SSL (lambda_ssl) — чувствительность к весу псевдоразметки
    "E6_fusion_ssl_lam01": lambda: Stage2SSLConfig(lambda_ssl=0.1),
    "E7_fusion_ssl_lam03": lambda: Stage2SSLConfig(lambda_ssl=0.3),
    # E8: Fusion + SSL + GradNorm (динамическая балансировка)
    "E8_fusion_gradnorm": lambda: Stage2GradNormConfig()
}

METRICS = ["emo_mf1", "emo_uar", "ah_mf1", "ah_uar", "overall_f1"]


def run_multiseed(name, cfg_factory):
    per_seed = {m: [] for m in METRICS}
    for seed in SEEDS:
        cfg = cfg_factory()
        cfg.output_path  = f"best_models/{name}_seed{seed}.pt"
        cfg.history_path = f"best_models/{name}_seed{seed}_history.json"
        print(f"\n{'#'*22} {name} | seed {seed} {'#'*22}")
        test_log = run(cfg, seed=seed)
        for m in METRICS:
            per_seed[m].append(test_log[m])
    return per_seed


def main():
    all_results = {name: run_multiseed(name, f) for name, f in EXPERIMENTS.items()}

    with open("best_models/ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)

    print("\n\n" + "=" * 78)
    print(f"ABLATION STUDY — mean ± std по сидам {SEEDS}")
    print("=" * 78)
    header = f"{'Experiment':<22}" + "".join(f"{m:>16}" for m in METRICS)
    print(header); print("-" * len(header))
    for name, res in all_results.items():
        row = f"{name:<22}"
        for m in METRICS:
            arr = np.array(res[m])
            row += f"{arr.mean():>8.4f}±{arr.std():.3f}"
        print(row)

    if "E3_fusion_no_ssl" in all_results and "E4_fusion_ssl" in all_results:
        print("\n" + "=" * 50)
        print("ВКЛАД SSL (E4 − E3):")
        print("=" * 50)
        for m in METRICS:
            e3 = np.array(all_results["E3_fusion_no_ssl"][m]).mean()
            e4 = np.array(all_results["E4_fusion_ssl"][m]).mean()
            delta = e4 - e3
            print(f"  {m:<14} {'+' if delta>=0 else ''}{delta:.4f}")



if __name__ == "__main__":
    main()
