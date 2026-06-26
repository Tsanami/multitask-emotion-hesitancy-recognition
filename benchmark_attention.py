"""
Бенчмарк сменных механизмов внимания (скорость + пиковая память) на формах,
которые реально встречаются в этом проекте: stage-1 EMO (dim=256, heads=4) и
stage-1 AH (dim=512, heads=8). Синтетические тензоры,
работает и на CPU (тогда только латентность; память меряется лишь на CUDA).

    python benchmark_attention.py
    python benchmark_attention.py --seq 16 32 64 128 256 --batch 32 --iters 50

Зачем: linear/mhla обещают sub-quadratic рост по T, elsa — экономию памяти при
softmax-математике. Тексты MOSEI/BAH короткие (T обычно <60 токенов), поэтому
бенчмарк показывает, есть ли вообще выигрыш на наших длинах, или квадратичность
softmax здесь несущественна.
"""
import argparse
import json
import os
import time

import torch

from models.attention import make_attention, ATTENTION_TYPES
from configs.configs import RESULTS_DIR


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench_one(attn, dim, heads, B, T, device, iters, warmup=10):
    m = make_attention(attn, dim, num_heads=heads).to(device).eval()
    x = torch.randn(B, T, dim, device=device)

    for _ in range(warmup):
        m(x)
    _sync(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    for _ in range(iters):
        m(x)
    _sync(device)
    dt = (time.perf_counter() - t0) / iters * 1000.0  # мс/итерация

    peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
               if device.type == "cuda" else float("nan"))
    n_params = sum(p.numel() for p in m.parameters()) / 1e6
    return dt, peak_mb, n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--cpu", action="store_true", help="принудительно CPU")
    args = ap.parse_args()

    device = torch.device("cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu")
    configs = [("EMO", 256, 4), ("AH", 512, 8)]
    results = {}

    print(f"Устройство: {device} | batch={args.batch} | iters={args.iters}")
    print(f"Память меряется только на CUDA (на CPU — NaN).\n")

    for cfg_name, dim, heads in configs:
        print(f"{'='*72}\n{cfg_name}-конфиг: dim={dim}, heads={heads}\n{'='*72}")
        header = f"{'T':>5} | " + " | ".join(f"{a:>22}" for a in ATTENTION_TYPES)
        print(header)
        print(f"{'':>5} | " + " | ".join(f"{'мс  /  пик МБ':>22}" for _ in ATTENTION_TYPES))
        print("-" * len(header))
        for T in args.seq:
            row = f"{T:>5} | "
            cells = []
            for attn in ATTENTION_TYPES:
                dt, mem, npar = bench_one(attn, dim, heads, args.batch, T, device, args.iters)
                results[f"{cfg_name}_{attn}_T{T}"] = {
                    "ms": dt, "peak_mb": mem, "params_M": npar, "dim": dim,
                    "heads": heads, "batch": args.batch, "T": T,
                }
                cells.append(f"{dt:8.3f} / {mem:8.1f}")
            print(row + " | ".join(f"{c:>22}" for c in cells))
        print()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "attention_benchmark.json")
    with open(out, "w") as f:
        json.dump({"device": str(device), "results": results}, f, indent=2)
    print(f"JSON сохранён: {out}")

    # сводка масштабирования: во сколько раз растёт латентность T:16→max
    print(f"\n{'='*72}\nМАСШТАБИРОВАНИЕ латентности (T={args.seq[0]} → T={args.seq[-1]})\n{'='*72}")
    for cfg_name, dim, heads in configs:
        print(f"{cfg_name}:")
        for attn in ATTENTION_TYPES:
            a = results[f"{cfg_name}_{attn}_T{args.seq[0]}"]["ms"]
            b = results[f"{cfg_name}_{attn}_T{args.seq[-1]}"]["ms"]
            factor = b / a if a > 0 else float("nan")
            tfac = args.seq[-1] / args.seq[0]
            print(f"  {attn:<8} ×{factor:5.1f}   (T вырос ×{tfac:.0f}; "
                  f"квадратично ожидалось ×{tfac**2:.0f})")


if __name__ == "__main__":
    main()
