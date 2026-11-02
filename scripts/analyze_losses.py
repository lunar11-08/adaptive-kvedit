"""Where does ours lose to KV-Edit? Per mask type, per category, and per image (worst first).

python scripts/analyze_losses.py --kvedit runs/dev/metrics.csv --ours runs/dev_tuned/metrics.csv [--noise none]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from adaptive_kvedit.data.pie_bench import CATEGORIES

METRICS = {"psnr": 1, "lpips_1e3": -1, "clip_edited": 1}     # +1: higher is better


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvedit", required=True)
    ap.add_argument("--ours", required=True)
    ap.add_argument("--ours-method", default="fused_sched")
    ap.add_argument("--noise", default="none", help="mask type for the per-image list")
    ap.add_argument("--worst", type=int, default=10)
    args = ap.parse_args()

    kv = pd.read_csv(args.kvedit).query("method == 'kvedit'")
    us = pd.read_csv(args.ours).query("method == @args.ours_method")
    m = kv.merge(us, on=["key", "noise"], suffixes=("_kv", "_us"))
    m["category"] = m["category_kv"].astype(str).map(CATEGORIES)
    for k, sign in METRICS.items():
        m[f"d_{k}"] = m[f"{k}_us"] - m[f"{k}_kv"]
        m[f"win_{k}"] = (sign * m[f"d_{k}"]) > 0
    m["mask_kept"] = m["final_tokens_us"] / m["user_tokens_us"]

    pd.set_option("display.width", 200)
    print("== per mask type: mean difference (ours - KV-Edit) and win rate ==")
    agg = {f"d_{k}": "mean" for k in METRICS} | {f"win_{k}": "mean" for k in METRICS} | {"key": "count"}
    print(m.groupby("noise").agg(agg).round(2).rename(columns={"key": "n"}).to_string())

    print(f"\n== per category, mask type '{args.noise}' ==")
    sub = m[m["noise"] == args.noise]
    print(sub.groupby("category").agg(agg).round(2).rename(columns={"key": "n"}).to_string())

    print(f"\n== {args.worst} worst images by PSNR, mask type '{args.noise}' ==")
    cols = ["key", "category", "psnr_kv", "psnr_us", "d_psnr", "d_lpips_1e3", "d_clip_edited", "mask_kept"]
    print(sub.sort_values("d_psnr")[cols].head(args.worst).round(2).to_string(index=False))
    print("\nmask_kept = our final mask size / mask given (1.0 = same size).")


if __name__ == "__main__":
    main()
