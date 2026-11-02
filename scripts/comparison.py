"""Results table from one or more metrics.csv files (markdown to stdout and <first run>/results.md).

python scripts/comparison.py runs/main/metrics.csv [runs/ablation/metrics.csv] [--by-category]
Reported numbers for older baselines go in configs/reported_baselines.csv (copied from the KV-Edit paper).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

COLS = ["psnr", "lpips_1e3", "mse_1e4", "ssim_1e2", "clip_whole", "clip_edited"]
ARROWS = {"psnr": "↑", "lpips_1e3": "↓", "mse_1e4": "↓", "ssim_1e2": "↑", "clip_whole": "↑", "clip_edited": "↑"}
REPORTED = Path(__file__).resolve().parent.parent / "configs" / "reported_baselines.csv"


def table(df: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    cols = [c for c in COLS if c in df]
    g = df.groupby(by)[cols].mean().round(2)
    g.insert(0, "n", df.groupby(by).size())
    return g.rename(columns={c: f"{c} {ARROWS[c]}" for c in cols})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--by-category", action="store_true")
    args = ap.parse_args()

    df = pd.concat([pd.read_csv(p).assign(run=Path(p).parent.name) for p in args.csv])
    parts = ["## Method x mask noise\n", table(df, ["run", "noise", "method"]).to_markdown(), ""]
    if args.by_category:
        parts += ["## Per PIE-Bench category (noise=none)\n",
                  table(df[df.noise == "none"], ["category", "method"]).to_markdown(), ""]
    if REPORTED.exists() and REPORTED.stat().st_size > 0:
        rep = pd.read_csv(REPORTED).dropna(how="all")
        if len(rep):
            parts += ["## Reported in the KV-Edit paper (full PIE-Bench, not rerun)\n",
                      rep.to_markdown(index=False), ""]
    md = "\n".join(parts)
    print(md)
    (Path(args.csv[0]).parent / "results.md").write_text(md)


if __name__ == "__main__":
    sys.exit(main())
