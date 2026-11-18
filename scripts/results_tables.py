"""Results and ablation tables (markdown) from metrics.csv files.

python scripts/results_tables.py \
    --variant "KV-Edit=runs/test/metrics.csv:kvedit" \
    --variant "Fused + schedule (v1)=runs/test/metrics.csv:fused_sched" \
    --variant "+ hole fill inside mask=runs/test_v2/metrics.csv:fused_sched" \
    --variant "+ removal rule (final)=runs/test_v3_removal/ours_merged.csv:fused_sched" \
    --masks "v1=runs/test" --masks "final=runs/test_final" --out runs/test_final/results.md

The first --variant is the baseline, the last is "ours" in the main table. A variant with only
noise=none (attention-only, which ignores the user mask) is reused for every mask condition.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

NOISES = [("none", "clean"), ("erode", "too small"), ("dilate", "too big"), ("box", "box")]
METRICS = [("psnr", "PSNR ↑"), ("lpips_1e3", "LPIPS×1e3 ↓"), ("mse_1e4", "MSE×1e4 ↓"), ("ssim_1e2", "SSIM×1e2 ↑"),
           ("clip_whole", "CLIP whole ↑"), ("clip_edited", "CLIP edited ↑")]
TOL = 1e-6                           # per-image differences below this count as ties (CSV round-off)
CATS = {"0": "random", "1": "change object", "2": "add object", "3": "delete object", "4": "change content",
        "5": "change pose", "6": "change color", "7": "change material", "8": "change background",
        "9": "change style"}


def load(spec: str) -> tuple[str, pd.DataFrame]:
    label, rest = spec.split("=", 1)
    path, method = rest.rsplit(":", 1)
    df = pd.read_csv(path, dtype={"key": str, "category": str})
    df = df[df.method == method].copy()
    if df.empty:
        raise SystemExit(f"{label}: no rows for method {method!r} in {path}")
    df["key"] = df.key.str.zfill(12)
    if set(df.noise) == {"none"}:                       # mask-independent: same output for every mask
        df = pd.concat([df.assign(noise=nz) for nz, _ in NOISES])
        label += " †"
    return label, df[["key", "category", "noise", *[m for m, _ in METRICS]]]


def paired(a: pd.DataFrame, b: pd.DataFrame, nz: str) -> pd.DataFrame:
    return a[a.noise == nz].merge(b[b.noise == nz], on=["key", "category", "noise"], suffixes=("_a", "_b"))


def fmt(x, d=2):
    return "—" if pd.isna(x) else f"{x:.{d}f}"


def main_table(base, ours, bl, ol) -> str:
    rows = []
    for nz, name in NOISES:
        m = paired(base, ours, nz)
        bg = m.dropna(subset=["psnr_a"])
        dl = (bg.lpips_1e3_b - bg.lpips_1e3_a).where(lambda x: x.abs() > TOL, 0)
        dp = (bg.psnr_b - bg.psnr_a).where(lambda x: x.abs() > TOL, 0)
        p = wilcoxon(dp[dp != 0]).pvalue if (dp != 0).sum() > 5 else np.nan
        cells = [name]
        for col, h in METRICS:
            a, b = m[f"{col}_a"].mean(), m[f"{col}_b"].mean()
            if pd.isna(a):
                cells.append("—")
                continue
            better = (b > a) if h.endswith("↑") else (b < a)
            cells.append(f"{fmt(a)} → **{fmt(b)}**" if better and round(a, 2) != round(b, 2) else f"{fmt(a)} → {fmt(b)}")
        cells += [f"{(dl < 0).sum()} / {(dl == 0).sum()} / {(dl > 0).sum()}", f"{p:.0e}" if not pd.isna(p) else "—"]
        rows.append(cells)
    head = ["Mask given", *[h for _, h in METRICS], "better / same / worse", "p (PSNR)"]
    note = (f"{bl} → {ol}; bold = better. Background metrics on the {len(bg)} non-style images; CLIP on all. "
            "better / same / worse: images by LPIPS. p: Wilcoxon signed-rank test on per-image PSNR.")
    return md(head, rows) + "\n" + note


def ablation_table(variants, base) -> str:
    head = ["Variant", *[f"{n} PSNR / LPIPS" for _, n in NOISES], "CLIP edited (all)", "CLIP edited (style)",
            "worst image ΔPSNR", "edits −1 dB or worse"]
    rows = []
    for label, df in variants:
        cells = [label]
        for nz, _ in NOISES:
            x = df[df.noise == nz]
            cells.append(f"{fmt(x.psnr.mean())} / {fmt(x.lpips_1e3.mean())}")
        cells.append(fmt(df.clip_edited.mean()))
        cells.append(fmt(df[df.category == "9"].clip_edited.mean()))
        d = pd.concat([paired(base, df, nz) for nz, _ in NOISES]).dropna(subset=["psnr_a"])
        dp = d.psnr_b - d.psnr_a
        cells += [fmt(dp.min()), str(int((dp <= -1).sum()))]
        rows.append(cells)
    return (md(head, rows) + "\nMeans over all test images per mask condition (PSNR/LPIPS on non-style images). "
            "Worst image and −1 dB counts are per-edit PSNR changes vs the first row. "
            "† ignores the user mask, so one result is used for every mask condition.")


def category_table(base, ours, nz="box") -> str:
    m = paired(base, ours, nz)
    rows = []
    for c, g in m.groupby("category"):
        rows.append([CATS.get(c, c), str(len(g)), f"{fmt(g.psnr_a.mean())} → {fmt(g.psnr_b.mean())}",
                     f"{fmt(g.lpips_1e3_a.mean())} → {fmt(g.lpips_1e3_b.mean())}",
                     f"{fmt(g.clip_edited_a.mean())} → {fmt(g.clip_edited_b.mean())}"])
    return md(["Category", "n", "PSNR", "LPIPS×1e3", "CLIP edited"], rows)


def outside_table(specs) -> str:
    rows = []
    for spec in specs:
        label, run = spec.split("=", 1)
        n = out = 0
        imgs = set()
        for f in sorted(Path(run).glob("*/fused_sched__*_masks.npz")):
            z = np.load(f)
            u = z["user"] > 0.5
            widest = z["loose"] if "loose" in z else (z["final"] if "final" in z else z["user"])
            n += 1
            if ((widest > 0.5) & ~u).any():
                out += 1
                imgs.add(f.parent.name)
        rows.append([label, run, str(n), str(out), str(len(imgs))])
    return md(["Version", "Run", "Edits", "Edits with blocks outside the user mask", "Images"], rows)


def md(head, rows) -> str:
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", required=True, help="LABEL=metrics.csv:method (baseline first)")
    ap.add_argument("--masks", action="append", default=[], help="LABEL=run_dir: count blocks outside the user mask")
    ap.add_argument("--out")
    args = ap.parse_args()

    variants = [load(v) for v in args.variant]
    (bl, base), (ol, ours) = variants[0], variants[-1]
    parts = ["## Main result\n", main_table(base, ours, bl, ol),
             "## Ablation\n", ablation_table(variants, base),
             "## Per category (box mask)\n", category_table(base, ours)]
    if args.masks:
        parts += ["## Hole-filling check\n", outside_table(args.masks)]
    text = "\n".join(parts)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


if __name__ == "__main__":
    main()
