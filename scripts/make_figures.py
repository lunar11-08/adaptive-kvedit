"""Qualitative grid: rows = images, cols = source + each method (for one mask-noise setting).

python scripts/make_figures.py --run runs/ablation --noise dilate --methods kvedit fused_sched --n 8
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--noise", default="none")
    ap.add_argument("--methods", nargs="+", default=["kvedit", "fused_sched"])
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out")
    args = ap.parse_args()

    run = Path(args.run)
    names = [f"{m}__{args.noise}" if m != "attn" else "attn__none" for m in args.methods]
    dirs = [run / k for k in args.keys] if args.keys else \
        [d for d in sorted(run.iterdir()) if d.is_dir() and all((d / f"{n}.png").exists() for n in names)]
    dirs = dirs[:args.n]
    if not dirs:
        raise SystemExit("no images with all requested outputs")

    cols = len(names) + 1
    fig, axes = plt.subplots(len(dirs), cols, figsize=(2.4 * cols, 2.4 * len(dirs)), squeeze=False)
    for r, d in enumerate(dirs):
        src = np.array(Image.open(d / "source.png"))
        npz = d / f"{names[0]}_masks.npz"
        axes[r, 0].imshow(src)
        if npz.exists():
            axes[r, 0].contour(np.load(npz)["user_px"], levels=[0.5], colors="lime", linewidths=1)
        for c, n in enumerate(names, start=1):
            axes[r, c].imshow(Image.open(d / f"{n}.png"))
        for c, title in enumerate(["source + user mask", *args.methods]):
            if r == 0:
                axes[r, c].set_title(title, fontsize=9)
        for ax in axes[r]:
            ax.axis("off")
    fig.tight_layout()
    out = Path(args.out or run / f"figure_{args.noise}.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
