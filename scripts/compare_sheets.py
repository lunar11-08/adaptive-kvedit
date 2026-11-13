"""Side-by-side sheets: original | mask given | KV-Edit | ours | our corrected mask.

python scripts/compare_sheets.py --kvedit runs/dev --ours runs/dev_tuned --noise box --out runs/dev_tuned/sheets
"""
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from adaptive_kvedit.data.pie_bench import CATEGORIES, load_pie_bench

S = 300                                  # tile size
FONT = Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans.ttf"
NOISE_NAME = {"none": "clean mask", "dilate": "mask too big", "erode": "mask too small", "box": "box mask"}


def font(n):
    return ImageFont.truetype(str(FONT), n)


def tint(img, mask, color, a=0.45):
    m = np.array(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(img.shape[1::-1]))[..., None] / 255
    return (img * (1 - a * m) + np.array(color) * a * m).astype(np.uint8)


def outline(img, mask, color, w=3):
    from scipy.ndimage import binary_dilation
    m = np.array(Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(img.shape[1::-1])) > 127
    edge = binary_dilation(m, iterations=w) & ~m
    out = img.copy()
    out[edge] = color
    return out


def tile(img):
    return Image.fromarray(img).resize((S, S))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvedit", required=True, help="run dir with kvedit__<noise>.png")
    ap.add_argument("--ours", required=True, help="run dir with fused_sched__<noise>.png")
    ap.add_argument("--noise", nargs="+", default=["box", "dilate", "none"])
    ap.add_argument("--pie-root", default="data/PIE-Bench_v1")
    ap.add_argument("--per-page", type=int, default=5)
    ap.add_argument("--keys", nargs="*", help="only these images, in this order")
    ap.add_argument("--metrics", help="metrics.csv with both methods: adds PSNR/LPIPS under each caption")
    ap.add_argument("--kvedit-method", default="kvedit", help="method file prefix in --kvedit (e.g. fused_sched)")
    ap.add_argument("--kvedit-label", default="KV-Edit", help="column/caption name for --kvedit")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kv_name = args.kvedit_label

    samples = {s.key: s for s in load_pie_bench(args.pie_root)}
    ours_root, kv_root, out = Path(args.ours), Path(args.kvedit), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = args.keys or sorted(p.parent.name for p in ours_root.glob("*/source.png"))
    scores = {}
    if args.metrics:
        import pandas as pd
        df = pd.read_csv(args.metrics, dtype={"key": str})
        for r in df.itertuples():
            scores[(r.key.zfill(12), r.noise, r.method)] = (r.psnr, r.lpips_1e3)
    cols = ["Original (green = true edit area)", "Mask given to both methods", f"{kv_name} result",
            "Our result", "Our corrected mask"]
    written = []
    for nz in args.noise:
        for page in range(0, len(keys), args.per_page):
            chunk = keys[page:page + args.per_page]
            head, row_h = 70, S + 70
            sheet = Image.new("RGB", (5 * S, head + row_h * len(chunk)), "white")
            d = ImageDraw.Draw(sheet)
            d.text((10, 8), f"{NOISE_NAME.get(nz, nz)}  —  {kv_name} vs ours", fill="black", font=font(24))
            for c, name in enumerate(cols):
                d.text((c * S + 8, 42), name, fill="#444", font=font(15))
            for r, k in enumerate(chunk):
                s = samples[k]
                src = np.array(Image.open(ours_root / k / "source.png").convert("RGB"))
                gt = s.mask(src.shape[:2])
                z = np.load(ours_root / k / f"fused_sched__{nz}_masks.npz")
                kv = np.array(Image.open(kv_root / k / f"{args.kvedit_method}__{nz}.png").convert("RGB"))
                ours = np.array(Image.open(ours_root / k / f"fused_sched__{nz}.png").convert("RGB"))
                given = tint(src, z["user_px"], (255, 40, 40))
                corrected = tint(src, z["final"] if "final" in z else z["user"], (40, 140, 255))  # removal: no trim
                y = head + r * row_h
                caption = f"{CATEGORIES.get(s.category, s.category)}:  \"{s.source_clean}\"  →  \"{s.target_clean}\""
                lines = textwrap.wrap(caption, 150)[:2]
                kv_s, us_s = scores.get((k, nz, "kvedit")), scores.get((k, nz, "fused_sched"))
                if kv_s and us_s:
                    lines = lines[:1] + [f"Background PSNR: {kv_name} {kv_s[0]:.1f} → ours {us_s[0]:.1f}   ·   "
                                         f"LPIPS×1e3: {kv_name} {kv_s[1]:.0f} → ours {us_s[1]:.0f}"]
                d.text((8, y + 4), "\n".join(lines), fill="black", font=font(14))
                for c, im in enumerate([outline(src, gt, (0, 255, 0)), outline(given, gt, (0, 255, 0)),
                                        kv, ours, outline(corrected, gt, (0, 255, 0))]):
                    sheet.paste(tile(im), (c * S, y + 44))
            p = out / f"{nz}_{page // args.per_page + 1}.png"
            sheet.save(p)
            written.append(p)
    print("\n".join(map(str, written)))


if __name__ == "__main__":
    main()
