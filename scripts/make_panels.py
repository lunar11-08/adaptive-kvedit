"""Per-image panels across all mask conditions.

Rows: clean / too small / too big / box mask. Columns: original, input mask, edit-word attention,
refined mask, KV-Edit, ours.

python scripts/make_panels.py --kvedit runs/test --ours runs/test_final \
    --metrics runs/test_final/metrics_vs_kvedit.csv --out runs/test_final/panels
--select median (default) picks, per category, the image whose LPIPS gain over KV-Edit is closest to
the median; --select best picks the largest gain; --keys overrides both.
"""
import argparse
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

from adaptive_kvedit.data.pie_bench import CATEGORIES, load_pie_bench
from adaptive_kvedit.features.visualization import overlay
from compare_sheets import font, outline, tint

S = 256
NOISES = [("none", "Clean mask"), ("erode", "Mask too small (−24 px)"),
          ("dilate", "Mask too big (+24 px)"), ("box", "Box mask")]
COLS = ["Original (green = true edit area)", "Input mask", "Edit-word attention",
        "Refined mask (ours)", "KV-Edit", "Ours"]
SKIP_CATEGORIES = {"3", "9"}          # removal: identical to KV-Edit; style: no background


def select(df: pd.DataFrame, how: str) -> list[str]:
    piv = df.pivot_table(index=["key", "category"], columns=["method", "noise"], values="lpips_1e3")
    gain = (piv["kvedit"] - piv["fused_sched"]).mean(axis=1).dropna().reset_index(name="gain")
    gain = gain[~gain.category.astype(str).isin(SKIP_CATEGORIES)]
    keys = []
    for _, g in gain.groupby("category"):
        if how == "best":
            keys.append(g.loc[g.gain.idxmax(), "key"])
        else:
            keys.append(g.loc[(g.gain - g.gain.median()).abs().idxmin(), "key"])
    return keys


def gray_tile(text: str) -> Image.Image:
    im = Image.new("RGB", (S, S), "#ddd")
    ImageDraw.Draw(im).multiline_text((S // 2, S // 2), text, fill="#555", font=font(15), anchor="mm",
                                      align="center")
    return im


def panel(k, s, ours_root, kv_root, scores) -> Image.Image:
    src = np.array(Image.open(ours_root / k / "source.png").convert("RGB"))
    gt = s.mask(src.shape[:2])
    head, cap, row_h = 64, 22, S + 22
    sheet = Image.new("RGB", (6 * S, head + row_h * len(NOISES)), "white")
    d = ImageDraw.Draw(sheet)
    title = f"{CATEGORIES.get(s.category, s.category)}:  \"{s.source_clean}\"  →  \"{s.target_clean}\""
    d.text((8, 6), textwrap.shorten(title, 175), fill="black", font=font(15))
    for c, name in enumerate(COLS):
        d.text((c * S + 6, 40), name, fill="#444", font=font(14))
    for r, (nz, label) in enumerate(NOISES):
        y = head + r * row_h
        z = np.load(ours_root / k / f"fused_sched__{nz}_masks.npz")
        user = z["user_px"]
        refined = z["final"] if "final" in z else z["user"]
        line = label
        kv_s, us_s = scores.get((k, nz, "kvedit")), scores.get((k, nz, "fused_sched"))
        if kv_s and us_s and not np.isnan(kv_s[0]):
            line += (f"   ·   background PSNR: KV-Edit {kv_s[0]:.1f} → ours {us_s[0]:.1f}"
                     f"   ·   LPIPS×1e3: KV-Edit {kv_s[1]:.0f} → ours {us_s[1]:.0f}")
        d.text((8, y + 3), line, fill="black", font=font(14))
        if "attn" in z:
            att = (overlay(src, z["attn"]) * 255).astype(np.uint8)
            att_tile = Image.fromarray(outline(att, user, (255, 255, 255), w=2)).resize((S, S))
        else:
            att_tile = gray_tile("not used")
        tiles = [Image.fromarray(outline(src, gt, (0, 255, 0))).resize((S, S)),
                 Image.fromarray(tint(src, user, (255, 40, 40))).resize((S, S)),
                 att_tile,
                 Image.fromarray(outline(tint(src, refined, (40, 140, 255)), user, (255, 40, 40), w=2)).resize((S, S)),
                 Image.open(kv_root / k / f"kvedit__{nz}.png").convert("RGB").resize((S, S)),
                 Image.open(ours_root / k / f"fused_sched__{nz}.png").convert("RGB").resize((S, S))]
        for c, t in enumerate(tiles):
            sheet.paste(t, (c * S, y + cap))
    return sheet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kvedit", required=True, help="run dir with kvedit__<noise>.png")
    ap.add_argument("--ours", required=True, help="run dir with fused_sched__<noise>.png and _masks.npz")
    ap.add_argument("--metrics", required=True, help="metrics.csv with kvedit and fused_sched rows")
    ap.add_argument("--select", choices=["median", "best"], default="median")
    ap.add_argument("--keys", nargs="*")
    ap.add_argument("--pie-root", default="data/PIE-Bench_v1")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.metrics, dtype={"key": str, "category": str})
    df["key"] = df.key.str.zfill(12)
    scores = {(r.key, r.noise, r.method): (r.psnr, r.lpips_1e3) for r in df.itertuples()}
    keys = args.keys or select(df, args.select)
    samples = {s.key: s for s in load_pie_bench(args.pie_root)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in keys:
        s = samples[k]
        p = out / f"{s.category}_{CATEGORIES.get(s.category, s.category)}_{k}.png"
        panel(k, s, Path(args.ours), Path(args.kvedit), scores).save(p)
        print(p)


if __name__ == "__main__":
    main()
