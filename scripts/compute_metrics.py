"""Metrics for every output in a run dir -> <run>/metrics.csv.

python scripts/compute_metrics.py --run runs/main [--device cuda] [--no-clip]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from adaptive_kvedit.config import load_config
from adaptive_kvedit.data.pie_bench import load_pie_bench
from adaptive_kvedit.evaluation.metrics import Evaluator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-clip", action="store_true")
    ap.add_argument("--no-lpips", action="store_true")
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(args.config)
    pie = {s.key: s for s in load_pie_bench(cfg["pie_root"])}
    ev = Evaluator(args.device, use_clip=not args.no_clip, use_lpips=not args.no_lpips)

    rows = []
    for meta_path in sorted(run.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        s = pie[meta["key"]]
        src = np.array(Image.open(meta_path.parent / "source.png").convert("RGB"))
        gt = s.mask(src.shape[:2], border=True)
        for name, info in meta["edits"].items():
            out_path = meta_path.parent / f"{name}.png"
            if not out_path.exists():
                continue
            method, noise = name.split("__")
            out = np.array(Image.open(out_path).convert("RGB"))
            r = ev(src, out, gt, s.target_clean)
            rows.append({"key": s.key, "category": s.category, "method": method, "noise": noise,
                         "final_tokens": info.get("final_tokens"), "user_tokens": info.get("user_tokens"),
                         "attn_fallback": meta.get("attn_fallback", False), **r})
        print(meta["key"], len(meta["edits"]), flush=True)

    if not rows:
        sys.exit(f"no outputs under {run}")
    with open(run / "metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {run / 'metrics.csv'}")


if __name__ == "__main__":
    main()
