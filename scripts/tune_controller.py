"""Offline controller tuning on a dev run: no GPU, uses the saved attention maps.

For each image x mask noise, the controller turns the noisy user mask into a final mask; we score
its token IoU against the clean PIE-Bench mask (KV-Edit's IoU = the noisy mask itself).

python scripts/tune_controller.py --run runs/dev [--noise none dilate erode box]
-> prints the best configs, writes <run>/tuning.csv and <run>/tuned.yaml (use: --set @runs/dev/tuned.yaml)
"""
import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import yaml

from adaptive_kvedit.config import controller_config, load_config
from adaptive_kvedit.controller import build_schedule
from adaptive_kvedit.data.mask_perturb import perturb_mask
from adaptive_kvedit.data.pie_bench import load_pie_bench
from adaptive_kvedit.features.extraction import AttentionRecorder, select_from_config

SELECTS = {
    "all": {"double": "all", "single": "all"},
    "double": {"double": "all", "single": "none"},
    "single": {"double": "none", "single": "all"},
    "double_late": {"double": list(range(10, 19)), "single": "none"},
}
GRID = {
    "w_user": [0.0, 0.2, 0.4, 0.6],
    "tau_tight": [0.4, 0.5, 0.6, 0.7],
    "loose_gap": [0.0, 0.1, 0.2],
    "dilate_tokens": [1, 2, 3, 4],
}


def iou(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a > 0, b > 0
    u = (a | b).sum().item()
    return 1.0 if u == 0 else (a & b).sum().item() / u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=None, help="defaults to the run's saved config")
    ap.add_argument("--noise", nargs="+", default=["none", "dilate", "erode", "box"])
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    run = Path(args.run)
    cfg = load_config(args.config) if args.config else json.loads((run / "config.json").read_text())["cfg"]
    base = controller_config(cfg)
    n_steps = cfg["sampling"]["num_steps"] - cfg["sampling"]["skip_step"]
    radius = cfg.get("mask_noise_radius", 24)
    samples = {s.key: s for s in load_pie_bench(cfg["pie_root"])}
    probe_steps = cfg["attention"]["probe_steps"]
    step_opts = {"all": None, **{f"s{p}": [p] for p in probe_steps}}

    # per image: GT token mask, noisy pixel masks, attention maps per selection
    items = []
    for d in sorted(p.parent for p in run.glob("*/attn.pt")):
        rec = AttentionRecorder.from_state_dict(torch.load(d / "attn.pt", weights_only=False))
        if not rec.maps:
            continue
        s = samples[d.name]
        gt = s.mask((512, 512))
        H, W = gt.shape
        gt_t = torch.from_numpy(gt).float()[None, None]
        gt_tok = build_schedule("user", gt_t, None, H, W, n_steps, base)[1]["user"]
        attn = {(sn, tn): select_from_config(rec, {**sv, "steps": tv})
                for (sn, sv), (tn, tv) in itertools.product(SELECTS.items(), step_opts.items())}
        noisy = {nz: torch.from_numpy(perturb_mask(gt, nz, radius)).float()[None, None] for nz in args.noise}
        items.append((d.name, H, W, gt_tok, attn, noisy))
    print(f"{len(items)} images, noise={args.noise}")

    rows = []
    for name, H, W, gt_tok, _, noisy in items:            # KV-Edit: uses the noisy mask as is
        for nz, um in noisy.items():
            user_tok = build_schedule("user", um, None, H, W, n_steps, base)[1]["user"]
            rows.append({"cfg": "kvedit", "key": name, "noise": nz, "iou": iou(user_tok, gt_tok),
                         "frac": user_tok.sum().item() / gt_tok.sum().item()})

    combos = list(itertools.product(SELECTS, step_opts, *GRID.values()))
    for ci, (sel, st, w, tt, gap, dil) in enumerate(combos):
        cc = dataclasses.replace(base, w_user=w, tau_tight=tt, tau_loose=max(tt - gap, 0.05), dilate_tokens=dil)
        tag = f"{sel}|{st}|w{w}|t{tt}|g{gap}|d{dil}"
        for name, H, W, gt_tok, attn, noisy in items:
            for nz, um in noisy.items():
                final = build_schedule("fused_sched", um, attn[(sel, st)], H, W, n_steps, cc)[1]["final"]
                rows.append({"cfg": tag, "key": name, "noise": nz, "iou": iou(final, gt_tok),
                             "frac": final.sum().item() / gt_tok.sum().item()})
        if ci % 200 == 0:
            print(f"  {ci}/{len(combos)} configs", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(run / "tuning.csv", index=False)
    table = df.pivot_table(index="cfg", columns="noise", values="iou", aggfunc="mean")[args.noise]
    table["mean"] = table.mean(1)
    noisy_cols = [c for c in args.noise if c != "none"]
    table["noisy_mean"] = table[noisy_cols].mean(1) if noisy_cols else table["mean"]
    ranked = table.drop(index="kvedit").sort_values("mean", ascending=False)
    pd.set_option("display.width", 200)
    print("\nmask IoU vs clean PIE-Bench mask (higher = better)\n")
    print(pd.concat([table.loc[["kvedit"]], ranked.head(args.top)]).round(3).to_string())

    best = ranked.index[0]
    sel, st, w, tt, gap, dil = best.split("|")
    tt, w = float(tt[1:]), float(w[1:])
    tuned = {"attention": {"select": {**SELECTS[sel], "steps": step_opts[st]}},
             "controller": {"w_user": w, "tau_tight": tt, "tau_loose": round(max(tt - float(gap[1:]), 0.05), 3),
                            "dilate_tokens": int(dil[1:])}}
    (run / "tuned.yaml").write_text(yaml.safe_dump(tuned, sort_keys=False))
    print(f"\nbest: {best}  ->  {run / 'tuned.yaml'}")


if __name__ == "__main__":
    main()
