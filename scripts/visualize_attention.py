"""Day-1 make-or-break check: do edit-token attention maps localise the edit?

python scripts/visualize_attention.py --run runs/smoke [--keys 000000000000 ...]
Writes <run>/<key>/viz_layers.png (per block), viz_steps.png (per probe step), viz_masks_<noise>.png.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from adaptive_kvedit.features.extraction import AttentionRecorder
from adaptive_kvedit.features.visualization import plot_layer_grid, plot_masks, plot_step_grid


def localisation_score(rec, token_hw, mask_tok) -> dict:
    """Per block: attention mass inside the GT mask / mask area (1 = chance, higher = better)."""
    out = {}
    area = max(mask_tok.mean(), 1e-8)
    for kd, b in sorted({(kd, b) for (_, kd, b) in rec.maps}):
        m = rec.aggregate(kinds=(kd,), blocks={kd: {b}}).reshape(token_hw).numpy()
        out[f"{kd[0].upper()}{b}"] = float((m * mask_tok).sum() / max(m.sum(), 1e-8) / area)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--keys", nargs="*")
    args = ap.parse_args()

    run = Path(args.run)
    dirs = [run / k for k in args.keys] if args.keys else sorted(p.parent for p in run.glob("*/attn.pt"))
    scores = []
    for d in dirs:
        image = np.array(Image.open(d / "source.png").convert("RGB"))
        H, W = image.shape[:2]
        token_hw = (H // 16, W // 16)
        rec = AttentionRecorder.from_state_dict(torch.load(d / "attn.pt", weights_only=False))
        if not rec.maps:
            print(f"{d.name}: no edit tokens")
            continue
        masks_npz = sorted(d.glob("*__none_masks.npz"))
        user_px = np.load(masks_npz[0])["user_px"] if masks_npz else None
        plot_layer_grid(image, rec, token_hw, d / "viz_layers.png", user_mask=user_px, title=d.name)
        plot_step_grid(image, rec, token_hw, d / "viz_steps.png", user_mask=user_px, title=d.name)
        for npz in d.glob("*_masks.npz"):
            z = np.load(npz)
            panels = {k: z[k] for k in ("user", "attn", "score", "loose", "final") if k in z}
            plot_masks(image, panels, d / f"viz_masks_{npz.stem.replace('_masks', '')}.png", title=npz.stem)
        if user_px is not None:
            tok = torch.nn.functional.adaptive_max_pool2d(torch.from_numpy(user_px).float()[None, None],
                                                          token_hw)[0, 0].numpy()
            if tok.mean() < 0.9:                # near-full masks say nothing about localisation
                scores.append(localisation_score(rec, token_hw, tok))
        print(f"{d.name}: done")

    if scores:
        keys = scores[0].keys()
        mean = {k: np.mean([s[k] for s in scores]) for k in keys}
        best = sorted(mean.items(), key=lambda x: -x[1])[:10]
        print(f"\nattention lift inside GT mask (1 = chance), mean over {len(scores)} images, top 10 blocks:")
        for k, v in best:
            print(f"  {k:>4}: {v:.3f}")


if __name__ == "__main__":
    main()
