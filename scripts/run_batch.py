"""Batch editing on a PIE-Bench subset: invert once per image, then run every method x mask-noise.

Main run:  python scripts/run_batch.py --out runs/main --limit 100 --methods kvedit fused_sched
Ablation:  python scripts/run_batch.py --out runs/ablation --limit 30 \
               --methods kvedit attn fused fused_sched --noise none dilate erode box
Tuning:    python scripts/run_batch.py --out runs/dev --split dev --limit 20 --methods kvedit fused_sched \
               --noise none dilate erode box      (disjoint from the 100 test images)
Resumable: finished outputs are skipped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from PIL import Image

from adaptive_kvedit.config import controller_config, edit_options, load_config
from adaptive_kvedit.controller import build_schedule
from adaptive_kvedit.data.mask_perturb import perturb_mask
from adaptive_kvedit.data.pie_bench import dev_subset, load_pie_bench, stratified_subset
from adaptive_kvedit.features.extraction import AttentionRecorder, select_from_config
from adaptive_kvedit.models.editing_backbone import EditingBackbone
from adaptive_kvedit.models.text_encoder import edit_token_indices


def pick_tokens(tokenizer, s, acfg) -> tuple[str, list[int]]:
    blended = s.blended_words if acfg.get("use_blended_word", True) else []
    tgt_extra = blended[1:2] if len(blended) > 1 else blended
    src_extra = blended[:1]
    src = acfg.get("token_source", "auto")
    if src in ("target", "auto"):
        idx = edit_token_indices(tokenizer, s.target_prompt, tgt_extra)
        if idx or src == "target":
            return "target", idx
    return "source", edit_token_indices(tokenizer, s.source_prompt, src_extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--keys", nargs="*", help="explicit PIE-Bench keys (overrides --limit)")
    ap.add_argument("--split", choices=["test", "dev"], default="test",
                    help="dev = --limit tuning images disjoint from the --n-test test images")
    ap.add_argument("--n-test", type=int, default=100)
    ap.add_argument("--methods", nargs="+", default=["kvedit", "fused_sched"])
    ap.add_argument("--noise", nargs="+", default=["none"])
    ap.add_argument("--set", nargs="*", default=[], help="config overrides, e.g. controller.w_user=0.5")
    args = ap.parse_args()

    cfg = load_config(args.config, args.set)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.json").write_text(json.dumps({"cfg": cfg, "args": vars(args)}, indent=2))

    samples = load_pie_bench(cfg["pie_root"])
    if args.keys:
        samples = [s for s in samples if s.key in set(args.keys)]
    elif args.split == "dev":
        samples = dev_subset(samples, args.limit, args.n_test, seed=cfg.get("seed", 0))
    else:
        samples = stratified_subset(samples, args.limit, seed=cfg.get("seed", 0))
    print(f"{len(samples)} samples, methods={args.methods}, noise={args.noise}")

    opts = edit_options(cfg)
    methods = {m: cfg["methods"][m] for m in args.methods}
    needs_attn = any(m["mode"] != "user" for m in methods.values())
    bb = EditingBackbone(**cfg["backbone"])
    # phase 1: text embeddings (T5/CLIP alone on the GPU), phase 2: FLUX
    bb.cache_prompts([p for s in samples for p in (s.source_clean, s.target_clean)])
    bb.load_model()

    for n, s in enumerate(samples):
        d = out_root / s.key
        jobs = [(nz, m) for nz in args.noise for m in methods
                if not (methods[m]["mode"] == "attn" and nz != "none")      # attn ignores the user mask
                and not (d / f"{m}__{nz}.png").exists()]
        if not jobs:
            continue
        d.mkdir(exist_ok=True)
        t0 = time.time()
        image = bb.crop16(s.load_image())
        H, W = image.shape[:2]
        Image.fromarray(image).save(d / "source.png")

        latent = bb.encode(image)
        inp_src = bb.prepare(latent, s.source_clean)
        inp_tgt = bb.prepare(latent, s.target_clean)
        state = bb.invert(inp_src, opts, H, W)
        t_inv = time.time() - t0
        try:
            meta = run_sample(bb, s, d, state, inp_src, inp_tgt, H, W, jobs, methods, cfg, opts, needs_attn)
        finally:
            state.close()                       # drop the ~17 GB disk cache even on a crash
        meta["t_invert"] = t_inv
        meta_path = d / "meta.json"
        if meta_path.exists():
            old = json.loads(meta_path.read_text())
            old["edits"].update(meta["edits"])
            meta["edits"] = old["edits"]
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"[{n + 1}/{len(samples)}] {s.key} cat={s.category} jobs={len(jobs)} "
              f"invert={t_inv:.0f}s total={time.time() - t0:.0f}s", flush=True)


def run_sample(bb, s, d, state, inp_src, inp_tgt, H, W, jobs, methods, cfg, opts, needs_attn):
    meta = {"key": s.key, "category": s.category, "source": s.source_prompt, "target": s.target_prompt,
            "blended": s.blended_words, "edits": {}}
    attn_map = None
    if needs_attn:
        which, tok = pick_tokens(bb.tokenizer, s, cfg["attention"])
        meta.update(token_prompt=which, token_idx=tok)
        rec = AttentionRecorder(bb.model, tok, head_chunk=cfg["attention"].get("head_chunk", 4))
        bb.probe(state, inp_tgt if which == "target" else inp_src, cfg["attention"]["probe_steps"], rec,
                 guidance=opts.denoise_guidance)
        torch.save(rec.state_dict(), d / "attn.pt")
        attn_map = select_from_config(rec, cfg["attention"]["select"]) if tok else None
        meta["attn_fallback"] = attn_map is None
        if attn_map is None:
            print(f"  {s.key}: no edit tokens, adaptive modes fall back to the user mask")

    base_mask = s.mask((H, W), border=False)
    for nz, m in jobs:
        t1 = time.time()
        um = perturb_mask(base_mask, nz, cfg.get("mask_noise_radius", 24))
        um_t = torch.from_numpy(um).float()[None, None].to(bb.device)
        mode = methods[m]["mode"] if attn_map is not None else "user"
        sched, dbg = build_schedule(mode, um_t, attn_map, H, W, len(state.timesteps) - 1,
                                    controller_config(cfg, methods[m]))
        z = bb.edit(state, inp_tgt, sched, opts)
        Image.fromarray(bb.decode(z, H, W)).save(d / f"{m}__{nz}.png")
        np.savez_compressed(d / f"{m}__{nz}_masks.npz", user_px=um,
                            **{k: v.cpu().numpy() for k, v in dbg.items()})
        meta["edits"][f"{m}__{nz}"] = {"seconds": time.time() - t1,
                                       "final_tokens": int(dbg.get("final", dbg["user"]).sum()),
                                       "user_tokens": int(dbg["user"].sum())}
    return meta


if __name__ == "__main__":
    main()
