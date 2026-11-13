# Adaptive KV-Edit

Training-free editing on FLUX-dev with KV-Edit, where the user mask is refined by FLUX's own
image→edit-word attention and tightened over the denoising steps.

## Method (one paragraph)

Invert the source image once with KV-Edit (mask-independent, `attn_mask=False`). Run a few
full-image forward passes on the cached inverted latents with the target prompt and record, at every
double/single block, how much each image token attends to the edit-word tokens. Normalise the averaged
map `A`, and score each token as `s = w_user·M_user + (1−w_user)·A`, restricted to a dilated user mask.
Denoise with KV-Edit, regenerating tokens with `s > τ_loose` early and shrinking to `s > τ_tight` by
`sched_end`; tokens dropped from the mask are reset to the inverted source latent at that step
(per-step binarisation, which fits KV-Edit's fixed token set). With a constant user mask this is
bit-identical to KV-Edit (`tests/test_cpu_pipeline.py::test_matches_original_kvedit`).

Object removal is the exception: the whole object must be redrawn, so trimming would leave parts of it
behind. Removals (PIE-Bench `delete_object`, or the demo's "Object removal" checkbox) use the user mask
as-is, exactly like KV-Edit (`controller.keep_mask_on_removal`).

## Layout

```
adaptive_kvedit/
  models/editing_backbone.py   FLUX/T5/CLIP/AE loading, offload, invert / edit(mask schedule) / probe
  models/feature_store.py      KV cache in RAM or spilled to disk
  models/text_encoder.py       [bracket] parsing, edit-token indices for FLUX's T5
  features/extraction.py       AttentionRecorder: per-(step, block) image→edit-token maps
  features/visualization.py    overlays, per-layer and per-step grids
  controller/                  fusion, attention-only, timestep schedule
  evaluation/metrics.py        PnPInversion protocol: bg PSNR/LPIPS/MSE/SSIM, CLIP whole/edited
  data/                        PIE-Bench loader, noisy-mask perturbations
scripts/  run_batch, compute_metrics, comparison, visualize_attention, make_figures,
          tune_controller, analyze_losses, compare_sheets, convert_fp8
demo/app.py                    Gradio demo
configs/default.yaml
```

## Setup

```bash
git clone --recursive <this repo> && cd adaptive-kvedit   # --recursive pulls third_party/KV-Edit
# (already cloned without it: git submodule update --init)
conda create -n akv python=3.10 -y && conda activate akv
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124  # match your CUDA driver
pip install -r requirements.txt
huggingface-cli login            # accept the FLUX.1-dev license first
# PIE-Bench (PnPInversion, via Google Form) -> data/PIE-Bench_v1/{mapping_file.json,annotation_images/}
pytest tests -q                  # CPU, ~10 s
python scripts/convert_fp8.py    # one-time, CPU only: FLUX-dev -> checkpoints/flux-dev-fp8 (~12 GB)
```

Memory (defaults target a 16 GB GPU): FP8 weight-only FLUX (~12 GB), prompt embeddings computed
first with T5/CLIP alone on the GPU and cached in `cache/text/`, KV cache spilled to `cache/features/`
(~17 GB per 512px image, deleted after each image). FP8 forward at 512px ≈ 1.0 s on an RTX 4060 Ti.
On a 40–80 GB GPU use `--set backbone.precision=bf16 backbone.feature_store=ram`.

The GPU needs ~15 GB free: stop other GPU jobs before a run.

## Runbook

```bash
# 1. Sanity check: KV-Edit's own example (data/mini_pie); compare with data/mini_pie/kvedit_reference_nana.png
python scripts/run_batch.py --out runs/mini --set pie_root=data/mini_pie --methods kvedit fused_sched
python scripts/visualize_attention.py --run runs/mini      # where FLUX attends for the edit word

# 2. Tuning on 20 dev images (disjoint from the test set), clean + noisy masks
python scripts/run_batch.py --out runs/dev --split dev --limit 20 --methods kvedit fused_sched \
    --noise none dilate erode box
python scripts/tune_controller.py --run runs/dev            # CPU; writes runs/dev/tuned.yaml
python scripts/run_batch.py --out runs/dev_tuned --split dev --limit 20 --methods fused_sched \
    --noise none dilate erode box --set @runs/dev/tuned.yaml
python scripts/compute_metrics.py --run runs/dev --device cuda
python scripts/compute_metrics.py --run runs/dev_tuned --device cuda
python scripts/analyze_losses.py --kvedit runs/dev/metrics.csv --ours runs/dev_tuned/metrics.csv
python scripts/compare_sheets.py --kvedit runs/dev --ours runs/dev_tuned --out runs/dev_tuned/sheets

# 3. Test set (100 images) and ablations, with the tuned settings
python scripts/run_batch.py --out runs/main --limit 100 --methods kvedit fused_sched --set @runs/dev/tuned.yaml
python scripts/run_batch.py --out runs/ablation --limit 30 --set @runs/dev/tuned.yaml \
    --methods kvedit attn fused fused_sched --noise none dilate erode box
python scripts/compute_metrics.py --run runs/main --device cuda
python scripts/compute_metrics.py --run runs/ablation --device cuda
python scripts/comparison.py runs/main/metrics.csv runs/ablation/metrics.csv --by-category
python scripts/make_figures.py --run runs/ablation --noise dilate --methods kvedit fused_sched
python demo/app.py
```

Older baselines (P2P, MasaCtrl, RF-Edit, …) are cited, not rerun: copy the KV-Edit paper's PIE-Bench
numbers into `configs/reported_baselines.csv` and `comparison.py` appends them.

## Results (work in progress)

PIE-Bench, background preservation outside the ground-truth mask (PSNR ↑ / LPIPS×1e3 ↓ / MSE×1e4 ↓).
Style edits have full-image masks, so no background, and are left out of these metrics.

Baseline reproduction:

| | PSNR | LPIPS | MSE |
|---|---|---|---|
| KV-Edit, paper (620 images) | 35.87 | 9.92 | 4.69 |
| KV-Edit, this repo (100 test images, FP8) | 35.66 | 8.04 | 4.63 |

### Test set: 100 images

Adaptive mask (`fused_sched`) vs KV-Edit on 100 stratified test images (10 per category, disjoint from
the dev images), settings fixed from dev tuning (`runs/dev_v2/tuned.yaml`: `w_user=0.6, tau 0.6→0.7`).
Each image is edited with the ground-truth mask and three rough versions of it. KV-Edit: `runs/test`;
ours: `runs/test_v2` (hole filling kept inside the user mask) with the 10 removal images rerun in
`runs/test_v3_removal` (no trimming on removals).

| Mask given | KV-Edit PSNR | Ours PSNR | KV-Edit LPIPS | Ours LPIPS | Ours better / same / worse | CLIP edited KV / ours |
|---|---|---|---|---|---|---|
| clean | 35.66 | **36.00** | 8.04 | **7.56** | 42 / 30 / 6 | 22.50 / 22.32 |
| too small (erode 24 px) | 37.43 | **37.45** | 6.31 | **6.29** | 20 / 53 / 5 | 21.29 / 21.20 |
| too big (dilate 24 px) | 28.76 | **30.05** | 25.25 | **19.40** | 59 / 15 / 4 | 22.77 / 22.60 |
| bounding box | 27.22 | **29.06** | 48.59 | **32.03** | 55 / 18 / 5 | 22.96 / 22.73 |

Better / same / worse counts images by LPIPS, out of the 78 non-style images (removals count as "same":
they are pixel-identical to KV-Edit).

- Ours is better than or equal to KV-Edit with every mask type.
- Rough masks: ours preserves much more background (box: +1.8 dB PSNR, −34% LPIPS; too big: +1.3 dB,
  −23% LPIPS).
- Clean masks: ours is also slightly better (+0.34 dB), because it drops edge tokens the edit does not need.
- Too-small masks: tie. The tuned controller only trims masks, never grows them.
- No image gets noticeably worse: the largest PSNR loss on any image is 0.2 dB.
- Cost: a slightly lower edit score (CLIP edited −0.1 to −0.2).

### Examples

Selected test images where ours does best and the edit worked for both methods; they are not typical
of every image (averages are in the table above). Columns: original with the true edit area outlined in
green, the mask given to both methods (red), KV-Edit, ours, and our corrected mask (blue).

**Box mask.** KV-Edit redraws the whole box, so the woman, the cat and the bowl change; ours trims the box
and keeps them:

![Box-mask examples](docs/figures/examples_box_mask.jpg)

**Mask 24 px too big:**

![Too-big-mask examples](docs/figures/examples_too_big_mask.jpg)

**Clean mask.** Differences are usually small; here KV-Edit reshapes the arms and hair, ours keeps them:

![Clean-mask example](docs/figures/examples_clean_mask.jpg)

**Object removal: ours before vs after the removal rule.** Trimming kept parts of the bottle, the cat
and the painting; now the whole mask is redrawn, as in KV-Edit. The background score drops, but the
object is actually removed:

![Removal before/after](docs/figures/removal_before_after.jpg)

To rebuild them, or browse all 100 test images: `runs/test_final` links each image's final result
(`runs/test_v2`, or `runs/test_v3_removal` for removals), and `metrics_vs_kvedit.csv` holds both
methods' rows. `python scripts/compare_sheets.py --kvedit runs/test --ours runs/test_final --metrics
runs/test_final/metrics_vs_kvedit.csv --noise box --keys 122000000004 821000000009 813000000000
--out runs/test_final/examples` (leave out `--keys` for every image; `--kvedit runs/test_v2
--kvedit-method fused_sched --kvedit-label "Ours before"` compares two of our runs).

### Dev set: 20 images (used for tuning)

Same comparison on the 20 dev images (KV-Edit: `runs/dev_v2`; ours: `runs/dev_v3_tuned`). These images
were used to pick the settings, so the results are optimistic:

| Mask given | KV-Edit PSNR | Ours PSNR | KV-Edit LPIPS | Ours LPIPS | CLIP edited KV / ours |
|---|---|---|---|---|---|
| clean | 36.15 | **36.17** | 9.69 | **9.45** | 19.84 / 19.80 |
| too small (erode 24 px) | 37.25 | 37.25 | 7.89 | **7.88** | 19.21 / 19.26 |
| too big (dilate 24 px) | 29.93 | **30.47** | 27.46 | **23.74** | 20.55 / 20.65 |
| bounding box | 30.00 | **30.81** | 39.37 | **28.40** | 20.48 / 20.49 |

History of the clean-mask PSNR gap to KV-Edit:

1. First version: about −2 dB on clean dev masks. Two fixes closed most of it. PIE-Bench masks contain
   a 1-px image frame, now removed from the methods' input (kept for scoring, as in the PnPInversion
   protocol; it had turned every `box` mask into the full image). The adaptive mask now also keeps the
   user mask's sub-token edges instead of whole 16×16 tokens.
2. Test set after those fixes (`runs/test`): −0.57 dB, with about 10 images losing 3–6 dB. Hole filling
   (`close_tokens=1`) added whole tokens just outside the user mask on 63 of 100 images.
3. Hole filling now stays inside the user mask (`runs/test_v2`): +0.36 dB, with no image losing more
   than 0.2 dB. Rough-mask gains are kept (box) or grow (too big: +1.2 → +1.4 dB).
4. Removals use the user mask as-is (`runs/test_v3_removal`). Trimming had left parts of removed
   objects; the 10 removal images now match KV-Edit exactly. Final: +0.34 dB on clean masks, +1.8 dB
   with box masks.

## Limitations and future work

- The tuned controller only shrinks masks; too-small masks are not corrected.
- Trimming the mask slightly lowers the edit score (CLIP edited −0.1 to −0.2).
- Removals gain nothing over KV-Edit: they need to know the edit is a removal (PIE-Bench category or the
  demo checkbox) and then fall back to the user mask. Trimming with the removed words' attention is
  possible future work.
- The final mask is built from 16×16 tokens, so the boundary between kept and regenerated areas can
  show as a faint blocky seam on flat backgrounds.
- Attention is from one probe pass per step on the inverted latent, not from the edit trajectory.
- Masks are binary per step; soft latent blending is the alternative.
- Deferred: DINOv2 feature fusion, CLIP/dense-CLIP relevance, layer-adaptive control.
