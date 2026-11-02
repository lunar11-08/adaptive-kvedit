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

Baseline reproduction, PIE-Bench background preservation (PSNR ↑ / LPIPS×1e3 ↓ / MSE×1e4 ↓):

| | PSNR | LPIPS | MSE |
|---|---|---|---|
| KV-Edit, paper (620 images) | 35.87 | 9.92 | 4.69 |
| KV-Edit, this repo (20 dev images, FP8) | 36.15 | 9.69 | 4.36 |

Adaptive masks on the 20 dev images: better than KV-Edit when the given mask is too big or a box,
about 2 dB worse with clean masks. Preliminary; the 100-image test run and ablations are pending.

Known issues:
- PIE-Bench masks contain a 1-px border frame, which distorts the `box`/`dilate` perturbations.
- The adaptive mask works per 16×16 token, coarser at edges than KV-Edit's per-channel mask.

## Limitations and future work

- Attention is from one probe pass per step on the inverted latent, not from the edit trajectory.
- Masks are binary per step; soft latent blending is the alternative (roadmap week 5).
- Deferred: DINOv2 feature fusion, CLIP/dense-CLIP relevance, layer-adaptive control.
