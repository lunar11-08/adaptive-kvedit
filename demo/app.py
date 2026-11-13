"""Gradio demo: KV-Edit's app plus attention-guided masks.

python demo/app.py [--offload model] [--port 41032]
Mark edit words in the target prompt with [brackets], e.g. "a [dog] on a sofa".
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr
import numpy as np
import torch
from PIL import Image

from adaptive_kvedit.config import controller_config, edit_options, load_config
from adaptive_kvedit.controller import MODES, build_schedule
from adaptive_kvedit.features.extraction import AttentionRecorder, select_from_config
from adaptive_kvedit.features.visualization import overlay
from adaptive_kvedit.models.editing_backbone import EditingBackbone
from adaptive_kvedit.models.text_encoder import edit_token_indices, strip_brackets


class Demo:
    def __init__(self, cfg):
        self.cfg = cfg
        self.bb = EditingBackbone(**cfg["backbone"])
        self.bb.load_model()
        self.state = None

    def fit(self, img: np.ndarray) -> np.ndarray:
        """Downscale so the long side is at most max_side (16 GB GPU), then crop to multiples of 16."""
        h, w = img.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        self.scaled = (round(w * s), round(h * s)) if s < 1 else None
        if self.scaled:
            img = np.asarray(Image.fromarray(img).resize(self.scaled, Image.LANCZOS))
        return self.bb.crop16(img)

    def invert(self, canvas, source_prompt, steps, skip):
        if self.state is not None:
            self.state.close()
            self.state = None
        if canvas is None or canvas.get("background") is None:
            raise gr.Error("upload an image first")
        img = self.fit(np.asarray(canvas["background"])[..., :3])
        self.image = img
        self.opts = edit_options(self.cfg)
        self.opts.num_steps, self.opts.skip_step = int(steps), int(skip)
        try:
            self.latent = self.bb.encode(img)
            self.source_prompt = strip_brackets(source_prompt)
            inp = self.bb.prepare(self.latent, strip_brackets(source_prompt))
            self.state = self.bb.invert(inp, self.opts, *img.shape[:2])
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise gr.Error("GPU out of memory: try a smaller image (--max-side)")
        return f"inverted ({img.shape[1]}×{img.shape[0]})"

    def edit(self, canvas, target_prompt, mode, removal, w_user, tau_tight, tau_loose, guidance, attn_scale):
        if self.state is None:
            raise gr.Error("run Inverse first")
        H, W = self.image.shape[:2]
        if not canvas.get("layers"):
            raise gr.Error("draw a mask")
        alpha = np.asarray(canvas["layers"][0])[..., 3]
        if self.scaled:                                       # same resize + crop as the image
            alpha = np.asarray(Image.fromarray(alpha).resize(self.scaled, Image.NEAREST))
        alpha = alpha[:H, :W]
        user = torch.from_numpy((alpha > 0).astype(np.float32))[None, None].to(self.bb.device)
        if not user.any():
            raise gr.Error("draw a mask")
        self.opts.denoise_guidance, self.opts.attn_scale = guidance, attn_scale
        self.bb.cache_prompts([strip_brackets(target_prompt)])   # swaps FLUX out briefly on 16 GB
        inp_tgt = self.bb.prepare(self.latent, strip_brackets(target_prompt))

        attn, heat = None, None
        tok = edit_token_indices(self.bb.tokenizer, target_prompt)
        cc = controller_config(self.cfg)
        cc.w_user, cc.tau_tight, cc.tau_loose = w_user, tau_tight, tau_loose
        if removal and cc.keep_mask_on_removal:
            mode = "user"                                     # no trimming, so skip the attention probe
        if mode != "user" and tok:
            rec = AttentionRecorder(self.bb.model, tok)
            self.bb.probe(self.state, inp_tgt, self.cfg["attention"]["probe_steps"], rec, guidance)
            attn = select_from_config(rec, self.cfg["attention"]["select"])
            heat = overlay(self.image, attn.reshape(H // 16, W // 16).float().cpu().numpy())
        if attn is None:
            mode = "user"
        sched, dbg = build_schedule(mode, user, attn, H, W, len(self.state.timesteps) - 1, cc)
        out = self.bb.decode(self.bb.edit(self.state, inp_tgt, sched, self.opts), H, W)
        final = overlay(self.image, dbg.get("final", dbg["user"]).float().cpu().numpy(), cmap="magma")
        return out, heat, final


def build(demo: Demo):
    with gr.Blocks(title="Adaptive KV-Edit") as ui:
        gr.Markdown("## Adaptive KV-Edit\nInverse once, then edit with any mask mode. "
                    "Mark edit words in the target prompt with [brackets].")
        with gr.Row():
            with gr.Column():
                canvas = gr.ImageEditor(label="Image + mask", sources=("upload",), transforms=[],
                                        brush=gr.Brush(colors=["#ff0000"], color_mode="fixed"), format="png")
                src = gr.Textbox(label="Source prompt")
                tgt = gr.Textbox(label="Target prompt (edit words in [brackets])")
                steps = gr.Slider(1, 50, 28, step=1, label="Steps")
                skip = gr.Slider(0, 30, 4, step=1, label="Skip steps")
                inv_btn = gr.Button("Inverse")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                mode = gr.Dropdown(list(MODES), value="fused_sched", label="Mask mode")
                removal = gr.Checkbox(label="Object removal (use the mask as drawn, no trimming)", value=False)
                c = demo.cfg["controller"]
                w_user = gr.Slider(0, 1, c["w_user"], step=0.05, label="w_user")
                tau_tight = gr.Slider(0, 1, c["tau_tight"], step=0.05, label="tau_tight")
                tau_loose = gr.Slider(0, 1, c["tau_loose"], step=0.05, label="tau_loose")
                guidance = gr.Slider(1, 10, 5.5, step=0.1, label="Denoise guidance")
                attn_scale = gr.Slider(0, 5, 1, step=0.1, label="attn_scale")
                edit_btn = gr.Button("Edit")
                out = gr.Image(label="Edited")
                with gr.Row():
                    heat = gr.Image(label="Edit-token attention")
                    final = gr.Image(label="Final mask")
        inv_btn.click(demo.invert, [canvas, src, steps, skip], [status])
        edit_btn.click(demo.edit, [canvas, tgt, mode, removal, w_user, tau_tight, tau_loose, guidance, attn_scale],
                       [out, heat, final])
    return ui


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--port", type=int, default=41032)
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--max-side", type=int, default=512, help="uploads are downscaled to this long side")
    args = ap.parse_args()
    demo = Demo(load_config(args.config, args.set))
    demo.max_side = args.max_side
    build(demo).launch(server_name="0.0.0.0", server_port=args.port,
                                                           share=args.share)
