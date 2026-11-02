"""KV-Edit wrapper: cached inversion, per-step mask schedules, attention probe passes."""
import contextlib
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from collections.abc import MutableMapping

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor

import adaptive_kvedit  # noqa: F401  (puts KV-Edit on sys.path)
from flux.sampling import denoise_kv, get_schedule, unpack

from .feature_store import CopyOnRead, NullStore, make_store

TXT_LEN = 512  # hard-coded in KV-Edit's Flux_kv


@dataclass
class EditOptions:
    num_steps: int = 28
    skip_step: int = 4
    inversion_guidance: float = 1.5
    denoise_guidance: float = 5.5
    attn_scale: float = 1.0
    re_init: bool = False
    seed: int = 42


@dataclass
class InversionState:
    z0: Tensor                      # packed source latent (1, L, 64)
    feature: MutableMapping         # KV-Edit cache: '{t}_img' latents and '{t}_{id}_{MB|SB}_{K|V}'
    timesteps: list[float]
    skip_step: int
    height: int
    width: int
    meta: dict = field(default_factory=dict)

    @property
    def num_tokens(self) -> int:
        return self.z0.shape[1]

    @property
    def token_hw(self) -> tuple[int, int]:
        return self.height // 16, self.width // 16

    def close(self):
        if hasattr(self.feature, "close"):
            self.feature.close()
        else:
            self.feature.clear()


# ---------- mask packing ----------

def pack_pixel_mask(mask: Tensor, height: int, width: int) -> Tensor:
    """Pixel mask (1,1,H,W) -> packed (1,L,64), exactly as KV-Edit does it."""
    h, w = height // 8, width // 8
    mask = F.interpolate(mask.float(), size=(h, w), mode="bilinear", align_corners=False)
    mask[mask > 0] = 1
    mask = repeat(mask, "b c h w -> b (repeat c) h w", repeat=16)
    return rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)


def token_mask_to_packed(token_mask: Tensor) -> Tensor:
    """Token mask (1,L) or (h,w) in [0,1] -> packed (1,L,64)."""
    tm = token_mask.reshape(1, -1).float()
    return tm[..., None].expand(-1, -1, 64).contiguous()


def packed_to_token_mask(packed: Tensor) -> Tensor:
    """Packed (1,L,64) -> bool (1,L); same rule KV-Edit uses for mask_indices."""
    return packed.sum(dim=2) > 0.5


def attention_scale(num_tokens: int, regen_idx: Tensor, scale: float, dtype, device) -> Tensor:
    """Vectorised KV-Edit create_attention_scale: additive bias on background keys."""
    s = torch.zeros(1, 1, num_tokens + TXT_LEN, dtype=dtype, device=device)
    bg = torch.ones(num_tokens, dtype=torch.bool, device=device)
    bg[regen_idx] = False
    s[0, 0, TXT_LEN:][bg] = scale
    return s


# ---------- core (model only; testable with a tiny Flux_kv) ----------

class KVEditCore:
    def __init__(self, model, device="cuda", name="flux-dev", feature_store="ram", feature_dir=None):
        self.model = model
        self.device = torch.device(device)
        self.name = name
        self.feature_store = feature_store
        self.feature_dir = feature_dir

    def _model_ctx(self):
        return contextlib.nullcontext()

    def schedule(self, num_tokens: int, opts: EditOptions) -> list[float]:
        ts = get_schedule(opts.num_steps, num_tokens, shift=(self.name != "flux-schnell"))
        return ts[opts.skip_step:]

    @torch.inference_mode()
    def invert(self, inp: dict, opts: EditOptions, height: int, width: int) -> InversionState:
        """Mask-independent inversion (KV-Edit with attn_mask=False), so one run serves every mask config."""
        ts = self.schedule(inp["img"].shape[1], opts)
        info = {"feature": make_store(self.feature_store, self.feature_dir), "inverse": True}
        z0 = inp["img"].clone()
        with self._model_ctx():
            denoise_kv(self.model, **inp, timesteps=ts, guidance=opts.inversion_guidance, inverse=True, info=info)
        return InversionState(z0=z0, feature=info["feature"], timesteps=ts, skip_step=opts.skip_step,
                              height=height, width=width)

    @torch.inference_mode()
    def edit(self, state: InversionState, inp_tgt: dict, mask_schedule: list[Tensor], opts: EditOptions) -> Tensor:
        """KV-Edit denoising where the packed mask may change every step.

        Tokens regenerated = union over steps; at step i tokens outside mask_schedule[i]
        are reset to the inverted source latent (per-step binarisation). Final composite uses
        the last mask. A constant schedule reproduces Flux_kv_edit.denoise exactly.
        """
        assert opts.skip_step == state.skip_step, "edit skip_step must match inversion"
        ts = state.timesteps
        dev, dtype = self.device, state.z0.dtype
        assert len(mask_schedule) == len(ts) - 1, f"need {len(ts) - 1} masks, got {len(mask_schedule)}"
        masks = [m.to(dev, dtype) for m in mask_schedule]
        L = state.num_tokens

        union = torch.stack([packed_to_token_mask(m) for m in masks]).any(0)[0]
        regen_idx = torch.nonzero(union).squeeze(1)
        if len(regen_idx) == 0:
            return state.z0.clone()
        info = {
            "feature": CopyOnRead(state.feature, dev),
            "mask_indices": regen_idx,
            "inverse": False,
            "attention_scale": (attention_scale(L, regen_idx, opts.attn_scale, dtype, dev)
                                if opts.attn_scale != 0 and len(regen_idx) < L else None),  # full mask: as KV-Edit
        }

        torch.manual_seed(opts.seed)
        if opts.re_init:
            noise = torch.randn_like(state.z0)
            img = (state.z0 * (1 - ts[0]) + noise * ts[0])[:, regen_idx]
        else:
            img = info["feature"][f"{ts[0]}_img"].to(dev)[:, regen_idx]

        g = torch.full((img.shape[0],), opts.denoise_guidance, device=dev, dtype=img.dtype)
        with self._model_ctx():
            for i, (tc, tp) in enumerate(zip(ts[:-1], ts[1:])):
                info["t"] = tc
                m = masks[i][:, regen_idx]
                src = info["feature"][f"{tc}_img"].to(dev)[:, regen_idx]
                img = src * (1 - m) + img * m
                tv = torch.full((img.shape[0],), tc, dtype=img.dtype, device=dev)
                pred = self.model(img=img, img_ids=inp_tgt["img_ids"], txt=inp_tgt["txt"],
                                  txt_ids=inp_tgt["txt_ids"], y=inp_tgt["vec"], timesteps=tv,
                                  guidance=g, info=info)
                img = img + (tp - tc) * pred

        z = state.z0.clone()
        m = masks[-1][:, regen_idx]
        z[:, regen_idx] = z[:, regen_idx] * (1 - m) + img * m
        return z

    @torch.inference_mode()
    def probe(self, state: InversionState, inp: dict, step_indices: list[int], recorder, guidance: float):
        """Full-image forward passes on cached inverted latents; recorder captures attention."""
        dev = self.device
        g = torch.full((1,), guidance, device=dev, dtype=state.z0.dtype)
        with self._model_ctx():
            for si in step_indices:
                t = state.timesteps[si]
                z = state.feature[f"{t}_img"].to(dev)
                info = {"inverse": True, "feature": NullStore(), "t": t}
                tv = torch.full((1,), t, dtype=z.dtype, device=dev)
                with recorder.capture(step=si):
                    self.model(img=z, img_ids=inp["img_ids"], txt=inp["txt"], txt_ids=inp["txt_ids"],
                               y=inp["vec"], timesteps=tv, guidance=g, info=info)


# ---------- full backbone (FLUX + T5 + CLIP + AE) ----------

def enable_block_offload(model, device, pin: bool = False):
    """Keep transformer blocks on CPU and stream each to `device` for its forward."""
    for name, mod in model.named_children():
        if name not in ("double_blocks", "single_blocks"):
            mod.to(device)
    for blk in [*model.double_blocks, *model.single_blocks]:
        blk.to("cpu")
        cpu = {n: (p.data.pin_memory() if pin else p.data) for n, p in blk.named_parameters()}
        for n, p in blk.named_parameters():
            p.data = cpu[n]

        def pre(m, args, kwargs, cpu=cpu):
            for n, p in m.named_parameters():
                p.data = cpu[n].to(device, non_blocking=pin)

        def post(m, args, kwargs, out, cpu=cpu):
            for n, p in m.named_parameters():
                p.data = cpu[n]
            return out

        blk.register_forward_pre_hook(pre, with_kwargs=True)
        blk.register_forward_hook(post, with_kwargs=True)


class EditingBackbone(KVEditCore):
    """FLUX + AE, with T5/CLIP used only to fill an on-disk prompt-embedding cache.

    precision: 'bf16' (original weights) or 'fp8' (converted, ~12 GB; scripts/convert_fp8.py).
    offload:   'none' | 'model' (swap whole modules via CPU) | 'blocks' (stream blocks; bf16 only).
    On a 16 GB GPU use precision=fp8, offload=none: text encoders are loaded only while the
    FLUX transformer is unloaded (see cache_prompts).
    """

    def __init__(self, name="flux-dev", device="cuda", offload="none", precision="bf16", fp8_dir=None,
                 text_cache="cache/text", feature_store="ram", feature_dir=None, pin_memory=False):
        from flux.util import load_ae

        assert precision in ("bf16", "fp8") and offload in ("none", "model", "blocks")
        assert not (precision == "fp8" and offload == "blocks"), "fp8 already fits; blocks offload is for bf16"
        self.offload, self.precision, self.fp8_dir, self.pin_memory = offload, precision, fp8_dir, pin_memory
        self.text_cache = Path(text_cache)
        self.text_cache.mkdir(parents=True, exist_ok=True)
        dev = torch.device(device)
        self.ae = load_ae(name, device="cpu" if offload != "none" else dev).eval()
        self._tokenizer = None
        super().__init__(None, dev, name, feature_store, feature_dir)

    # -- transformer --
    def load_model(self):
        if self.model is not None:
            return self.model
        from flux.model import Flux_kv
        from flux.util import configs, load_flow_model

        home = "cpu" if self.offload != "none" else self.device
        if self.precision == "fp8":
            from .fp8 import load_fp8_flux
            if not self.fp8_dir or not (Path(self.fp8_dir) / "index.json").exists():
                raise FileNotFoundError(f"no FP8 checkpoint at {self.fp8_dir}; run scripts/convert_fp8.py")
            self.model = load_fp8_flux(configs[self.name].params, self.fp8_dir, home, Flux_kv)
        else:
            self.model = load_flow_model(self.name, device=home, flux_cls=Flux_kv).eval()
            if self.offload == "blocks":
                enable_block_offload(self.model, self.device, pin=self.pin_memory)
        return self.model

    def unload_model(self):
        self.model = None
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # -- text --
    @property
    def tokenizer(self):
        """FLUX's T5 tokenizer only (no weights), for edit-token indices."""
        if self._tokenizer is None:
            from transformers import T5Tokenizer
            self._tokenizer = T5Tokenizer.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="tokenizer_2")
        return self._tokenizer

    def _text_path(self, prompt: str) -> Path:
        tag = "s" if self.name == "flux-schnell" else "d"
        return self.text_cache / f"{tag}_{hashlib.sha1(prompt.encode()).hexdigest()}.pt"

    @torch.inference_mode()
    def cache_prompts(self, prompts: list[str]):
        """Encode uncached prompts with T5/CLIP on the GPU, then free them.

        If the transformer is resident without offload, it is unloaded first and reloaded after.
        """
        todo = sorted({p for p in prompts if not self._text_path(p).exists()})
        if not todo:
            return
        from flux.modules.conditioner import HFEmbedder

        reload = self.model is not None and self.offload == "none"
        if reload:
            self.unload_model()
        kw = dict(torch_dtype=torch.bfloat16, device_map=str(self.device), low_cpu_mem_usage=True)
        t5 = HFEmbedder("black-forest-labs/FLUX.1-dev", max_length=256 if self.name == "flux-schnell" else 512,
                        is_clip=False, **kw)
        clip = HFEmbedder("black-forest-labs/FLUX.1-dev", max_length=77, is_clip=True, **kw)
        for p in todo:
            torch.save({"txt": t5([p]).cpu(), "vec": clip([p]).cpu()}, self._text_path(p))
        del t5, clip
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if reload:
            self.load_model()

    # -- device helpers --
    @contextlib.contextmanager
    def _on_device(self, *modules):
        if self.offload == "none":
            yield
            return
        for m in modules:
            m.to(self.device)
        try:
            yield
        finally:
            for m in modules:
                m.to("cpu")
            torch.cuda.empty_cache()

    def _model_ctx(self):
        if self.model is None:
            raise RuntimeError("call load_model() first")
        return self._on_device(self.model) if self.offload == "model" else contextlib.nullcontext()

    @staticmethod
    def crop16(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        return image[: h - h % 16, : w - w % 16]

    @torch.inference_mode()
    def encode(self, image: np.ndarray) -> Tensor:
        x = torch.from_numpy(np.ascontiguousarray(image[..., :3])).permute(2, 0, 1).float()[None] / 127.5 - 1
        with self._on_device(self.ae):
            return self.ae.encode(x.to(self.device)).to(torch.bfloat16)

    @torch.inference_mode()
    def decode(self, z_packed: Tensor, height: int, width: int) -> np.ndarray:
        z = unpack(z_packed.float(), height, width)
        with self._on_device(self.ae), torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(z.to(self.device))
        x = rearrange(x.clamp(-1, 1)[0].float().cpu(), "c h w -> h w c")
        return (127.5 * (x + 1.0)).round().clamp(0, 255).byte().numpy()

    @torch.inference_mode()
    def prepare(self, latent: Tensor, prompt: str) -> dict:
        """Same output as flux.sampling.prepare, with text embeddings from the cache."""
        path = self._text_path(prompt)
        if not path.exists():
            self.cache_prompts([prompt])
        emb = torch.load(path, weights_only=True)
        _, _, h, w = latent.shape
        img = rearrange(latent, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        img_ids = torch.zeros(h // 2, w // 2, 3)
        img_ids[..., 1] += torch.arange(h // 2)[:, None]
        img_ids[..., 2] += torch.arange(w // 2)[None, :]
        dev = self.device
        return {"img": img.to(dev), "img_ids": img_ids.reshape(1, -1, 3).to(dev),
                "txt": emb["txt"].to(dev), "txt_ids": torch.zeros(1, emb["txt"].shape[1], 3, device=dev),
                "vec": emb["vec"].to(dev)}
