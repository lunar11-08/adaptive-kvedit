"""FP8 weight-only FLUX: block Linears stored as float8_e4m3fn with per-row scales, dequantised per call.

Conversion streams one tensor at a time (low RAM); loading streams shards straight to the GPU.
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

FP8 = torch.float8_e4m3fn
FP8_MAX = 448.0
BLOCK_PREFIXES = ("double_blocks.", "single_blocks.")


class FP8Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=FP8, device=device))
        self.register_buffer("scale", torch.empty(out_features, 1, dtype=torch.float32, device=device))
        self.register_buffer("bias", torch.empty(out_features, dtype=dtype, device=device) if bias else None)

    def forward(self, x: Tensor) -> Tensor:
        w = self.weight.to(x.dtype) * self.scale.to(x.dtype)
        return F.linear(x, w, None if self.bias is None else self.bias.to(x.dtype))

    @classmethod
    def from_linear(cls, lin: nn.Linear) -> "FP8Linear":
        m = cls(lin.in_features, lin.out_features, lin.bias is not None, lin.weight.dtype, lin.weight.device)
        m.weight, m.scale = quantize(lin.weight.data)
        if lin.bias is not None:
            m.bias = lin.bias.data.clone()
        return m


def quantize(w: Tensor) -> tuple[Tensor, Tensor]:
    w = w.float()
    scale = (w.abs().amax(dim=1, keepdim=True) / FP8_MAX).clamp_min(1e-12)
    return (w / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8), scale


def should_quantize(key: str, t: Tensor) -> bool:
    return key.startswith(BLOCK_PREFIXES) and key.endswith(".weight") and t.ndim == 2


def swap_block_linears(model: nn.Module) -> nn.Module:
    """Replace nn.Linear inside transformer blocks with (empty or quantised) FP8Linear."""
    for blocks in (model.double_blocks, model.single_blocks):
        for blk in blocks:
            for name, mod in list(blk.named_modules()):
                for cname, child in list(mod.named_children()):
                    if isinstance(child, nn.Linear):
                        if child.weight.is_meta:
                            new = FP8Linear(child.in_features, child.out_features, child.bias is not None,
                                            child.weight.dtype, device="meta")
                        else:
                            new = FP8Linear.from_linear(child)
                        setattr(mod, cname, new)
    return model


def convert_checkpoint(src: str | Path, dst_dir: str | Path, shard_bytes: int = 1 << 30) -> Path:
    """flux1-dev.safetensors -> dst_dir/{shard_*.safetensors, index.json}."""
    dst = Path(dst_dir)
    dst.mkdir(parents=True, exist_ok=True)
    shard, size, n, files = {}, 0, 0, []

    def flush():
        nonlocal shard, size, n
        if shard:
            name = f"shard_{n:03d}.safetensors"
            save_file(shard, dst / name)
            files.append(name)
            shard, size, n = {}, 0, n + 1

    with safe_open(str(src), framework="pt", device="cpu") as f:
        for key in f.keys():
            t = f.get_tensor(key)
            if should_quantize(key, t):
                w8, s = quantize(t)
                out = {key: w8, key[: -len("weight")] + "scale": s}
            else:
                out = {key: t.to(torch.bfloat16) if t.is_floating_point() else t}
            for k, v in out.items():
                shard[k] = v.contiguous()
                size += v.numel() * v.element_size()
            if size >= shard_bytes:
                flush()
    flush()
    (dst / "index.json").write_text(json.dumps({"format": "fp8_e4m3fn_rowwise", "shards": files}))
    return dst


def load_fp8_flux(params, ckpt_dir: str | Path, device, flux_cls) -> nn.Module:
    """Build flux_cls on meta, swap block Linears, stream converted shards to `device`."""
    ckpt_dir = Path(ckpt_dir)
    with torch.device("meta"):
        model = flux_cls(params).to(torch.bfloat16)
    swap_block_linears(model)
    expected = set(model.state_dict())
    seen = set()
    for name in json.loads((ckpt_dir / "index.json").read_text())["shards"]:
        sd = load_file(ckpt_dir / name, device=str(device))
        model.load_state_dict(sd, strict=False, assign=True)
        seen |= set(sd)
        del sd
    missing = expected - seen
    if missing:
        raise RuntimeError(f"{len(missing)} weights missing from FP8 checkpoint, e.g. {sorted(missing)[:5]}")
    return model.eval()
