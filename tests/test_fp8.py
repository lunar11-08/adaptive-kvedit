"""FP8 conversion/loading round trip on the tiny Flux_kv."""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_cpu_pipeline import DT, H, W, make_inp, tiny_model  # noqa: E402

from flux.model import Flux_kv  # noqa: E402
from adaptive_kvedit.models.editing_backbone import EditOptions, KVEditCore  # noqa: E402
from adaptive_kvedit.models.fp8 import FP8Linear, convert_checkpoint, load_fp8_flux, quantize  # noqa: E402


def test_quantize_error_small():
    w = torch.randn(256, 512) * 0.02
    w8, s = quantize(w)
    err = ((w8.float() * s) - w).abs().mean() / w.abs().mean()
    assert w8.dtype == torch.float8_e4m3fn and err < 0.05


def test_convert_and_load_round_trip(tmp_path):
    model = tiny_model()
    src = tmp_path / "flux.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, src)
    convert_checkpoint(src, tmp_path / "fp8", shard_bytes=50_000)
    assert len(list((tmp_path / "fp8").glob("shard_*.safetensors"))) > 1
    q = load_fp8_flux(model.params, tmp_path / "fp8", "cpu", Flux_kv)
    n_fp8 = sum(isinstance(m, FP8Linear) for m in q.modules())
    assert n_fp8 == 2 * 10 + 2 * 3                     # double: qkv/proj x2, mlp x4, mod x2; single: lin1/lin2/mod

    opts = EditOptions(num_steps=6, skip_step=1)
    a = KVEditCore(model, "cpu").invert(make_inp(1), opts, H, W)
    b = KVEditCore(q, "cpu").invert(make_inp(1), opts, H, W)
    za = a.feature[f"{a.timesteps[0]}_img"].float()
    zb = b.feature[f"{b.timesteps[0]}_img"].float()
    rel = (za - zb).norm() / za.norm()
    assert 0 < rel < 0.05, rel
