"""CPU checks against the real KV-Edit code using a tiny random Flux_kv (no weights needed)."""
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import adaptive_kvedit  # noqa: F401
import flux.modules.layers as layers
from flux.math import apply_rope
from flux.model import Flux_kv, FluxParams
from flux.sampling import unpack
from models.kv_edit import Flux_kv_edit

from adaptive_kvedit.controller import build_schedule
from adaptive_kvedit.controller.adaptive_mask import ControllerConfig
from adaptive_kvedit.data.mask_perturb import perturb_mask
from adaptive_kvedit.data.pie_bench import PieSample, mask_decode, stratified_subset
from adaptive_kvedit.evaluation.metrics import Evaluator
from adaptive_kvedit.features.extraction import AttentionRecorder
from adaptive_kvedit.models.editing_backbone import EditOptions, KVEditCore, packed_to_token_mask
from adaptive_kvedit.models.text_encoder import bracketed_word_positions, edit_token_indices, strip_brackets

H = W = 128            # latent 16x16 -> 8x8 = 64 tokens
DT = torch.bfloat16


def tiny_model(seed=0):
    torch.manual_seed(seed)
    p = FluxParams(in_channels=64, vec_in_dim=32, context_in_dim=32, hidden_size=64, mlp_ratio=2.0,
                   num_heads=2, depth=2, depth_single_blocks=2, axes_dim=[8, 12, 12], theta=10_000,
                   qkv_bias=True, guidance_embed=True)
    return Flux_kv(p).to(DT).eval()


def make_inp(seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.zeros(8, 8, 3)
    ids[..., 1] += torch.arange(8)[:, None]
    ids[..., 2] += torch.arange(8)[None, :]
    return {"img": torch.randn(1, 64, 64, generator=g).to(DT),
            "img_ids": ids.reshape(1, 64, 3),
            "txt": torch.randn(1, 512, 32, generator=g).to(DT),
            "txt_ids": torch.zeros(1, 512, 3),
            "vec": torch.randn(1, 32, generator=g).to(DT)}


def square_mask(y0=32, y1=80, x0=48, x1=96):
    m = torch.zeros(1, 1, H, W)
    m[..., y0:y1, x0:x1] = 1
    return m


@pytest.fixture(scope="module")
def setup():
    model = tiny_model()
    core = KVEditCore(model, "cpu")
    opts = EditOptions(num_steps=6, skip_step=1, attn_scale=1.0)
    src, tgt = make_inp(1), make_inp(2)
    tgt["img"] = src["img"].clone()
    state = core.invert(src, opts, H, W)
    return SimpleNamespace(model=model, core=core, opts=opts, src=src, tgt=tgt, state=state)


def n_steps(state):
    return len(state.timesteps) - 1


@pytest.mark.parametrize("full", [False, True])
def test_matches_original_kvedit(setup, full):
    s = setup
    mask = torch.ones_like(square_mask()) if full else square_mask()
    sched, _ = build_schedule("user", mask, None, H, W, n_steps(s.state), ControllerConfig())
    ours = s.core.edit(s.state, dict(s.tgt), sched, s.opts)

    orig = Flux_kv_edit.__new__(Flux_kv_edit)
    torch.nn.Module.__init__(orig)
    orig.model, orig.name, orig.device = s.model, "flux-dev", "cpu"
    ns = SimpleNamespace(height=H, width=W, denoise_num_steps=6, skip_step=1, re_init=False, attn_scale=1.0,
                         denoise_guidance=5.5, inversion_guidance=1.5, attn_mask=False)
    z0, zt, info = orig.inverse(dict(s.src), mask.to(DT), ns)
    for k in s.state.feature:                         # inversion caches match
        assert torch.equal(info["feature"][k], s.state.feature[k]), k
    theirs = orig.denoise(z0, zt, dict(s.tgt), mask.to(DT), ns, info)
    assert torch.equal(unpack(ours.float(), H, W), theirs)


def test_background_untouched_and_cache_reusable(setup):
    s = setup
    sched, _ = build_schedule("user", square_mask(), None, H, W, n_steps(s.state), ControllerConfig())
    a = s.core.edit(s.state, dict(s.tgt), sched, s.opts)
    b = s.core.edit(s.state, dict(s.tgt), sched, s.opts)
    assert torch.equal(a, b), "cache was mutated by the first edit"
    bg = ~packed_to_token_mask(sched[0])[0]
    assert torch.equal(a[:, bg], s.state.z0[:, bg])
    assert not torch.equal(a[:, ~bg], s.state.z0[:, ~bg])


def test_disk_store_matches_ram(setup, tmp_path):
    s = setup
    disk = KVEditCore(s.model, "cpu", feature_store="disk", feature_dir=str(tmp_path))
    st = disk.invert(s.src, s.opts, H, W)
    sched, _ = build_schedule("user", square_mask(), None, H, W, n_steps(st), ControllerConfig())
    assert torch.equal(disk.edit(st, dict(s.tgt), sched, s.opts), s.core.edit(s.state, dict(s.tgt), sched, s.opts))
    root = st.feature.root
    st.close()
    assert not root.exists()


def test_recorder_matches_manual_softmax_and_is_passive(setup):
    s = setup
    tok = [3, 4, 10]
    rec = AttentionRecorder(s.model, tok, head_chunk=1)
    grabbed = []
    t = s.state.timesteps[1]
    z = s.state.feature[f"{t}_img"]
    kw = dict(img=z, img_ids=s.tgt["img_ids"], txt=s.tgt["txt"], txt_ids=s.tgt["txt_ids"], y=s.tgt["vec"],
              timesteps=torch.full((1,), t, dtype=DT), guidance=torch.full((1,), 5.5, dtype=DT))
    with torch.inference_mode():
        ref = s.model(**kw, info={"inverse": True, "feature": {}, "t": t})
        with rec.capture(step=0):
            inner = layers.attention

            def spy(q, k, v, pe, **kws):
                grabbed.append((q, k, pe))
                return inner(q, k, v, pe, **kws)

            layers.attention = spy
            try:
                out = s.model(**kw, info={"inverse": True, "feature": {}, "t": t})
            finally:
                layers.attention = inner
    assert torch.equal(out, ref)
    assert len(rec.maps) == 4                          # 2 double + 2 single blocks
    q, k, pe = grabbed[0]
    q, k = apply_rope(q, k, pe)
    p = torch.softmax(q[:, :, 512:].float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1]), -1)
    manual = p[..., tok].mean(-1).mean(1)[0]
    torch.testing.assert_close(rec.maps[(0, "double", 0)], manual)
    rt = AttentionRecorder.from_state_dict(rec.state_dict())
    torch.testing.assert_close(rt.aggregate(), rec.aggregate())


def test_adaptive_pipeline_end_to_end(setup):
    s = setup
    rec = AttentionRecorder(s.model, [5, 6])
    s.core.probe(s.state, s.tgt, [0, 2], rec, guidance=5.5)
    attn = rec.aggregate()
    assert attn.shape == (64,)
    n = n_steps(s.state)
    for mode in ("attn", "fused", "fused_sched"):
        sched, dbg = build_schedule(mode, square_mask(), attn, H, W, n, ControllerConfig(min_tokens=1))
        assert len(sched) == n
        toks = [packed_to_token_mask(m)[0] for m in sched]
        for a, b in zip(toks, toks[1:]):
            assert not (b & ~a).any(), "schedule must shrink monotonically"
        z = s.core.edit(s.state, dict(s.tgt), sched, s.opts)
        keep = ~toks[-1]
        assert torch.equal(z[:, keep], s.state.z0[:, keep])


def test_fusion_shrinks_dilated_mask_toward_attention():
    obj = torch.zeros(16, 16)
    obj[5:10, 5:10] = 1
    dilated = torch.zeros(16, 16)
    dilated[2:13, 2:13] = 1
    attn = obj * 1.0 + 0.02 * torch.rand(16, 16)
    px = torch.nn.functional.interpolate(dilated[None, None], size=(256, 256), mode="nearest")
    cfg = ControllerConfig(smooth=1, close_tokens=0)
    _, dbg = build_schedule("fused", px, attn.flatten(), 256, 256, 3, cfg)
    final = dbg["final"]

    def iou(a, b):
        return float((a * b).sum() / ((a + b) > 0).sum())

    assert iou(final, obj) > iou(dilated, obj)
    assert iou(final, obj) > 0.8


def test_text_token_indices():
    class Tok:                                         # 1 token per 3 characters, sentencepiece-like
        def __call__(self, text, add_special_tokens=False):
            return SimpleNamespace(input_ids=[0] * sum(math.ceil(len(w) / 3) for w in text.split()))

    p = "a slanted [rusty red] mountain bicycle"
    assert strip_brackets(p) == "a slanted rusty red mountain bicycle"
    assert bracketed_word_positions(p) == [2, 3]
    # tokens: a(0) slanted(1-3) rusty(4-5) red(6) mountain(7-9) bicycle(10-12)
    assert edit_token_indices(Tok(), p) == [4, 5, 6]
    assert edit_token_indices(Tok(), p, ["bicycle"]) == [4, 5, 6, 10, 11, 12]


def test_pie_mask_and_perturbations():
    m = mask_decode([10, 5, 100, 3], (16, 16))
    assert m.sum() == 8 and m.flat[10:15].all()
    assert mask_decode([], (16, 16), border=True).sum() == 60
    base = np.zeros((64, 64), np.uint8)
    base[20:40, 25:35] = 1
    assert perturb_mask(base, "dilate", 4).sum() > base.sum()
    assert 0 < perturb_mask(base, "erode", 3).sum() < base.sum()
    box = perturb_mask(base, "box")
    assert box.sum() == base.sum()                     # rectangle already a box
    samples = [PieSample(str(i).zfill(4), Path("x"), "", "", "", str(i % 10), [], []) for i in range(100)]
    sub = stratified_subset(samples, 30)
    assert len(sub) == 30 and len({s.category for s in sub}) == 10


def test_background_metrics():
    rng = np.random.default_rng(0)
    src = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    out = src.copy()
    mask = np.zeros((64, 64), np.uint8)
    mask[10:30, 10:30] = 1
    out[10:30, 10:30] = 0                              # change only inside the edit region
    ev = Evaluator("cpu", use_clip=False, use_lpips=False)
    r = ev(src, out, mask, "x")
    assert r["psnr"] == float("inf") and r["mse_1e4"] == 0 and r["ssim_1e2"] > 99.9
