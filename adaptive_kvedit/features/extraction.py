"""Image->edit-token attention maps from FLUX's joint attention.

Wraps flux.modules.layers.attention (both block types call it with q/k = [txt; img]).
For each image query, softmax over all keys, then keep the columns of the edit tokens.
Maps for every (step, block) are kept so layer/step selection can be redone offline.
"""
import contextlib
import math

import torch
from torch import Tensor

import adaptive_kvedit  # noqa: F401
import flux.modules.layers as layers
from flux.math import apply_rope

TXT_LEN = 512


class AttentionRecorder:
    def __init__(self, model, token_idx: list[int], head_chunk: int = 4):
        self.model = model
        self.token_idx = torch.as_tensor(token_idx, dtype=torch.long)
        self.head_chunk = head_chunk
        self.maps: dict[tuple[int, str, int], Tensor] = {}   # (step, 'double'|'single', block) -> (N_img,)
        self._step = None
        self._block = None
        self._hooks = []

    # -- hooks --
    def _register(self):
        for kind, blocks in (("double", self.model.double_blocks), ("single", self.model.single_blocks)):
            for i, blk in enumerate(blocks):
                def pre(m, args, kind=kind, i=i):
                    self._block = (kind, i)
                self._hooks.append(blk.register_forward_pre_hook(pre))

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @contextlib.contextmanager
    def capture(self, step: int):
        if len(self.token_idx) == 0:
            yield
            return
        orig = layers.attention
        self._step = step
        self._register()

        def wrapped(q, k, v, pe, pe_q=None, attention_mask=None):
            if pe_q is None and self._block is not None:
                self._record(q, k, pe)
            return orig(q, k, v, pe, pe_q=pe_q, attention_mask=attention_mask)

        layers.attention = wrapped
        try:
            yield
        finally:
            layers.attention = orig
            self._remove()
            self._step = self._block = None

    @torch.no_grad()
    def _record(self, q: Tensor, k: Tensor, pe: Tensor):
        q, k = apply_rope(q, k, pe)
        B, H, Lq, D = q.shape
        idx = self.token_idx.to(q.device)
        scale = 1.0 / math.sqrt(D)
        acc = torch.zeros(B, Lq - TXT_LEN, device=q.device, dtype=torch.float32)
        for h0 in range(0, H, self.head_chunk):
            qi = q[:, h0:h0 + self.head_chunk, TXT_LEN:].float()
            kh = k[:, h0:h0 + self.head_chunk].float()
            p = torch.softmax(qi @ kh.transpose(-1, -2) * scale, dim=-1)    # (B,h,N_img,L_all)
            acc += p[..., idx].mean(-1).sum(1)
        self.maps[(self._step, *self._block)] = (acc[0] / H).cpu()

    # -- aggregation --
    def aggregate(self, steps=None, kinds=("double", "single"), blocks=None) -> Tensor:
        """Mean of selected maps -> (N_img,). `blocks` maps kind -> iterable of block ids (None = all)."""
        sel = [m for (s, kd, b), m in self.maps.items()
               if (steps is None or s in steps) and kd in kinds
               and (blocks is None or blocks.get(kd) is None or b in blocks[kd])]
        if not sel:
            raise ValueError("no attention maps match the selection")
        return torch.stack(sel).mean(0)

    def state_dict(self) -> dict:
        return {"token_idx": self.token_idx.tolist(),
                "maps": {f"{s}|{kd}|{b}": m for (s, kd, b), m in self.maps.items()}}

    @classmethod
    def from_state_dict(cls, sd: dict) -> "AttentionRecorder":
        r = cls(model=None, token_idx=sd["token_idx"])
        for key, m in sd["maps"].items():
            s, kd, b = key.split("|")
            r.maps[(int(s), kd, int(b))] = m
        return r


def select_from_config(rec: AttentionRecorder, cfg: dict) -> Tensor:
    """cfg: {'steps': [..]|None, 'double': [..]|'all'|'none', 'single': [..]|'all'|'none'}."""
    kinds, blocks = [], {}
    for kd in ("double", "single"):
        v = cfg.get(kd, "all")
        if v == "none":
            continue
        kinds.append(kd)
        blocks[kd] = None if v == "all" else set(v)
    return rec.aggregate(steps=cfg.get("steps"), kinds=tuple(kinds), blocks=blocks)
