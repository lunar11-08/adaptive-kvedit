"""Fuse the user mask with FLUX attention on the token grid."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class ControllerConfig:
    w_user: float = 0.4          # score = w_user * user + (1 - w_user) * attn
    tau_tight: float = 0.5       # final-step threshold on the score
    tau_loose: float = 0.3       # early-step threshold (schedule only)
    dilate_tokens: int = 3       # attention may grow the region at most this far past the user mask
    smooth: int = 3              # box blur on attention, in tokens (1 = off)
    norm_quantile: float = 0.98
    attn_only_tau: float | None = None   # threshold for attention-only mode; None = Otsu
    close_tokens: int = 1        # morphological closing radius on final masks
    min_tokens: int = 4          # fall back to the user mask below this
    sched_start: float = 0.2     # fraction of steps held at tau_loose
    sched_end: float = 0.7       # fraction of steps where tau_tight is reached


def dilate(m: Tensor, r: int) -> Tensor:
    if r <= 0:
        return m
    return F.max_pool2d(m[None, None].float(), 2 * r + 1, stride=1, padding=r)[0, 0]


def erode(m: Tensor, r: int) -> Tensor:
    return 1 - dilate(1 - m.float(), r)


def close(m: Tensor, r: int) -> Tensor:
    return erode(dilate(m, r), r)


def box_blur(x: Tensor, k: int) -> Tensor:
    if k <= 1:
        return x
    return F.avg_pool2d(x[None, None], k, stride=1, padding=k // 2, count_include_pad=False)[0, 0]


def normalize_attention(attn: Tensor, cfg: ControllerConfig) -> Tensor:
    a = box_blur(attn.float(), cfg.smooth)
    lo, hi = a.min(), torch.quantile(a.flatten(), cfg.norm_quantile)
    return ((a - lo) / (hi - lo).clamp_min(1e-8)).clamp(0, 1)


def otsu(x: Tensor, bins: int = 64) -> float:
    hist = torch.histc(x.flatten().float(), bins=bins, min=0, max=1)
    p = hist / hist.sum()
    centers = (torch.arange(bins, dtype=torch.float32) + 0.5) / bins
    w0 = p.cumsum(0)
    mu = (p * centers).cumsum(0)
    between = (mu[-1] * w0 - mu) ** 2 / (w0 * (1 - w0)).clamp_min(1e-12)
    return float(centers[between.argmax()])


def fused_score(user: Tensor, attn_n: Tensor, cfg: ControllerConfig) -> Tensor:
    """Score in [0,1]; zero outside the dilated user mask."""
    allowed = dilate(user, cfg.dilate_tokens)
    return (cfg.w_user * user.float() + (1 - cfg.w_user) * attn_n) * allowed


def finalize(m: Tensor, user: Tensor, cfg: ControllerConfig, fill_within: Tensor | None = None) -> Tensor:
    """Fill holes (closing); with fill_within, filled tokens must lie inside it."""
    m = m.float()
    filled = close(m, cfg.close_tokens)
    m = filled if fill_within is None else torch.maximum(m, filled * fill_within.float())
    if m.sum() < cfg.min_tokens:
        m = user.float()
    if m.all():
        m = user.float()
    return (m > 0.5).float()


def attention_only_mask(attn_n: Tensor, user: Tensor, cfg: ControllerConfig) -> Tensor:
    tau = cfg.attn_only_tau if cfg.attn_only_tau is not None else otsu(attn_n)
    return finalize(attn_n > tau, user, cfg)
