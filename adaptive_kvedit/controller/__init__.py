"""Builds the per-step packed mask list for one editing method."""
from torch import Tensor

from ..models.editing_backbone import pack_pixel_mask, packed_to_token_mask, token_mask_to_packed
from .adaptive_mask import ControllerConfig, attention_only_mask, finalize, fused_score, normalize_attention
from .schedule import constant, nested_masks, threshold_schedule

MODES = ("user", "attn", "fused", "fused_sched")


def build_schedule(mode: str, user_pixel_mask: Tensor, attn_map: Tensor | None, height: int, width: int,
                   n_steps: int, cfg: ControllerConfig) -> tuple[list[Tensor], dict]:
    """Returns (packed masks per step, debug token grids).

    'user' uses KV-Edit's own pixel packing, so it is the exact baseline.
    """
    th, tw = height // 16, width // 16
    user_packed = pack_pixel_mask(user_pixel_mask, height, width)
    user_tok = packed_to_token_mask(user_packed).reshape(th, tw).float()
    debug = {"user": user_tok}

    if mode == "user":
        return constant(user_packed, n_steps), debug
    if attn_map is None:
        raise ValueError(f"mode {mode!r} needs an attention map")

    attn_n = normalize_attention(attn_map.reshape(th, tw).to(user_tok), cfg)
    debug["attn"] = attn_n
    if mode == "attn":
        m = attention_only_mask(attn_n, user_tok, cfg)
        debug["final"] = m
        return constant(token_mask_to_packed(m), n_steps), debug

    score = fused_score(user_tok, attn_n, cfg)
    debug["score"] = score
    if mode == "fused":
        m = finalize(score > cfg.tau_tight, user_tok, cfg)
        debug["final"] = m
        return constant(token_mask_to_packed(m), n_steps), debug
    if mode == "fused_sched":
        taus = threshold_schedule(n_steps, cfg.tau_loose, cfg.tau_tight, cfg.sched_start, cfg.sched_end)
        masks = nested_masks(score, user_tok, taus, cfg)
        debug["loose"], debug["final"] = masks[0], masks[-1]
        return [token_mask_to_packed(m) for m in masks], debug
    raise ValueError(f"unknown mode {mode!r}; expected one of {MODES}")
