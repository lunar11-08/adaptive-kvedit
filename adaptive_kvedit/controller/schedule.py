"""Per-step mask schedules for the editing loop (loose early, tight late)."""
import torch
from torch import Tensor

from .adaptive_mask import ControllerConfig, finalize


def threshold_schedule(n_steps: int, tau_loose: float, tau_tight: float, start: float, end: float) -> list[float]:
    taus = []
    for i in range(n_steps):
        f = i / max(n_steps - 1, 1)
        s = min(max((f - start) / max(end - start, 1e-8), 0.0), 1.0)
        taus.append(tau_loose + (tau_tight - tau_loose) * s)
    return taus


def nested_masks(score: Tensor, user: Tensor, taus: list[float], cfg: ControllerConfig) -> list[Tensor]:
    """Binary masks score > tau_i, forced to shrink monotonically so the regenerated set is fixed."""
    out, prev = [], None
    for tau in taus:
        m = finalize(score > tau, user, cfg)
        if prev is not None:
            m = m * prev
            if m.sum() == 0:
                m = prev
        out.append(m)
        prev = m
    return out


def constant(mask: Tensor, n_steps: int) -> list[Tensor]:
    return [mask] * n_steps
