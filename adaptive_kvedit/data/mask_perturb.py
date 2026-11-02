"""Noisy-mask stress test: dilated, eroded and box-only user masks."""
import numpy as np
from scipy import ndimage

NOISE_KINDS = ("none", "dilate", "erode", "box")


def _disk(r: int) -> np.ndarray:
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def perturb_mask(mask: np.ndarray, kind: str, radius: int = 24) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if kind == "none":
        return m
    if kind == "dilate":
        return ndimage.binary_dilation(m, _disk(radius)).astype(np.uint8)
    if kind == "erode":
        e = ndimage.binary_erosion(m, _disk(radius)).astype(np.uint8)
        return e if e.any() else m
    if kind == "box":
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            return m
        b = np.zeros_like(m)
        b[ys.min():ys.max() + 1, xs.min():xs.max() + 1] = 1
        return b
    raise ValueError(f"unknown mask noise {kind!r}; expected one of {NOISE_KINDS}")
