"""Heatmap overlays and per-layer / per-step grids for the attention sanity check."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


def to_grid(token_map: torch.Tensor, token_hw: tuple[int, int]) -> np.ndarray:
    return token_map.reshape(token_hw).float().cpu().numpy()


def upsample(grid: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    t = torch.from_numpy(np.ascontiguousarray(grid))[None, None].float()
    return F.interpolate(t, size=hw, mode="bilinear", align_corners=False)[0, 0].numpy()


def normalize(x: np.ndarray, q: float = 0.99) -> np.ndarray:
    lo, hi = x.min(), np.quantile(x, q)
    return np.clip((x - lo) / max(hi - lo, 1e-8), 0, 1)


def overlay(image: np.ndarray, grid: np.ndarray, alpha: float = 0.5, cmap: str = "jet") -> np.ndarray:
    heat = plt.get_cmap(cmap)(normalize(upsample(grid, image.shape[:2])))[..., :3]
    return ((1 - alpha) * image[..., :3] / 255.0 + alpha * heat).clip(0, 1)


def contour(ax, mask: np.ndarray, color: str):
    if mask is not None and mask.any() and not mask.all():
        ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=1.2)


def plot_layer_grid(image, rec, token_hw, out_path, step=None, user_mask=None, title=""):
    """One panel per block (double then single) for a single probe step (or mean over steps)."""
    keys = sorted({(kd, b) for (_, kd, b) in rec.maps}, key=lambda x: (x[0] != "double", x[1]))
    steps = None if step is None else [step]
    n = len(keys) + 1
    cols = 10
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.6, rows * 1.7))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(image)
    contour(axes[0], user_mask, "lime")
    axes[0].set_title("input", fontsize=7)
    for ax, (kd, b) in zip(axes[1:], keys):
        m = rec.aggregate(steps=steps, kinds=(kd,), blocks={kd: {b}})
        ax.imshow(overlay(image, to_grid(m, token_hw)))
        contour(ax, user_mask, "white")
        ax.set_title(f"{kd[0].upper()}{b}", fontsize=7)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_step_grid(image, rec, token_hw, out_path, user_mask=None, title=""):
    steps = sorted({s for (s, _, _) in rec.maps})
    fig, axes = plt.subplots(1, len(steps) + 1, figsize=(2.2 * (len(steps) + 1), 2.4))
    axes = np.atleast_1d(axes)
    axes[0].imshow(image)
    contour(axes[0], user_mask, "lime")
    axes[0].set_title("input", fontsize=8)
    for ax, s in zip(axes[1:], steps):
        ax.imshow(overlay(image, to_grid(rec.aggregate(steps=[s]), token_hw)))
        contour(ax, user_mask, "white")
        ax.set_title(f"step {s}", fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def plot_masks(image, panels: dict, out_path, title=""):
    """panels: name -> (H,W) or token-grid array; shown as overlays."""
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(2.2 * (len(panels) + 1), 2.4))
    axes[0].imshow(image)
    axes[0].set_title("input", fontsize=8)
    for ax, (name, m) in zip(axes[1:], panels.items()):
        ax.imshow(overlay(image, np.asarray(m, dtype=float), alpha=0.45, cmap="magma"))
        ax.set_title(name, fontsize=8)
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
