"""PIE-Bench metrics, following PnPInversion's protocol.

Background metrics compare source and edited images with the edit region zeroed
(mask = 1 - gt_edit_mask, border counted as edit). CLIP: whole image vs target
prompt, and edit region only (image * gt_edit_mask).
"""
import numpy as np
import torch


def _t(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img[..., :3]).permute(2, 0, 1).float()[None] / 255.0


def background_tensors(src: np.ndarray, out: np.ndarray, edit_mask: np.ndarray):
    bg = torch.from_numpy(1 - edit_mask.astype(np.float32))[None, None]
    return _t(src) * bg, _t(out) * bg


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a - b) ** 2).item()
    return float("inf") if mse == 0 else 10 * np.log10(1.0 / mse)


def mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.mean((a - b) ** 2).item()


class Evaluator:
    def __init__(self, device="cpu", clip_model="openai/clip-vit-large-patch14", use_clip=True, use_lpips=True):
        from torchmetrics.image import StructuralSimilarityIndexMeasure

        self.device = device
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        self.lpips = None
        if use_lpips:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
        self.clip = None
        if use_clip:
            from torchmetrics.multimodal.clip_score import CLIPScore
            self.clip = CLIPScore(model_name_or_path=clip_model).to(device)

    @torch.no_grad()
    def clip_score(self, img: np.ndarray, prompt: str) -> float:
        x = torch.from_numpy(img[..., :3]).permute(2, 0, 1)[None].to(self.device)
        self.clip.reset()
        return self.clip(x, [prompt]).item()

    @torch.no_grad()
    def __call__(self, src: np.ndarray, out: np.ndarray, edit_mask: np.ndarray, target_prompt: str) -> dict:
        r = {}
        if edit_mask.all():                     # no background (global edits): bg metrics undefined
            r.update(psnr=float("nan"), mse_1e4=float("nan"), ssim_1e2=float("nan"))
            if self.lpips is not None:
                r["lpips_1e3"] = float("nan")
        else:
            self._background(src, out, edit_mask, r)
        if self.clip is not None:
            r["clip_whole"] = self.clip_score(out, target_prompt)
            edited = (out[..., :3] * edit_mask[..., None]).astype(np.uint8)
            r["clip_edited"] = self.clip_score(edited, target_prompt)
        return r

    def _background(self, src, out, edit_mask, r):
        a, b = background_tensors(src, out, edit_mask)
        r.update(psnr=psnr(a, b), mse_1e4=mse(a, b) * 1e4)
        a, b = a.to(self.device), b.to(self.device)
        self.ssim.reset()
        r["ssim_1e2"] = self.ssim(b, a).item() * 1e2
        if self.lpips is not None:
            self.lpips.reset()
            r["lpips_1e3"] = self.lpips(b * 2 - 1, a * 2 - 1).item() * 1e3
