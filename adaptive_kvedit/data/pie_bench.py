"""PIE-Bench (PnPInversion release): mapping_file.json + annotation_images/."""
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from ..models.text_encoder import strip_brackets

CATEGORIES = {
    "0": "random", "1": "change_object", "2": "add_object", "3": "delete_object",
    "4": "change_attribute_content", "5": "change_attribute_pose", "6": "change_attribute_color",
    "7": "change_attribute_material", "8": "change_background", "9": "change_style",
}


@dataclass
class PieSample:
    key: str
    image_path: Path
    source_prompt: str          # with [brackets]
    target_prompt: str          # with [brackets]
    instruction: str
    category: str
    blended_words: list[str]
    mask_rle: list[int]

    @property
    def source_clean(self) -> str:
        return strip_brackets(self.source_prompt)

    @property
    def target_clean(self) -> str:
        return strip_brackets(self.target_prompt)

    def load_image(self) -> np.ndarray:
        return np.array(Image.open(self.image_path).convert("RGB"))

    def mask(self, shape=(512, 512), border: bool = False) -> np.ndarray:
        return mask_decode(self.mask_rle, shape, border)


def mask_decode(encoded: list[int], shape=(512, 512), border: bool = False) -> np.ndarray:
    """PnPInversion RLE. The data has a 1-px image frame baked in: border=True keeps it (their eval
    protocol, frame counted as edit region), border=False removes it (the actual object mask)."""
    n = shape[0] * shape[1]
    m = np.zeros(n, dtype=np.uint8)
    for start, length in zip(encoded[0::2], encoded[1::2]):
        m[start:min(start + length, n)] = 1
    m = m.reshape(shape)
    m[0, :] = m[-1, :] = m[:, 0] = m[:, -1] = int(border)
    return m


def load_pie_bench(root: str | Path) -> list[PieSample]:
    root = Path(root)
    mapping = json.loads((root / "mapping_file.json").read_text())
    out = []
    for key, d in mapping.items():
        out.append(PieSample(
            key=key,
            image_path=root / "annotation_images" / d["image_path"],
            source_prompt=d["original_prompt"],
            target_prompt=d["editing_prompt"],
            instruction=d.get("editing_instruction", ""),
            category=str(d["editing_type_id"]),
            blended_words=d.get("blended_word", "").split(),
            mask_rle=d["mask"],
        ))
    return sorted(out, key=lambda s: s.key)


def stratified_subset(samples: list[PieSample], n: int, seed: int = 0) -> list[PieSample]:
    """~n samples spread evenly over editing categories; deterministic."""
    rng = random.Random(seed)
    by_cat: dict[str, list[PieSample]] = {}
    for s in samples:
        by_cat.setdefault(s.category, []).append(s)
    cats = sorted(by_cat)
    for c in cats:
        rng.shuffle(by_cat[c])
    picked, i = [], 0
    while len(picked) < n and any(by_cat.values()):
        c = cats[i % len(cats)]
        if by_cat[c]:
            picked.append(by_cat[c].pop())
        i += 1
    return sorted(picked, key=lambda s: s.key)


def dev_subset(samples: list[PieSample], n_dev: int, n_test: int = 100, seed: int = 0,
               max_mask_frac: float = 0.9) -> list[PieSample]:
    """Tuning images disjoint from stratified_subset(samples, n_test, seed); skips near-full masks."""
    test = {s.key for s in stratified_subset(samples, n_test, seed)}
    rest = [s for s in samples if s.key not in test and s.mask(border=True).mean() < max_mask_frac]
    return stratified_subset(rest, n_dev, seed + 1)
