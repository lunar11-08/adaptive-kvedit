"""Stores for KV-Edit's inversion features (per-step latents and per-block K/V)."""
import shutil
import tempfile
from collections.abc import MutableMapping
from pathlib import Path

import torch


class DiskFeatureStore(MutableMapping):
    """Dict-like store that spills tensors to disk (~17 GB per 512px image at 24 steps)."""

    def __init__(self, root: str | Path | None = None):
        if root is not None:
            Path(root).mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix="kvfeat_", dir=root))
        self._keys: dict[str, Path] = {}

    def __setitem__(self, key, value):
        path = self._keys.get(key) or self.root / f"{len(self._keys)}_{abs(hash(key))}.pt"
        torch.save(value.detach().cpu().contiguous(), path)
        self._keys[key] = path

    def __getitem__(self, key):
        return torch.load(self._keys[key], map_location="cpu", weights_only=True)

    def __delitem__(self, key):
        self._keys.pop(key).unlink(missing_ok=True)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def clear(self):
        shutil.rmtree(self.root, ignore_errors=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self._keys.clear()

    def close(self):
        shutil.rmtree(self.root, ignore_errors=True)
        self._keys.clear()


class NullStore(dict):
    """Discards writes; used for probe passes that run the model in 'inverse' mode."""

    def __setitem__(self, key, value):
        pass


class CopyOnRead(MutableMapping):
    """Read-only view that clones CPU tensors, so KV-Edit's in-place K/V writes never touch the cache."""

    def __init__(self, base: MutableMapping, device: torch.device):
        self.base = base
        self.device = torch.device(device)

    def __getitem__(self, key):
        t = self.base[key]
        if t.device == self.device and isinstance(self.base, dict):
            return t.clone()
        return t

    def __setitem__(self, key, value):
        raise RuntimeError("feature cache is read-only during editing")

    def __delitem__(self, key):
        raise RuntimeError("feature cache is read-only during editing")

    def __iter__(self):
        return iter(self.base)

    def __len__(self):
        return len(self.base)


def make_store(kind: str, root: str | None = None) -> MutableMapping:
    if kind == "ram":
        return {}
    if kind == "disk":
        return DiskFeatureStore(root)
    raise ValueError(f"unknown feature store {kind!r}")
