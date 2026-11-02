"""YAML config loading with dotted CLI overrides (e.g. controller.w_user=0.5)."""
from dataclasses import fields
from pathlib import Path

import yaml

from .controller.adaptive_mask import ControllerConfig
from .models.editing_backbone import EditOptions


def _merge(dst: dict, src: dict) -> dict:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def load_config(path: str | Path, overrides: list[str] = ()) -> dict:
    """overrides: 'a.b=yaml_value' or '@file.yaml' (deep-merged, e.g. runs/dev/tuned.yaml)."""
    cfg = yaml.safe_load(Path(path).read_text())
    for ov in overrides:
        if ov.startswith("@"):
            _merge(cfg, yaml.safe_load(Path(ov[1:]).read_text()))
            continue
        key, val = ov.split("=", 1)
        d = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            d = d.setdefault(p, {})
        d[leaf] = yaml.safe_load(val)
    return cfg


def _build(cls, d: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(d) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**d)


def edit_options(cfg: dict) -> EditOptions:
    return _build(EditOptions, cfg["sampling"])


def controller_config(cfg: dict, method: dict | None = None) -> ControllerConfig:
    d = dict(cfg.get("controller", {}))
    d.update((method or {}).get("controller", {}))
    return _build(ControllerConfig, d)
