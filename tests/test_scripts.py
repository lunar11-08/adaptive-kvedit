"""Runs the batch/metrics/table/figure scripts on a fake 2-image PIE-Bench with the tiny model."""
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from test_cpu_pipeline import DT, H, W, make_inp, tiny_model  # noqa: E402

from adaptive_kvedit.models.editing_backbone import EditingBackbone, KVEditCore  # noqa: E402


class FakeTok:
    def __call__(self, text, add_special_tokens=False):
        return SimpleNamespace(input_ids=[0] * sum(math.ceil(len(w) / 3) for w in text.split()))


class FakeBackbone(KVEditCore):
    crop16 = staticmethod(EditingBackbone.crop16)

    def __init__(self, **kw):
        super().__init__(tiny_model(), "cpu", feature_store=kw.get("feature_store", "ram"))
        self.tokenizer = FakeTok()

    def cache_prompts(self, prompts):
        pass

    def load_model(self):
        return self.model

    def encode(self, image):
        return make_inp(0)["img"]

    def prepare(self, latent, prompt):
        inp = make_inp(abs(hash(prompt)) % 1000)
        inp["img"] = latent
        return inp

    def decode(self, z, h, w):
        x = torch.nn.functional.interpolate(z.float().mean(-1).reshape(1, 1, 8, 8), size=(h, w))
        return ((x[0, 0].tanh() + 1) * 127.5).byte().numpy()[..., None].repeat(3, -1)


def fake_pie(root: Path):
    (root / "annotation_images" / "0_random").mkdir(parents=True)
    mapping = {}
    for i in range(2):
        key = f"00000000000{i}"
        Image.fromarray(np.random.default_rng(i).integers(0, 255, (H, W, 3), dtype=np.uint8)).save(
            root / "annotation_images" / "0_random" / f"{key}.jpg")
        start = 40 * W
        rle = [s for r in range(40) for s in (start + r * W + 40, 40)]
        mapping[key] = {"image_path": f"0_random/{key}.jpg", "original_prompt": "a [cat] on a sofa",
                        "editing_prompt": "a [dog] on a sofa", "editing_instruction": "", "editing_type_id": str(i),
                        "blended_word": "cat dog", "mask": rle}
    (root / "mapping_file.json").write_text(json.dumps(mapping))


def test_scripts_end_to_end(tmp_path, monkeypatch):
    import comparison
    import compute_metrics
    import make_figures
    import run_batch
    import visualize_attention
    import yaml

    fake_pie(tmp_path / "pie")
    cfg = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs" / "default.yaml").read_text())
    cfg["pie_root"] = str(tmp_path / "pie")
    cfg["sampling"].update(num_steps=6, skip_step=1)
    cfg["attention"]["probe_steps"] = [0, 2]
    cfg["controller"]["min_tokens"] = 1
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"

    monkeypatch.setattr(run_batch, "EditingBackbone", FakeBackbone)
    argv = ["run_batch", "--config", str(cfg_path), "--out", str(run), "--limit", "2",
            "--methods", "kvedit", "attn", "fused", "fused_sched", "--noise", "none", "dilate", "box"]
    monkeypatch.setattr(sys, "argv", argv)
    run_batch.main()
    outs = sorted(p.name for p in (run / "000000000000").glob("*.png"))
    assert "attn__none.png" in outs and "attn__dilate.png" not in outs
    assert len([o for o in outs if o != "source.png"]) == 3 * 3 + 1
    meta = json.loads((run / "000000000000" / "meta.json").read_text())
    assert meta["token_prompt"] == "target" and meta["token_idx"]

    mtime = (run / "000000000000" / "kvedit__none.png").stat().st_mtime
    run_batch.main()                                    # resume: nothing recomputed
    assert (run / "000000000000" / "kvedit__none.png").stat().st_mtime == mtime

    monkeypatch.setattr(sys, "argv", ["cm", "--run", str(run), "--config", str(cfg_path), "--no-clip", "--no-lpips"])
    compute_metrics.main()
    monkeypatch.setattr(sys, "argv", ["cmp", str(run / "metrics.csv"), "--by-category"])
    comparison.main()
    assert "fused_sched" in (run / "results.md").read_text()
    monkeypatch.setattr(sys, "argv", ["va", "--run", str(run)])
    visualize_attention.main()
    assert (run / "000000000000" / "viz_layers.png").exists()
    monkeypatch.setattr(sys, "argv", ["mf", "--run", str(run), "--noise", "dilate", "--methods", "kvedit", "fused_sched"])
    make_figures.main()
    assert (run / "figure_dilate.png").exists()
