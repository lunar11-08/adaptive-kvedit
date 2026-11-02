"""One-time: FLUX-dev bf16 checkpoint -> FP8 shards (~12 GB). CPU only, streams one tensor at a time.

python scripts/convert_fp8.py [--src path/to/flux1-dev.safetensors] [--dst checkpoints/flux-dev-fp8]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adaptive_kvedit.models.fp8 import convert_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", help="defaults to the Hugging Face cache copy (downloaded if missing)")
    ap.add_argument("--dst", default="checkpoints/flux-dev-fp8")
    args = ap.parse_args()
    src = args.src
    if src is None:
        from huggingface_hub import hf_hub_download
        src = hf_hub_download("black-forest-labs/FLUX.1-dev", "flux1-dev.safetensors")
    t0 = time.time()
    out = convert_checkpoint(src, args.dst)
    size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 2**30
    print(f"wrote {out} ({size:.1f} GiB) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
