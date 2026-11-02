"""Adaptive feature-guided masking on top of KV-Edit."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KVEDIT_DIR = ROOT / "third_party" / "KV-Edit"
if str(KVEDIT_DIR) not in sys.path:
    sys.path.insert(0, str(KVEDIT_DIR))
