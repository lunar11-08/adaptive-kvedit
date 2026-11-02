import sys
import types

try:
    import imwatermark  # noqa: F401
except ImportError:  # only used by flux.util's watermark helper, never by these tests
    m = types.ModuleType("imwatermark")
    m.WatermarkEncoder = type("WatermarkEncoder", (), {"set_watermark": lambda *a: None})
    sys.modules["imwatermark"] = m
