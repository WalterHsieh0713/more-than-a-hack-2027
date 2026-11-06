"""torch must load before LightGBM in any process that uses both: see src/sequence.py."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch  # noqa: E402, F401
