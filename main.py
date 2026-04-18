"""
CS2 Texture Upscaler — AI upscaling for Counter-Strike 2 materials
Launch with: python main.py
"""

import sys
import os
import time
import warnings

# Suppress triton CUDA-not-found warning (PyTorch bundles its own CUDA runtime;
# triton only needs the full toolkit for the inductor backend which we guard).
warnings.filterwarnings("ignore", message="Failed to find CUDA", module="triton")

# Ensure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_t0 = time.perf_counter()


def _elapsed():
    return f"[{time.perf_counter() - _t0:5.1f}s]"


print(f"{_elapsed()} CS2 Texture Upscaler starting...")
print(f"{_elapsed()} Loading PyTorch / torchvision...", flush=True)

# Apply compatibility patches before any other imports
from src.core.compat import apply_patches
apply_patches()

print(f"{_elapsed()} Loading AI engine (basicsr / Real-ESRGAN)...", flush=True)
from src.app import run

print(f"{_elapsed()} Launching UI...", flush=True)

if __name__ == "__main__":
    run()
