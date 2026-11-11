"""Which Modal GPU Lens runs a model on, by its size.

Torch-free and network-free, so both the caller that sends an experiment to
the GPU app (lens/mcp_server/src/app/remote.py) and job cost estimates read
the same answer. SEESAW_GPU_SMALL / SEESAW_GPU_LARGE override the GPU types.
"""

from __future__ import annotations

import os
import re

# Models are loaded in float32. An A10G (24 GB) holds them, with room for
# activation caches, up to about 4B parameters; past that they go to an A100.
GPU_SMALL: str = os.getenv("SEESAW_GPU_SMALL", "A10G")
GPU_LARGE: str = os.getenv("SEESAW_GPU_LARGE", "A100-80GB")
LARGE_MODEL_PARAMS_B: float = 4.0


def params_b(model_name: str) -> float | None:
    """Billions of parameters, read from the name ('Qwen2.5-7B' → 7.0); None when it names no size.

    Read from the name rather than the config so it needs no download (and no
    token for gated models).
    """
    sizes = [float(x) for x in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)[bB](?![a-zA-Z])", model_name)]
    return max(sizes) if sizes else None


def gpu_for(model_name: str) -> str:
    """The Modal GPU for `model_name`, by the parameter count in its name. Names without a size are small."""
    size = params_b(model_name)
    return GPU_LARGE if size and size > LARGE_MODEL_PARAMS_B else GPU_SMALL
