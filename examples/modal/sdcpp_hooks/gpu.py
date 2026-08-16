"""Modal GPU names and per-recipe defaults.

Modal's live shortcode for the 96 GB Blackwell card is `RTX-PRO-6000`.
The older `RTX6000` alias is accepted here and rewritten to that name.
"""

from __future__ import annotations


PRO6000 = "RTX-PRO-6000"
DEFAULT_GPU = "L40S"
ALLOWED_GPU_IDS = ("L40S", "L4", PRO6000)
HEAVY_RECIPES = frozenset({"ideogram4", "flux2-dev"})

_ALIASES = {
    "RTX-PRO-6000": PRO6000,
    "RTXPRO6000": PRO6000,
    "RTX6000": PRO6000,
    "PRO-6000": PRO6000,
    "PRO6000": PRO6000,
}


def normalize_gpu(name: str) -> str:
    text = (name or DEFAULT_GPU).strip().upper().replace(" ", "-")
    gpu = _ALIASES.get(text, text)
    if gpu.startswith("A10") or gpu.startswith("A100"):
        raise ValueError(f"GPU {name!r} is blocked; use L4, L40S, or RTX-PRO-6000")
    if gpu not in ALLOWED_GPU_IDS:
        raise ValueError(f"unknown GPU {name!r}; known: {', '.join(ALLOWED_GPU_IDS)}")
    return gpu


def default_gpu_for_recipe(recipe: str) -> str:
    if recipe in HEAVY_RECIPES:
        return PRO6000
    return DEFAULT_GPU
