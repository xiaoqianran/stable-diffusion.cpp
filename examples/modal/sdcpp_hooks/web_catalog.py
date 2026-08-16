from __future__ import annotations

from typing import Any

from .cost import FALLBACK_RATES, GPU_ALIASES
from .recipes import RECIPES, recipe_uris


ALLOWED_GPUS = (
    {"id": "L40S", "vram_gb": 48, "notes": "default; needed for Ideogram4 / FLUX.2"},
    {"id": "L4", "vram_gb": 24, "notes": "can OOM on large diffusion buffers"},
    {"id": "RTX6000", "vram_gb": 48, "notes": "RTX PRO 6000"},
)

RECIPE_CARDS: list[dict[str, Any]] = [
    {
        "id": "z-image-turbo",
        "label": "Z-Image Turbo",
        "elo": 1131,
        "hint": "Photoreal and text. Official size is 512×1024, 8 steps.",
    },
    {
        "id": "ideogram4",
        "label": "Ideogram 4.0",
        "elo": 1217,
        "hint": "Prompt must be JSON, e.g. {\"high_level_description\":\"a cat\"}.",
    },
    {
        "id": "flux2-klein",
        "label": "FLUX.2 [klein] 9B",
        "elo": 1149,
        "hint": "4 steps, cfg 1.0. Uses Qwen3-8B, not Qwen3-VL.",
    },
    {
        "id": "flux2-dev",
        "label": "FLUX.2 [dev]",
        "elo": 1200,
        "hint": "Largest stack. 20 steps, euler, cfg 1.0.",
    },
    {
        "id": "sdxl-turbo",
        "label": "SDXL-Turbo",
        "elo": None,
        "hint": "512², 4 steps, cfg 1.0, euler.",
    },
    {
        "id": "sd2",
        "label": "Stable Diffusion 2.1",
        "elo": None,
        "hint": "512², 20 steps, cfg 7.0.",
    },
    {
        "id": "sd15",
        "label": "Stable Diffusion 1.5",
        "elo": None,
        "hint": "512², 20 steps, cfg 7.0.",
    },
]


def normalize_gpu(name: str) -> str:
    text = (name or "L40S").strip().upper().replace(" ", "-")
    aliases = {
        "RTX-PRO-6000": "RTX6000",
        "RTXPRO6000": "RTX6000",
        "PRO-6000": "RTX6000",
        "PRO6000": "RTX6000",
    }
    gpu = aliases.get(text, text)
    if gpu.startswith("A10") or gpu.startswith("A100"):
        raise ValueError(f"GPU {name!r} is blocked; use L4, L40S, or RTX6000")
    allowed = {item["id"] for item in ALLOWED_GPUS}
    if gpu not in allowed:
        raise ValueError(f"unknown GPU {name!r}; known: {', '.join(sorted(allowed))}")
    return gpu


def gpu_usd_per_hour(gpu: str) -> float:
    key = "gpu_hour_cost_" + GPU_ALIASES.get(gpu, gpu).lower().replace("-", "_")
    return float(FALLBACK_RATES.get(key, "1.95000"))


def list_gpus() -> list[dict[str, Any]]:
    rows = []
    for item in ALLOWED_GPUS:
        usd = gpu_usd_per_hour(item["id"])
        rows.append(
            {
                **item,
                "name": item["id"],
                "usd_per_hour": usd,
                "usd_per_second": usd / 3600.0,
            }
        )
    return rows


def list_models() -> list[dict[str, Any]]:
    cards = []
    for card in RECIPE_CARDS:
        recipe = RECIPES[card["id"]]
        cards.append(
            {
                **card,
                "name": card["label"],
                "width": recipe.get("width"),
                "height": recipe.get("height"),
                "default_steps": recipe.get("steps"),
                "cfg_scale": recipe.get("cfg_scale"),
                "sampling_method": recipe.get("sampling_method") or "",
                "uris": recipe_uris(card["id"]),
            }
        )
    return cards


def default_recipe() -> str:
    return "z-image-turbo"
