from __future__ import annotations

from typing import Any

from .cost import FALLBACK_RATES, GPU_ALIASES
from .gpu import PRO6000, default_gpu_for_recipe, normalize_gpu
from .recipes import DEFAULT_RECIPE, RECIPES, recipe_uris


ALLOWED_GPUS = (
    {"id": "L40S", "label": "L40S", "vram_gb": 48, "notes": "多数配方的默认卡"},
    {"id": "L4", "label": "L4", "vram_gb": 24, "notes": "大扩散缓冲可能 OOM"},
    {
        "id": PRO6000,
        "label": "RTX PRO 6000",
        "vram_gb": 96,
        "notes": "Ideogram4 / FLUX.2 Dev 默认",
    },
)

RECIPE_CARDS: list[dict[str, Any]] = [
    {
        "id": "z-image-turbo",
        "label": "Z-Image Turbo",
        "label_zh": "Z-Image Turbo",
        "elo": 1131,
        "hint": "Photoreal and text. Official size is 512×1024, 8 steps.",
        "hint_zh": "写实与文字。官方尺寸 512×1024，8 步。",
    },
    {
        "id": "ideogram4",
        "label": "Ideogram 4.0",
        "label_zh": "Ideogram 4.0",
        "elo": 1217,
        "hint": "Prompt must be JSON, e.g. {\"high_level_description\":\"a cat\"}.",
        "hint_zh": "提示词必须是 JSON，例如 {\"high_level_description\":\"一只猫\"}。",
    },
    {
        "id": "flux2-klein",
        "label": "FLUX.2 [klein] 9B",
        "label_zh": "FLUX.2 Klein 9B",
        "elo": 1149,
        "hint": "4 steps, cfg 1.0. Uses Qwen3-8B, not Qwen3-VL.",
        "hint_zh": "4 步，CFG 1.0。文本编码器是 Qwen3-8B，不是 Qwen3-VL。",
    },
    {
        "id": "flux2-dev",
        "label": "FLUX.2 [dev]",
        "label_zh": "FLUX.2 Dev",
        "elo": 1200,
        "hint": "Largest stack. 20 steps, euler, cfg 1.0.",
        "hint_zh": "最大一套。20 步，euler，CFG 1.0。",
    },
    {
        "id": "sdxl-turbo",
        "label": "SDXL-Turbo",
        "label_zh": "SDXL-Turbo",
        "elo": None,
        "hint": "512², 4 steps, cfg 1.0, euler.",
        "hint_zh": "512²，4 步，CFG 1.0，euler。",
    },
    {
        "id": "sd2",
        "label": "Stable Diffusion 2.1",
        "label_zh": "Stable Diffusion 2.1",
        "elo": None,
        "hint": "512², 20 steps, cfg 7.0.",
        "hint_zh": "512²，20 步，CFG 7.0。",
    },
    {
        "id": "sd15",
        "label": "Stable Diffusion 1.5",
        "label_zh": "Stable Diffusion 1.5",
        "elo": None,
        "hint": "512², 20 steps, cfg 7.0.",
        "hint_zh": "512²，20 步，CFG 7.0。",
    },
]


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
                "name": item.get("label") or item["id"],
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
                "default_gpu": default_gpu_for_recipe(card["id"]),
                "uris": recipe_uris(card["id"]),
            }
        )
    return cards


def default_recipe() -> str:
    return DEFAULT_RECIPE


__all__ = [
    "ALLOWED_GPUS",
    "RECIPE_CARDS",
    "default_gpu_for_recipe",
    "default_recipe",
    "gpu_usd_per_hour",
    "list_gpus",
    "list_models",
    "normalize_gpu",
]
