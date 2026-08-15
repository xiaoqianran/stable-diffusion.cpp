from __future__ import annotations

from typing import Any

from .contract import GenerateRequest


RECIPES: dict[str, dict[str, Any]] = {
    "sd15": {
        "model": "hf://stable-diffusion-v1-5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg_scale": 7.0,
    },
}


def recipe_uris(name: str) -> list[str]:
    if name not in RECIPES:
        known = ", ".join(sorted(RECIPES))
        raise KeyError(f"unknown recipe {name!r}; known: {known}")
    uris: list[str] = []
    for key in (
        "model",
        "diffusion_model",
        "vae",
        "clip_l",
        "clip_g",
        "clip_vision",
        "t5xxl",
        "llm",
        "llm_vision",
        "taesd",
        "control_net",
        "upscale_model",
    ):
        value = RECIPES[name].get(key)
        if value:
            uris.append(str(value))
    return uris


def apply_recipe(name: str, **overrides: Any) -> GenerateRequest:
    if name not in RECIPES:
        known = ", ".join(sorted(RECIPES))
        raise KeyError(f"unknown recipe {name!r}; known: {known}")
    payload = dict(RECIPES[name])
    for key, value in overrides.items():
        if value is not None and value != "":
            payload[key] = value
    return GenerateRequest.from_dict(payload)
