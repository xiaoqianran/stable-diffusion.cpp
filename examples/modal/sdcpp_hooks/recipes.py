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
    "sd2": {
        "model": "hf://Manojb/stable-diffusion-2-1-base/v2-1_512-ema-pruned.safetensors",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg_scale": 7.0,
    },
    "sd-turbo": {
        "model": "hf://stabilityai/sd-turbo/sd_turbo.safetensors",
        "width": 512,
        "height": 512,
        "steps": 4,
        "cfg_scale": 1.0,
        "sampling_method": "euler",
    },
    "sdxl-turbo": {
        "model": "hf://stabilityai/sdxl-turbo/sd_xl_turbo_1.0_fp16.safetensors",
        "width": 512,
        "height": 512,
        "steps": 4,
        "cfg_scale": 1.0,
        "sampling_method": "euler",
    },
    "ssd-1b": {
        "model": "hf://segmind/SSD-1B/SSD-1B.safetensors",
        "width": 512,
        "height": 512,
        "steps": 8,
        "cfg_scale": 7.0,
    },
    "dreamlike-photoreal": {
        "model": "hf://dreamlike-art/dreamlike-photoreal-2.0/dreamlike-photoreal-2.0.safetensors",
        "width": 512,
        "height": 512,
        "steps": 8,
        "cfg_scale": 7.0,
    },
    "ideogram4": {
        "diffusion_model": "hf://ideogram-ai/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors",
        "uncond_diffusion_model": "hf://ideogram-ai/ideogram-4-fp8/unconditional_transformer/diffusion_pytorch_model.safetensors",
        "vae": "hf://black-forest-labs/FLUX.2-dev/ae.safetensors",
        "llm": "hf://unsloth/Qwen3-VL-8B-Instruct-GGUF/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "cfg_scale": 4.0,
        "extra_cli": {
            "--diffusion-fa": True,
            "--offload-to-cpu": True,
        },
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
        "uncond_diffusion_model",
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
    recipe_extra = dict(payload.pop("extra_cli", None) or {})
    for key, value in overrides.items():
        if key == "extra_cli":
            continue
        if value is not None and value != "":
            payload[key] = value
    merged_extra = dict(recipe_extra)
    override_extra = overrides.get("extra_cli")
    if override_extra:
        merged_extra.update(override_extra)
    if merged_extra:
        payload["extra_cli"] = merged_extra
    return GenerateRequest.from_dict(payload)
