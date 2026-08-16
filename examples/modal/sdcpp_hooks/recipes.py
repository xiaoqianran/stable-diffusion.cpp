from __future__ import annotations

from typing import Any

from .contract import GenerateRequest


RECIPES: dict[str, dict[str, Any]] = {
    "ideogram4": {
        "diffusion_model": "hf://leejet/ideogram-4-GGUF/ideogram4-Q4_0.gguf",
        "uncond_diffusion_model": "hf://leejet/ideogram-4-GGUF/ideogram4_uncond-Q4_0.gguf",
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
    "flux2-klein": {
        "diffusion_model": "hf://leejet/FLUX.2-klein-9B-GGUF/flux-2-klein-9b-Q4_0.gguf",
        "vae": "hf://black-forest-labs/FLUX.2-dev/ae.safetensors",
        "llm": "hf://unsloth/Qwen3-8B-GGUF/Qwen3-8B-Q4_K_M.gguf",
        "width": 1024,
        "height": 1024,
        "steps": 4,
        "cfg_scale": 1.0,
        "extra_cli": {
            "--diffusion-fa": True,
            "--offload-to-cpu": True,
        },
    },
    "flux2-dev": {
        "diffusion_model": "hf://city96/FLUX.2-dev-gguf/flux2-dev-Q4_K_S.gguf",
        "vae": "hf://black-forest-labs/FLUX.2-dev/ae.safetensors",
        "llm": (
            "hf://unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/"
            "Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf"
        ),
        "width": 1024,
        "height": 1024,
        "steps": 20,
        "cfg_scale": 1.0,
        "sampling_method": "euler",
        "extra_cli": {
            "--diffusion-fa": True,
            "--offload-to-cpu": True,
        },
    },
    "z-image-turbo": {
        "diffusion_model": "hf://leejet/Z-Image-Turbo-GGUF/z_image_turbo-Q3_K.gguf",
        "vae": "hf://black-forest-labs/FLUX.1-schnell/ae.safetensors",
        "llm": (
            "hf://unsloth/Qwen3-4B-Instruct-2507-GGUF/"
            "Qwen3-4B-Instruct-2507-Q4_K_M.gguf"
        ),
        "width": 512,
        "height": 1024,
        "steps": 8,
        "cfg_scale": 1.0,
        "extra_cli": {
            "--diffusion-fa": True,
            "--offload-to-cpu": True,
        },
    },
    "sdxl-turbo": {
        "model": "hf://stabilityai/sdxl-turbo/sd_xl_turbo_1.0_fp16.safetensors",
        "width": 512,
        "height": 512,
        "steps": 4,
        "cfg_scale": 1.0,
        "sampling_method": "euler",
    },
    "sd2": {
        "model": "hf://Manojb/stable-diffusion-2-1-base/v2-1_512-ema-pruned.safetensors",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg_scale": 7.0,
    },
    "sd15": {
        "model": "hf://stable-diffusion-v1-5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors",
        "width": 512,
        "height": 512,
        "steps": 20,
        "cfg_scale": 7.0,
    },
}

_URI_KEYS = (
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
)


def _uris_for(name: str) -> list[str]:
    uris: list[str] = []
    for key in _URI_KEYS:
        value = RECIPES[name].get(key)
        if value:
            uris.append(str(value))
    return uris


def recipe_uris(name: str) -> list[str]:
    if name not in RECIPES:
        known = ", ".join(sorted(RECIPES))
        raise KeyError(f"unknown recipe {name!r}; known: {known}")
    return _uris_for(name)


def all_recipe_uris() -> list[str]:
    seen: set[str] = set()
    uris: list[str] = []
    for name in RECIPES:
        for uri in _uris_for(name):
            if uri not in seen:
                seen.add(uri)
                uris.append(uri)
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
