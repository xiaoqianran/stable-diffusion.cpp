from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .contract import GenerateRequest
from .discover import EngineCapabilities, Flag


# Stable field -> candidate CLI names. First match in the probed --help wins.
# Keep historical and likely future names here so upstream renames do not
# require a hooks change.
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "prompt": ("--prompt", "-p"),
    "negative_prompt": ("--negative-prompt", "-n"),
    "width": ("--width", "-W"),
    "height": ("--height", "-H"),
    "steps": ("--steps", "--sample-steps", "--sample_steps"),
    "cfg_scale": ("--cfg-scale", "--cfg_scale", "--txt-cfg", "--txt_cfg"),
    "seed": ("--seed", "-s"),
    "sampling_method": ("--sampling-method", "--sampler", "--sample-method"),
    "scheduler": ("--scheduler",),
    "batch_count": ("--batch-count", "-b"),
    "strength": ("--strength",),
    "model": ("--model", "-m"),
    "diffusion_model": ("--diffusion-model",),
    "vae": ("--vae",),
    "clip_l": ("--clip_l", "--clip-l"),
    "clip_g": ("--clip_g", "--clip-g"),
    "t5xxl": ("--t5xxl",),
    "llm": ("--llm",),
    "llm_vision": ("--llm_vision", "--llm-vision"),
    "lora_dir": ("--lora-model-dir",),
    "init_image": ("--init-img", "--init-image", "-i"),
    "output": ("--output", "-o"),
}


@dataclass
class PlannedCommand:
    argv: list[str]
    dropped_fields: list[str]


def _first_flag(engine: EngineCapabilities, names: Iterable[str]) -> Flag | None:
    for name in names:
        if engine.has_flag(name):
            return engine.flag(name)
    return None


def _emit(flag: Flag, value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if value is True:
        return [flag.name]
    return [flag.name, str(value)]


def _normalize_cli_name(name: str) -> str:
    if name.startswith("-"):
        return name
    return f"--{name}"


def adapt_request(
    request: GenerateRequest,
    engine: EngineCapabilities,
    output_path: str,
) -> PlannedCommand:
    request.validate()
    argv = [engine.binary]
    dropped: list[str] = []

    field_values = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or None,
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg_scale": request.cfg_scale,
        "seed": request.seed,
        "sampling_method": request.sampling_method,
        "scheduler": request.scheduler,
        "batch_count": request.batch_count,
        "strength": request.strength,
        "model": request.model,
        "diffusion_model": request.diffusion_model,
        "vae": request.vae,
        "clip_l": request.clip_l,
        "clip_g": request.clip_g,
        "t5xxl": request.t5xxl,
        "llm": request.llm,
        "llm_vision": request.llm_vision,
        "lora_dir": request.lora_dir,
        "init_image": request.init_image,
        "output": output_path,
    }

    for field_name, value in field_values.items():
        if value is None:
            continue
        flag = _first_flag(engine, FIELD_ALIASES[field_name])
        if flag is None:
            dropped.append(field_name)
            continue
        argv.extend(_emit(flag, value))

    for raw_name, value in request.extra_cli.items():
        name = _normalize_cli_name(raw_name)
        if not engine.has_flag(name):
            dropped.append(f"extra_cli.{name}")
            continue
        argv.extend(_emit(engine.flag(name), value))

    return PlannedCommand(argv=argv, dropped_fields=dropped)
