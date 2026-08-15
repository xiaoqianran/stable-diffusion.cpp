from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any


class ValidationError(ValueError):
    pass


@dataclass
class GenerateRequest:
    prompt: str
    negative_prompt: str = ""
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    sampling_method: str | None = None
    scheduler: str | None = None
    batch_count: int | None = None
    strength: float | None = None
    model: str | None = None
    diffusion_model: str | None = None
    vae: str | None = None
    clip_l: str | None = None
    clip_g: str | None = None
    t5xxl: str | None = None
    llm: str | None = None
    llm_vision: str | None = None
    lora_dir: str | None = None
    init_image: str | None = None
    extra_cli: dict[str, Any] = field(default_factory=dict)
    extra_http: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.prompt or not str(self.prompt).strip():
            raise ValidationError("prompt is required")
        if not self.model and not self.diffusion_model:
            raise ValidationError("model or diffusion_model is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GenerateRequest:
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


@dataclass
class GenerateResult:
    images: list[bytes]
    argv: list[str]
    dropped_fields: list[str]
    engine_id: str
    seed: int | None = None
