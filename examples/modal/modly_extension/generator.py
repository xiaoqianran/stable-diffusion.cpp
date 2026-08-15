"""Modly hook for stable-diffusion.cpp.

This file is the only Modly-facing adapter. It does not import sd.cpp
internals. Generation runs on the deployed Modal app `sdcpp-hooks`, which
probes `sd-cli -h` at container start.
"""

from __future__ import annotations

import base64
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

try:
    from services.generators.base import BaseGenerator
except ImportError:  # unit tests / standalone

    class BaseGenerator:  # type: ignore[no-redef]
        MODEL_ID = ""
        DISPLAY_NAME = ""
        VRAM_GB = 0

        def __init__(self, model_dir: Path, outputs_dir: Path) -> None:
            self.model_dir = model_dir
            self.outputs_dir = outputs_dir
            self._model = None

        def unload(self) -> None:
            self._model = None

        def _report(self, progress_cb, pct: int, step: str) -> None:
            if progress_cb:
                progress_cb(pct, step)

        def _check_cancelled(self, cancel_event) -> None:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("cancelled")


def _prompt_from(params: dict[str, Any]) -> str:
    for key in ("prompt", "text", "positive"):
        value = params.get(key)
        if value:
            return str(value)
    return ""


_SD15_MODEL = (
    "hf://stable-diffusion-v1-5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors"
)


def build_payload(params: dict[str, Any]) -> dict[str, Any]:
    prompt = _prompt_from(params)
    if not prompt.strip():
        raise ValueError("prompt is required")
    return {
        "prompt": prompt,
        "negative_prompt": params.get("negative_prompt") or "",
        "model": params.get("model") or _SD15_MODEL,
        "width": int(params.get("width") or 512),
        "height": int(params.get("height") or 512),
        "steps": int(params.get("steps") or 20),
        "cfg_scale": float(params.get("cfg_scale") or 7.0),
        "seed": int(params["seed"]) if params.get("seed") is not None else 42,
    }


class SDCppGenerator(BaseGenerator):
    MODEL_ID = "sdcpp"
    DISPLAY_NAME = "stable-diffusion.cpp"
    VRAM_GB = 8

    def is_downloaded(self) -> bool:
        return True

    def load(self) -> None:
        if self._model is not None:
            return
        import modal

        self._model = modal.Cls.from_name("sdcpp-hooks", "SDEngine")

    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event=None,
    ) -> Path:
        del image_bytes
        self._check_cancelled(cancel_event)
        payload = build_payload(params)
        self._report(progress_cb, 8, "Submitting to Modal sd-cli…")
        if self._model is None:
            self.load()
        result = self._model().generate.remote(payload)
        self._check_cancelled(cancel_event)
        images = result.get("images") or []
        if not images:
            dropped = result.get("dropped_fields") or []
            raise RuntimeError(f"sdcpp returned no images; dropped={dropped}")
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        dest = self.outputs_dir / f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        dest.write_bytes(base64.b64decode(images[0]))
        self._report(progress_cb, 100, "Done")
        return dest
