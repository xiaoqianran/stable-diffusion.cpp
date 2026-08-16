from __future__ import annotations

import hashlib
import io
import time
from collections.abc import Iterator
from typing import Any

from .recipes import apply_recipe


def render_placeholder(prompt: str, recipe: str, seed: int, width: int, height: int) -> bytes:
    from PIL import Image, ImageDraw, ImageFont

    digest = hashlib.sha256(f"{recipe}:{prompt}".encode("utf-8")).digest()
    color = (40 + digest[0] % 80, 32 + digest[1] % 70, 28 + digest[2] % 60)
    image = Image.new("RGB", (width, height), color)
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    text = f"DRY RUN\n{recipe}\nseed {seed}\n\n{prompt}"
    margin = max(24, width // 32)
    draw.multiline_text((margin, margin), text, fill=(244, 235, 225), font=font, spacing=8)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


class MockGenerator:
    def generate_one(
        self,
        *,
        prompt: str,
        recipe: str,
        seed: int,
        width: int,
        height: int,
        steps: int,
        cfg_scale: float,
        gpu: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        payload = apply_recipe(
            recipe,
            prompt=prompt,
            seed=seed,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
        )
        image = render_placeholder(
            prompt,
            recipe,
            seed,
            payload.width or width,
            payload.height or height,
        )
        latency_ms = int((time.perf_counter() - started) * 1000)
        return {
            "images": [image],
            "width": payload.width or width,
            "height": payload.height or height,
            "steps": payload.steps or steps,
            "cfg_scale": payload.cfg_scale or cfg_scale,
            "seed": seed,
            "duration_ms": latency_ms,
            "host": {"modal_gpu": gpu, "gpu_name": "dry-run"},
            "dry_run": True,
        }


class ModalGenerator:
    def generate_job(
        self,
        items: list[dict[str, Any]],
        *,
        gpu: str,
    ) -> Iterator[dict[str, Any]]:
        if not items:
            return
        from app import SDEngine, ensure_artifacts, gpu_app, storage_app
        from .modal_meter import billed_app, billed_remote

        first = apply_recipe(
            items[0]["recipe"],
            prompt=items[0]["prompt"],
            seed=items[0]["seed"],
            width=items[0].get("width"),
            height=items[0].get("height"),
            steps=items[0].get("steps"),
            cfg_scale=items[0].get("cfg_scale"),
        ).to_dict()
        with billed_app(storage_app, "storage"):
            billed_remote(ensure_artifacts, first, name="ensure_artifacts")
        with billed_app(gpu_app, "gpu"):
            try:
                engine = SDEngine.with_options(gpu=gpu)()
            except Exception:
                engine = SDEngine()
            for item in items:
                request = apply_recipe(
                    item["recipe"],
                    prompt=item["prompt"],
                    seed=item["seed"],
                    width=item.get("width"),
                    height=item.get("height"),
                    steps=item.get("steps"),
                    cfg_scale=item.get("cfg_scale"),
                )
                result = billed_remote(
                    engine.generate,
                    request.to_dict(),
                    name="generate",
                    gpu=True,
                )
                images = []
                for raw in result.get("images") or []:
                    if isinstance(raw, (bytes, bytearray)):
                        images.append(bytes(raw))
                    else:
                        import base64

                        images.append(base64.b64decode(raw))
                yield {
                    **item,
                    "images": images,
                    "width": result.get("width") or request.width,
                    "height": result.get("height") or request.height,
                    "steps": result.get("steps") or request.steps,
                    "cfg_scale": result.get("cfg_scale") or request.cfg_scale,
                    "seed": result.get("seed") or item["seed"],
                    "duration_ms": result.get("duration_ms"),
                    "host": result.get("host") or {},
                    "argv": result.get("argv"),
                    "dropped_fields": result.get("dropped_fields"),
                    "cost": result.get("cost"),
                }
