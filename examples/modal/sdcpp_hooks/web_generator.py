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
    def _requests(self, items: list[dict[str, Any]]) -> list[Any]:
        return [
            apply_recipe(
                item["recipe"],
                prompt=item["prompt"],
                seed=item["seed"],
                width=item.get("width"),
                height=item.get("height"),
                steps=item.get("steps"),
                cfg_scale=item.get("cfg_scale"),
            )
            for item in items
        ]

    def prepare_job(
        self,
        items: list[dict[str, Any]],
        *,
        gpu: str,
        job_id: str = "",
    ) -> list[Any]:
        """Stage every artifact on CPU and wait for the shared Volume commit."""
        if not items:
            return []
        import os

        from .deployed import ensure_deployed, storage_function
        from .meter import bind_task
        from .modal_meter import billed_remote, billed_service

        os.environ["SDCPP_GPU"] = gpu
        ensure_deployed()
        requests = self._requests(items)

        with bind_task(job_id=job_id, recipe=items[0]["recipe"], gpu=gpu):
            with billed_service("storage"):
                ensure_artifacts = storage_function("ensure_artifacts")
                seen: set[tuple[str, ...]] = set()
                for request in requests:
                    payload = request.to_dict()
                    artifact_key = tuple(
                        str(payload.get(key) or "")
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
                            "control_net",
                            "taesd",
                            "upscale_model",
                        )
                    )
                    if artifact_key in seen:
                        continue
                    seen.add(artifact_key)
                    billed_remote(
                        ensure_artifacts,
                        payload,
                        name="ensure_artifacts",
                    )
        return requests

    def generate_prepared_job(
        self,
        items: list[dict[str, Any]],
        requests: list[Any],
        *,
        gpu: str,
        job_id: str = "",
    ) -> Iterator[dict[str, Any]]:
        """Run only the GPU phase for requests whose artifacts are already staged."""
        if not items:
            return

        from .deployed import engine
        from .meter import bind_task
        from .modal_meter import billed_remote, billed_service

        if len(items) != len(requests):
            raise ValueError("items/requests length mismatch")

        with bind_task(job_id=job_id, recipe=items[0]["recipe"], gpu=gpu):
            with billed_service("gpu"):
                remote_engine = engine(gpu=gpu)
                for item, request in zip(items, requests):
                    with bind_task(image_id=item.get("id"), recipe=item["recipe"]):
                        result = billed_remote(
                            remote_engine.generate,
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

    def generate_job(
        self,
        items: list[dict[str, Any]],
        *,
        gpu: str,
        job_id: str = "",
    ) -> Iterator[dict[str, Any]]:
        """Compatibility wrapper: CPU stage first, then GPU generation."""
        requests = self.prepare_job(items, gpu=gpu, job_id=job_id)
        yield from self.generate_prepared_job(
            items,
            requests,
            gpu=gpu,
            job_id=job_id,
        )
