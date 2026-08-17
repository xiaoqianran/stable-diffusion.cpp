from __future__ import annotations

import base64
import hashlib
import io
import time
from collections.abc import Iterator
from typing import Any, Callable, Mapping

from .recipes import apply_recipe


ALLOWED_PARALLELISM = {1, 2, 4}


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
        from .deployed import ensure_deployed, storage_function
        from .meter import bind_task
        from .modal_meter import billed_remote, billed_service

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

    @staticmethod
    def _result(item: dict[str, Any], request: Any, result: dict[str, Any]) -> dict[str, Any]:
        images = []
        for raw in result.get("images") or []:
            if isinstance(raw, (bytes, bytearray)):
                images.append(bytes(raw))
            else:
                images.append(base64.b64decode(raw))
        return {
            **item,
            "images": images,
            "width": result.get("width") or request.width,
            "height": result.get("height") or request.height,
            "steps": result.get("steps") or request.steps,
            "cfg_scale": result.get("cfg_scale") or request.cfg_scale,
            "seed": result.get("seed") if result.get("seed") is not None else item["seed"],
            "duration_ms": result.get("duration_ms"),
            "host": result.get("host") or {},
            "argv": result.get("argv"),
            "dropped_fields": result.get("dropped_fields"),
            "cost": result.get("cost"),
            "model_resident": bool(result.get("model_resident")),
            "server_endpoint": result.get("server_endpoint"),
        }

    def generate_prepared_job(
        self,
        items: list[dict[str, Any]],
        requests: list[Any],
        *,
        gpu: str,
        job_id: str = "",
        parallelism: int = 1,
        existing_calls: Mapping[str, str] | None = None,
        on_spawn: Callable[[str, str], None] | None = None,
        on_done: Callable[[str, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run a prepared batch with a bounded set of durable Modal FunctionCalls.

        Unlike a ThreadPoolExecutor full of blocking `.remote()` calls, this uses
        `.spawn()` so call IDs can be persisted, cancelled, and re-attached after
        the local Web process restarts. At most `parallelism` calls are active.
        """
        if not items:
            return
        if len(items) != len(requests):
            raise ValueError("items/requests length mismatch")
        parallelism = int(parallelism)
        if parallelism not in ALLOWED_PARALLELISM:
            raise ValueError("parallelism must be one of 1, 2, 4")

        import modal

        from .deployed import WEB_GPU_POOL_MAX, engine

        recipe = str(items[0]["recipe"])
        if any(str(item["recipe"]) != recipe for item in items):
            raise ValueError("one GPU batch job must use a single recipe")
        remote_engine = engine(gpu=gpu, recipe=recipe, max_containers=WEB_GPU_POOL_MAX)
        persisted = dict(existing_calls or {})
        active: dict[str, tuple[Any, int]] = {}
        next_index = 0

        def notify_spawn(image_id: str, call_id: str) -> None:
            if on_spawn:
                on_spawn(image_id, call_id)

        def notify_done(image_id: str, status: str) -> None:
            if on_done:
                on_done(image_id, status)

        def add_call(index: int) -> None:
            item = items[index]
            image_id = str(item["id"])
            existing_id = persisted.pop(image_id, "")
            if existing_id:
                call = modal.FunctionCall.from_id(existing_id)
                call_id = existing_id
            else:
                call = remote_engine.generate.spawn(requests[index].to_dict())
                call_id = str(call.object_id)
                notify_spawn(image_id, call_id)
            active[call_id] = (call, index)

        while next_index < len(items) and len(active) < min(parallelism, len(items)):
            add_call(next_index); next_index += 1

        try:
            while active:
                if cancelled and cancelled():
                    for call_id, (call, index) in list(active.items()):
                        try:
                            call.cancel()
                        finally:
                            notify_done(str(items[index]["id"]), "cancelled")
                            active.pop(call_id, None)
                    return

                progressed = False
                for call_id, (call, index) in list(active.items()):
                    try:
                        result = call.get(timeout=0)
                    except TimeoutError:
                        continue
                    except Exception:
                        notify_done(str(items[index]["id"]), "failed")
                        active.pop(call_id, None)
                        raise
                    item = items[index]
                    request = requests[index]
                    notify_done(str(item["id"]), "completed")
                    active.pop(call_id, None)
                    progressed = True
                    yield self._result(item, request, result)
                    if next_index < len(items):
                        add_call(next_index); next_index += 1
                if not progressed and active:
                    time.sleep(0.1)
        finally:
            if cancelled and cancelled():
                for call_id, (call, index) in list(active.items()):
                    try:
                        call.cancel()
                    except Exception:
                        pass
                    notify_done(str(items[index]["id"]), "cancelled")
                    active.pop(call_id, None)

    def generate_job(
        self,
        items: list[dict[str, Any]],
        *,
        gpu: str,
        job_id: str = "",
        parallelism: int = 1,
    ) -> Iterator[dict[str, Any]]:
        requests = self.prepare_job(items, gpu=gpu, job_id=job_id)
        yield from self.generate_prepared_job(
            items,
            requests,
            gpu=gpu,
            job_id=job_id,
            parallelism=parallelism,
        )
