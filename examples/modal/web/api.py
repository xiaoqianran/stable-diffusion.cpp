from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from sdcpp_hooks.cost_view import ledger_report
from sdcpp_hooks.gpu import DEFAULT_GPU, default_gpu_for_recipe
from sdcpp_hooks.meter import client_ledger_path
from sdcpp_hooks.web_catalog import default_recipe, list_gpus, list_models, normalize_gpu
from sdcpp_hooks.web_events import EventBus
from sdcpp_hooks.web_jobs import JobService


router = APIRouter(prefix="/api")

_DATA_DIR = Path(os.environ.get("SDCPP_WEB_DATA", Path.home() / ".cache" / "sdcpp-modal" / "web"))
_events = EventBus()
_service = JobService(_DATA_DIR, events=_events)


def configure(data_dir: Path | None = None, events: EventBus | None = None) -> JobService:
    global _DATA_DIR, _events, _service
    _DATA_DIR = Path(data_dir or _DATA_DIR)
    _events = events or EventBus()
    _service = JobService(_DATA_DIR, events=_events)
    return _service


def service() -> JobService:
    return _service


class CreateJobBody(BaseModel):
    prompt: str | None = None
    prompts: list[str] = Field(default_factory=list)
    text: str | None = None
    count: int = 1
    recipe: str = default_recipe()
    model: str | None = None
    gpu: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int | None = None
    dry_run: bool = False


def _specs_from_body(body: CreateJobBody) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if body.prompt and body.prompt.strip():
        specs.append({"prompt": body.prompt, "count": body.count, "seed": body.seed})
    for line in body.prompts:
        if line.strip():
            specs.append({"prompt": line, "count": body.count, "seed": body.seed})
    if body.text:
        for line in body.text.splitlines():
            text = line.strip()
            if text:
                specs.append({"prompt": text, "count": body.count, "seed": body.seed})
    if not specs:
        raise HTTPException(400, "Provide prompt, prompts, or text")
    return specs


def _config_from_body(body: CreateJobBody) -> dict[str, Any]:
    recipe = body.recipe or body.model or default_recipe()
    return {
        "recipe": recipe,
        "gpu": body.gpu or default_gpu_for_recipe(recipe),
        "width": body.width,
        "height": body.height,
        "steps": body.steps,
        "cfg_scale": body.cfg_scale,
        "seed": body.seed,
        "count": body.count,
        "dry_run": body.dry_run or os.environ.get("SDCPP_WEB_DRY_RUN") == "1",
    }


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/cost")
def cost_ledger(job_id: str | None = None) -> dict[str, Any]:
    return ledger_report(job_id=job_id)


@router.get("/runtime/queue")
def runtime_queue() -> dict[str, Any]:
    """Current local GPU scheduler state used by the workbench UI."""
    return {"gpu": _service.queue_snapshot()}


@router.get("/meta")
def meta() -> dict[str, Any]:
    queue = _service.queue_snapshot()
    return {
        "models": list_models(),
        "gpus": list_gpus(),
        "defaults": {
            "model": default_recipe(),
            "recipe": default_recipe(),
            "gpu": DEFAULT_GPU,
            "port": 7863,
            "data_dir": str(_DATA_DIR),
            "cost_log": str(client_ledger_path()),
        },
        "version": "0.1.0",
        "runtime": {
            "note": "Local FastAPI. CPU/GPU work calls persistent deployed Modal apps.",
            "would_use": "deployed Function.from_name / Cls.from_name",
            "gpu_queue_max_active": queue["max_active"],
        },
    }


@router.get("/doctor")
def doctor() -> dict[str, Any]:
    checks = []
    try:
        list_models()
        checks.append({"name": "recipes", "ok": True, "detail": "7 bundled recipes"})
    except Exception as exc:
        checks.append({"name": "recipes", "ok": False, "detail": str(exc)})
    try:
        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        checks.append({"name": "data_dir", "ok": True, "detail": str(_DATA_DIR)})
    except Exception as exc:
        checks.append({"name": "data_dir", "ok": False, "detail": str(exc)})
    try:
        import modal  # noqa: F401

        checks.append({"name": "modal", "ok": True, "detail": "import ok"})
    except Exception as exc:
        checks.append({"name": "modal", "ok": False, "detail": str(exc)})
    checks.append(_proxy_extra_check())
    try:
        from PIL import Image  # noqa: F401

        checks.append({"name": "pillow", "ok": True, "detail": "import ok"})
    except Exception as exc:
        checks.append({"name": "pillow", "ok": False, "detail": str(exc)})
    skip = {"modal", "api_proxy"}
    return {"ready": all(item["ok"] for item in checks if item["name"] not in skip), "checks": checks}


def _proxy_extra_check() -> dict[str, Any]:
    try:
        import aiohttp_socks  # noqa: F401
        import python_socks  # noqa: F401
    except ImportError:
        return {
            "name": "api_proxy",
            "ok": False,
            "detail": "missing; install modal[api-proxy-support]",
        }
    detail = "modal[api-proxy-support]"
    if os.environ.get("MODAL_DISABLE_API_PROXY"):
        detail += "; MODAL_DISABLE_API_PROXY=1"
    elif any(os.environ.get(key) for key in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy")):
        detail += "; proxy env set"
    return {"name": "api_proxy", "ok": True, "detail": detail}


@router.post("/jobs")
def create_job(body: CreateJobBody) -> dict[str, Any]:
    try:
        config = _config_from_body(body)
        normalize_gpu(config["gpu"])
        job = _service.create_job(_specs_from_body(body), config)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    _service.start(job["id"])
    return job


@router.post("/jobs/from-file")
async def create_job_from_file(
    file: UploadFile = File(...),
    recipe: str = default_recipe(),
    model: str | None = None,
    gpu: str | None = None,
    count: int = 1,
    width: int | None = None,
    height: int | None = None,
    steps: int | None = None,
    cfg_scale: float | None = None,
    seed: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    text = (await file.read()).decode("utf-8", errors="replace")
    body = CreateJobBody(
        text=text,
        recipe=model or recipe,
        gpu=gpu,
        count=count,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=seed,
        dry_run=dry_run,
    )
    return create_job(body)


@router.get("/jobs")
def list_jobs() -> list[dict[str, Any]]:
    return _service.list_jobs()


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return _service.get_job_detail(job_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown job {job_id}") from exc


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict[str, Any]:
    try:
        return _service.resume(job_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown job {job_id}") from exc


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        return _service.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown job {job_id}") from exc


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    try:
        snapshot = _service.get_job(job_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown job {job_id}") from exc
    subscriber = _events.subscribe(job_id)

    async def stream():
        yield _sse({"type": "job.snapshot", "job_id": job_id, "payload": snapshot})
        for event in _events.history(job_id):
            yield _sse(event.as_dict())
        try:
            while True:
                event = await asyncio.to_thread(subscriber.get)
                yield _sse(event.as_dict())
                if event.type in {"job.completed", "job.failed", "job.cancelled"}:
                    break
        finally:
            _events.unsubscribe(job_id, subscriber)

    return StreamingResponse(stream(), media_type="text/event-stream")


@router.get("/gallery")
def gallery(
    job_id: str | None = None,
    recipe: str | None = None,
    model: str | None = None,
    q: str | None = None,
    sort: str = "newest",
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    return _service.list_images(
        job_id=job_id,
        recipe=recipe or model,
        q=q,
        sort=sort,
        page=page,
        per_page=per_page,
    )


@router.get("/images/{image_id}")
def image_meta(image_id: str) -> dict[str, Any]:
    try:
        return _service.get_image(image_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown image {image_id}") from exc


@router.get("/images/{image_id}/file")
def image_file(image_id: str) -> FileResponse:
    try:
        record = _service.get_image(image_id)
    except KeyError as exc:
        raise HTTPException(404, f"unknown image {image_id}") from exc
    path = Path(record["path"] or "")
    if not path.exists():
        raise HTTPException(404, "image file missing on disk")
    return FileResponse(path, media_type="image/png", filename=path.name)


@router.post("/images/{image_id}/regenerate")
def regenerate(image_id: str) -> dict[str, Any]:
    try:
        return _service.regenerate(image_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, default=str)}\n\n"
