from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from sdcpp_hooks.cost_view import ledger_report
from sdcpp_hooks.gpu import DEFAULT_GPU, default_gpu_for_recipe
from sdcpp_hooks.meter import client_ledger_path
from sdcpp_hooks.runtime_identity import deployment_identity
from sdcpp_hooks.web_catalog import default_recipe, list_gpus, list_models, normalize_gpu
from sdcpp_hooks.web_events import EventBus
from sdcpp_hooks.web_jobs import JobService


router = APIRouter(prefix="/api")
_DATA_DIR = Path(os.environ.get("SDCPP_WEB_DATA", Path.home() / ".cache" / "sdcpp-modal" / "web"))
_events = EventBus()
_service = JobService(_DATA_DIR, events=_events)


def _int_env(name: str, default: int) -> int:
    try:
        return max(1, int(os.environ.get(name, str(default))))
    except ValueError:
        return default


MAX_PROMPT_CHARS = _int_env("SDCPP_MAX_PROMPT_CHARS", 20_000)
MAX_PROMPTS = _int_env("SDCPP_MAX_PROMPTS", 5_000)
MAX_TOTAL_IMAGES = _int_env("SDCPP_MAX_TOTAL_IMAGES", 10_000)
MAX_COUNT = _int_env("SDCPP_MAX_COUNT", 100)
MAX_DIMENSION = _int_env("SDCPP_MAX_DIMENSION", 4096)
MAX_STEPS = _int_env("SDCPP_MAX_STEPS", 200)
MAX_UPLOAD_BYTES = _int_env("SDCPP_MAX_UPLOAD_BYTES", 10 * 1024 * 1024)


def configure(data_dir: Path | None = None, events: EventBus | None = None) -> JobService:
    global _DATA_DIR, _events, _service
    try:
        _service.close()
    except Exception:
        pass
    _DATA_DIR = Path(data_dir or _DATA_DIR)
    _events = events or EventBus()
    _service = JobService(_DATA_DIR, events=_events)
    return _service


def service() -> JobService:
    return _service


class CreateJobBody(BaseModel):
    prompt: str | None = Field(default=None, max_length=MAX_PROMPT_CHARS)
    prompts: list[str] = Field(default_factory=list, max_length=MAX_PROMPTS)
    text: str | None = Field(default=None, max_length=MAX_UPLOAD_BYTES)
    count: int = Field(default=1, ge=1, le=MAX_COUNT)
    recipe: str | None = None
    model: str | None = None
    gpu: str | None = None
    width: int | None = Field(default=None, ge=64, le=MAX_DIMENSION)
    height: int | None = Field(default=None, ge=64, le=MAX_DIMENSION)
    steps: int | None = Field(default=None, ge=1, le=MAX_STEPS)
    cfg_scale: float | None = Field(default=None, ge=0, le=100)
    seed: int | None = Field(default=None, ge=-1, le=2**63 - 1)
    parallelism: Literal[1, 2, 4] = 1
    dry_run: bool = False


def _validate_prompt(text: str) -> str:
    text = text.strip()
    if len(text) > MAX_PROMPT_CHARS:
        raise HTTPException(413, f"Prompt exceeds {MAX_PROMPT_CHARS} characters")
    return text


def _validate_specs(specs: list[dict[str, Any]], default_count: int) -> list[dict[str, Any]]:
    if not specs:
        raise HTTPException(400, "Provide prompt, prompts, text, or a JSONL/TXT file")
    if len(specs) > MAX_PROMPTS:
        raise HTTPException(413, f"Too many prompts; max {MAX_PROMPTS}")
    total = 0
    clean: list[dict[str, Any]] = []
    for spec in specs:
        prompt = _validate_prompt(str(spec.get("prompt") or ""))
        if not prompt:
            continue
        count = int(spec.get("count") or default_count)
        if not 1 <= count <= MAX_COUNT:
            raise HTTPException(400, f"count must be between 1 and {MAX_COUNT}")
        total += count
        if total > MAX_TOTAL_IMAGES:
            raise HTTPException(413, f"Job exceeds {MAX_TOTAL_IMAGES} total images")
        clean.append({"prompt": prompt, "count": count, "seed": spec.get("seed")})
    if not clean:
        raise HTTPException(400, "Every prompt was empty")
    return clean


def _specs_from_body(body: CreateJobBody) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if body.prompt and body.prompt.strip():
        specs.append({"prompt": body.prompt, "count": body.count, "seed": body.seed})
    specs.extend({"prompt": line, "count": body.count, "seed": body.seed} for line in body.prompts if line.strip())
    if body.text:
        specs.extend({"prompt": line, "count": body.count, "seed": body.seed} for line in body.text.splitlines() if line.strip())
    return _validate_specs(specs, body.count)


def _specs_from_jsonl(text: str, default_count: int, default_seed: int | None) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid JSONL at line {lineno}: {exc.msg}") from exc
        if not isinstance(data, dict) or not isinstance(data.get("prompt"), str):
            raise HTTPException(400, f"JSONL line {lineno} must be an object with string field 'prompt'")
        specs.append({
            "prompt": data["prompt"],
            "count": data.get("count", default_count),
            "seed": data.get("seed", default_seed),
        })
    return _validate_specs(specs, default_count)


def _adapt_prompt_for_recipe(prompt: str, recipe: str) -> str:
    if recipe != "ideogram4":
        return prompt
    stripped = prompt.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Ideogram JSON prompt is invalid: {exc.msg}") from exc
        if not isinstance(payload, dict):
            raise HTTPException(400, "Ideogram JSON prompt must be an object")
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return json.dumps({"high_level_description": stripped}, ensure_ascii=False, separators=(",", ":"))


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
        "parallelism": body.parallelism,
        "dry_run": body.dry_run or os.environ.get("SDCPP_WEB_DRY_RUN") == "1",
    }


def _create_from_specs(specs: list[dict[str, Any]], body: CreateJobBody) -> dict[str, Any]:
    try:
        config = _config_from_body(body)
        normalize_gpu(config["gpu"])
        specs = [{**spec, "prompt": _adapt_prompt_for_recipe(str(spec["prompt"]), config["recipe"])} for spec in specs]
        job = _service.create_job(specs, config)
    except (KeyError, ValueError) as exc:
        raise HTTPException(400, str(exc)) from exc
    _service.start(job["id"])
    return job


@router.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "identity": deployment_identity(role="web")}


@router.get("/cost")
def cost_ledger(job_id: str | None = None) -> dict[str, Any]:
    report = ledger_report(job_id=job_id)
    report.setdefault("kind", "estimate")
    return report


@router.get("/runtime/queue")
def runtime_queue() -> dict[str, Any]:
    return {"gpu": _service.queue_snapshot()}


@router.get("/meta")
def meta() -> dict[str, Any]:
    queue = _service.queue_snapshot()
    return {
        "models": list_models(),
        "gpus": list_gpus(),
        "defaults": {
            "model": default_recipe(), "recipe": default_recipe(), "gpu": DEFAULT_GPU,
            "port": 7863, "data_dir": str(_DATA_DIR), "cost_log": str(client_ledger_path()), "parallelism": 1,
        },
        "limits": {
            "prompt_chars": MAX_PROMPT_CHARS, "prompts": MAX_PROMPTS, "total_images": MAX_TOTAL_IMAGES,
            "count": MAX_COUNT, "dimension": MAX_DIMENSION, "steps": MAX_STEPS, "upload_bytes": MAX_UPLOAD_BYTES,
        },
        "version": "0.3.0",
        "identity": deployment_identity(role="web"),
        "runtime": {
            "note": "Durable local jobs; CPU stages artifacts; Modal FunctionCall IDs are persisted for cancellation/recovery.",
            "gpu_job_max_active": queue["max_active"], "batch_parallelism": [1, 2, 4],
            "model_pool_max_containers": 4, "same_model_affinity": True, "cost_kind": "estimate",
        },
    }


@router.get("/doctor")
def doctor() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    try:
        list_models(); checks.append({"name":"recipes","ok":True,"detail":"7 bundled recipes"})
    except Exception as exc:
        checks.append({"name":"recipes","ok":False,"detail":str(exc)})
    try:
        _DATA_DIR.mkdir(parents=True,exist_ok=True); checks.append({"name":"data_dir","ok":True,"detail":str(_DATA_DIR)})
    except Exception as exc:
        checks.append({"name":"data_dir","ok":False,"detail":str(exc)})
    try:
        import modal  # noqa: F401
        checks.append({"name":"modal","ok":True,"detail":"import ok"})
    except Exception as exc:
        checks.append({"name":"modal","ok":False,"detail":str(exc)})
    checks.append(_proxy_extra_check())
    try:
        from PIL import Image  # noqa: F401
        checks.append({"name":"pillow","ok":True,"detail":"import ok"})
    except Exception as exc:
        checks.append({"name":"pillow","ok":False,"detail":str(exc)})
    try:
        from sdcpp_hooks.deployed import deployment_status
        status = deployment_status(refresh=True)
        checks.append({"name":"deployment_identity","ok":bool(status.get("matches")),"detail":status})
    except Exception as exc:
        checks.append({"name":"deployment_identity","ok":False,"detail":str(exc)})
    skip={"modal","api_proxy","deployment_identity"}
    return {"ready":all(item["ok"] for item in checks if item["name"] not in skip),"checks":checks}


def _proxy_extra_check() -> dict[str, Any]:
    try:
        import aiohttp_socks  # noqa: F401
        import python_socks  # noqa: F401
    except ImportError:
        return {"name":"api_proxy","ok":False,"detail":"missing; install modal[api-proxy-support]"}
    detail="modal[api-proxy-support]"
    if os.environ.get("MODAL_DISABLE_API_PROXY"): detail += "; MODAL_DISABLE_API_PROXY=1"
    elif any(os.environ.get(key) for key in ("HTTPS_PROXY","https_proxy","ALL_PROXY","all_proxy")): detail += "; proxy env set"
    return {"name":"api_proxy","ok":True,"detail":detail}


@router.post("/jobs")
def create_job(body: CreateJobBody) -> dict[str, Any]:
    return _create_from_specs(_specs_from_body(body), body)


@router.post("/jobs/from-file")
async def create_job_from_file(
    file: UploadFile = File(...), recipe: str | None = None, model: str | None = None, gpu: str | None = None,
    count: int = Query(1, ge=1, le=MAX_COUNT), width: int | None = Query(None, ge=64, le=MAX_DIMENSION),
    height: int | None = Query(None, ge=64, le=MAX_DIMENSION), steps: int | None = Query(None, ge=1, le=MAX_STEPS),
    cfg_scale: float | None = Query(None, ge=0, le=100), seed: int | None = None,
    parallelism: Literal[1,2,4] = 1, dry_run: bool = False,
) -> dict[str, Any]:
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds {MAX_UPLOAD_BYTES} bytes")
    text = raw.decode("utf-8", errors="strict")
    body = CreateJobBody(recipe=recipe, model=model, gpu=gpu, count=count, width=width, height=height, steps=steps, cfg_scale=cfg_scale, seed=seed, parallelism=parallelism, dry_run=dry_run)
    suffix = Path(file.filename or "").suffix.lower()
    specs = _specs_from_jsonl(text, count, seed) if suffix == ".jsonl" else _validate_specs([{"prompt":line,"count":count,"seed":seed} for line in text.splitlines() if line.strip()], count)
    return _create_from_specs(specs, body)


@router.get("/jobs")
def list_jobs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), status: str | None = None) -> list[dict[str, Any]]:
    return _service.list_jobs(limit=limit, offset=offset, status=status)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try: return _service.get_job_detail(job_id)
    except KeyError as exc: raise HTTPException(404,f"unknown job {job_id}") from exc


@router.post("/jobs/{job_id}/resume")
def resume_job(job_id: str) -> dict[str, Any]:
    try: return _service.resume(job_id)
    except KeyError as exc: raise HTTPException(404,f"unknown job {job_id}") from exc


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    try: return _service.cancel(job_id)
    except KeyError as exc: raise HTTPException(404,f"unknown job {job_id}") from exc


@router.post("/maintenance/cleanup")
def cleanup(keep_days: int = Query(30, ge=1, le=3650)) -> dict[str, int]:
    return _service.cleanup(keep_days=keep_days)


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str) -> StreamingResponse:
    try: snapshot=_service.get_job(job_id)
    except KeyError as exc: raise HTTPException(404,f"unknown job {job_id}") from exc
    subscriber=_events.subscribe(job_id)
    async def stream():
        yield _sse({"type":"job.snapshot","job_id":job_id,"payload":snapshot})
        for event in _events.history(job_id): yield _sse(event.as_dict())
        try:
            while True:
                try: event=await asyncio.wait_for(asyncio.to_thread(subscriber.get),timeout=20)
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"; continue
                yield _sse(event.as_dict())
                if event.type in {"job.completed","job.failed","job.cancelled"}: break
        finally:
            _events.unsubscribe(job_id,subscriber)
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache","X-Accel-Buffering":"no"})


@router.get("/gallery")
def gallery(job_id: str | None=None, recipe: str | None=None, model: str | None=None, q: str | None=None, sort: str="newest", page: int=Query(1,ge=1), per_page: int=Query(50,ge=1,le=200)) -> dict[str, Any]:
    return _service.list_images(job_id=job_id,recipe=recipe or model,q=q,sort=sort,page=page,per_page=per_page)


@router.get("/images/{image_id}")
def image_meta(image_id: str) -> dict[str, Any]:
    try: return _service.get_image(image_id)
    except KeyError as exc: raise HTTPException(404,f"unknown image {image_id}") from exc


@router.get("/images/{image_id}/file")
def image_file(image_id: str) -> FileResponse:
    try: record=_service.get_image(image_id)
    except KeyError as exc: raise HTTPException(404,f"unknown image {image_id}") from exc
    path=Path(record["path"] or "")
    if not path.exists(): raise HTTPException(404,"image file missing on disk")
    return FileResponse(path,media_type="image/png",filename=path.name)


@router.post("/images/{image_id}/regenerate")
def regenerate(image_id: str) -> dict[str, Any]:
    try: return _service.regenerate(image_id)
    except KeyError as exc: raise HTTPException(404,str(exc)) from exc


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload,default=str)}\n\n"
