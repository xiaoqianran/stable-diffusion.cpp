from __future__ import annotations

import base64
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .adapt import FIELD_ALIASES
from .contract import GenerateRequest
from .hooks import use_engine, use_models
from .probe import probe_server_help
from .recipes import apply_recipe


SERVER_HOST = "127.0.0.1"
SERVER_PORT = 18080
SERVER_URL = f"http://{SERVER_HOST}:{SERVER_PORT}"


class ServerUnavailableError(RuntimeError):
    """The local sd-server process is unavailable and the container may recover it."""


SERVER_MODEL_FIELDS = (
    "model", "diffusion_model", "uncond_diffusion_model", "vae",
    "clip_l", "clip_g", "t5xxl", "llm", "llm_vision", "clip_vision",
    "lora_dir", "taesd", "upscale_model",
)
REQUEST_ONLY_FLAGS = {
    "--prompt", "--negative-prompt", "--width", "--height", "--steps",
    "--sample-steps", "--cfg-scale", "--seed", "--sampling-method",
    "--sampler", "--scheduler", "--batch-count", "--output", "--init-img",
    "--init-image", "--control-net", "--strength",
}


def _normalize_cli_name(name: str) -> str:
    return name if name.startswith("-") else f"--{name}"


def _emit(name: str, value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if value is True:
        return [name]
    return [name, str(value)]


def _resolve_request(request: GenerateRequest, cache_dir: Path) -> GenerateRequest:
    models = use_models(request, cache_dir=cache_dir, allow_download=False)
    data = request.to_dict()
    extra = dict(data.get("extra_cli") or {})
    for key, path in models.items():
        if key in extra or str(key).startswith("-"):
            extra[key] = str(path)
        else:
            data[key] = str(path)
    data["extra_cli"] = extra
    return GenerateRequest.from_dict(data)


def recipe_server_argv(recipe: str, cache_dir: Path, *, host: str = SERVER_HOST, port: int = SERVER_PORT) -> tuple[list[str], list[str]]:
    request = apply_recipe(recipe, prompt="sdcpp modal warm container", seed=0)
    request.validate()
    resolved = _resolve_request(request, Path(cache_dir))
    help_text, binary = probe_server_help()
    engine = use_engine(help_text=help_text, binary=binary)
    argv = [binary]
    dropped: list[str] = []
    for field in SERVER_MODEL_FIELDS:
        value = getattr(resolved, field)
        if value is None:
            continue
        flag = next((name for name in FIELD_ALIASES[field] if engine.has_flag(name)), None)
        if flag is None:
            raise RuntimeError(
                f"sd-server in the configured image does not support required model flag for {field!r}; "
                "the Docker image and Python deployment are incompatible"
            )
        argv.extend(_emit(flag, value))
    for raw_name, value in resolved.extra_cli.items():
        name = _normalize_cli_name(raw_name)
        if name in REQUEST_ONLY_FLAGS:
            continue
        if not engine.has_flag(name):
            dropped.append(f"extra_cli.{name}")
            continue
        argv.extend(_emit(name, value))
    if engine.has_flag("--listen-ip"):
        argv.extend(["--listen-ip", host])
    if engine.has_flag("--listen-port"):
        argv.extend(["--listen-port", str(port)])
    return argv, dropped


def _opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _request_json(method: str, url: str, payload: dict[str, Any] | None = None, *, timeout: float) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"} if data is not None else {})
    with _opener().open(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_server(process: subprocess.Popen[Any], *, url: str = SERVER_URL, timeout: float = 15 * 60) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not answer"
    while time.monotonic() < deadline:
        code = process.poll()
        if code is not None:
            raise RuntimeError(f"sd-server exited during model load with code {code}")
        try:
            _request_json("GET", f"{url}/sdapi/v1/options", timeout=2.0)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(0.5)
    process.terminate()
    raise TimeoutError(f"sd-server model load timed out: {last_error}")


def start_recipe_server(recipe: str, cache_dir: Path) -> tuple[subprocess.Popen[Any], list[str], list[str]]:
    argv, dropped = recipe_server_argv(recipe, cache_dir)
    process = subprocess.Popen(argv)
    wait_for_server(process)
    return process, argv, dropped




def server_is_alive(process: subprocess.Popen[Any] | None) -> bool:
    return process is not None and process.poll() is None

def stop_recipe_server(process: subprocess.Popen[Any] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def server_generate(request: GenerateRequest, *, url: str = SERVER_URL) -> dict[str, Any]:
    request.validate()
    if request.init_image or request.control_net:
        raise ValueError("persistent server path currently supports text-to-image requests only")
    payload: dict[str, Any] = {
        "prompt": request.prompt,
        "negative_prompt": request.negative_prompt or "",
        "width": request.width,
        "height": request.height,
        "steps": request.steps,
        "cfg_scale": request.cfg_scale,
        "seed": request.seed if request.seed is not None else -1,
        "batch_size": request.batch_count or 1,
    }
    if request.sampling_method:
        payload["sampler_name"] = request.sampling_method
    if request.scheduler:
        payload["scheduler"] = request.scheduler
    payload = {key: value for key, value in payload.items() if value is not None}
    started = time.perf_counter()
    try:
        response = _request_json("POST", f"{url}/sdapi/v1/txt2img", payload, timeout=2 * 60 * 60)
    except (ConnectionError, OSError, urllib.error.URLError) as exc:
        raise ServerUnavailableError(f"sd-server is unavailable: {exc}") from exc
    duration_ms = int((time.perf_counter() - started) * 1000)
    return {
        "images": [base64.b64decode(value) for value in response.get("images") or []],
        "duration_ms": duration_ms,
        "endpoint": "/sdapi/v1/txt2img",
        "request": payload,
    }
