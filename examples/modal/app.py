"""Modal workers for the standalone sdcpp CLI.

Downloads run only on CPU and write volume sdcpp-models.
GPU containers load those cached files and run sd-cli, then scale to zero.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

import modal

from sdcpp_hooks.artifacts import (
    collect_fetchable_uris,
    list_cached_artifacts,
    resolve_artifacts,
)
from sdcpp_hooks.contract import GenerateRequest
from sdcpp_hooks.hardware import collect_run_environment
from sdcpp_hooks.hooks import generate, use_engine, use_models
from sdcpp_hooks.meter import ContainerMeter
from sdcpp_hooks.probe import probe_cli_help
from sdcpp_hooks.recipes import convert_jobs
from sdcpp_hooks.runner import EngineError, run_cli


HERE = Path(__file__).resolve().parent
IMAGE_TAG = os.environ.get(
    "SDCPP_IMAGE",
    "ghcr.io/leejet/stable-diffusion.cpp:master-cuda",
)
def _gpu_name() -> str:
    raw = os.environ.get("SDCPP_GPU", "L40S")
    text = raw.strip().upper().replace(" ", "-")
    aliases = {
        "RTX-PRO-6000": "RTX6000",
        "RTXPRO6000": "RTX6000",
        "PRO-6000": "RTX6000",
        "PRO6000": "RTX6000",
    }
    gpu = aliases.get(text, text)
    if gpu.startswith("A10") or gpu.startswith("A100"):
        raise RuntimeError(
            f"SDCPP_GPU={raw!r} is blocked; use L4, L40S, or RTX6000 (RTX PRO 6000)"
        )
    return gpu


GPU = _gpu_name()
MODEL_ROOT = Path(os.environ.get("SDCPP_MODEL_ROOT", "/models"))
IDLE_SECONDS = int(os.environ.get("SDCPP_IDLE_SECONDS", "10"))

volume = modal.Volume.from_name("sdcpp-models", create_if_missing=True)


def _secret_exists(name: str) -> bool:
    if os.environ.get("MODAL_TASK_ID") or not name:
        return False
    try:
        completed = subprocess.run(
            ["modal", "secret", "list"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (FileNotFoundError, OSError):
        return False
    return name in completed.stdout


def _with_hooks(image: modal.Image) -> modal.Image:
    if hasattr(image, "add_local_python_source"):
        return image.add_local_python_source("sdcpp_hooks")
    return image.add_local_dir(str(HERE / "sdcpp_hooks"), remote_path="/pkg/sdcpp_hooks").env(
        {"PYTHONPATH": "/pkg"}
    )


def _cpu_image() -> modal.Image:
    return _with_hooks(
        modal.Image.debian_slim(python_version="3.12")
        .apt_install("aria2")
        .pip_install("huggingface_hub")
    )


def _cuda_image() -> modal.Image:
    return _with_hooks(
        modal.Image.from_registry(IMAGE_TAG, add_python="3.12")
        .entrypoint([])
        .env({"SDCPP_GPU": GPU, "SDCPP_IMAGE": IMAGE_TAG})
    )


def _convert_image() -> modal.Image:
    script = HERE.parent.parent / "scripts" / "convert_fp8_scale_to_bf16.py"
    return _with_hooks(
        modal.Image.debian_slim(python_version="3.12")
        .pip_install("numpy", "safetensors", "torch")
        .add_local_file(str(script), remote_path="/opt/convert_fp8_scale_to_bf16.py")
    )


def _secrets() -> list[modal.Secret]:
    local = {
        key: os.environ[key]
        for key in ("HF_TOKEN", "CIVITAI_TOKEN")
        if os.environ.get(key)
    }
    if local:
        return [modal.Secret.from_dict(local)]
    name = os.environ.get("SDCPP_SECRET", "sdcpp-tokens")
    if _secret_exists(name):
        return [modal.Secret.from_name(name)]
    return []


def _idle_kwargs() -> dict:
    return {
        "min_containers": 0,
        "buffer_containers": 0,
        "scaledown_window": IDLE_SECONDS,
    }


def _pull_workers() -> int:
    raw = os.environ.get("SDCPP_PULL_WORKERS", "4")
    try:
        return max(1, int(raw))
    except ValueError:
        return 4


def _pull_uris(uris: list[str]) -> list[dict]:
    labeled = {f"uri_{index}": uri for index, uri in enumerate(uris)}
    resolved = resolve_artifacts(
        labeled,
        cache_dir=MODEL_ROOT,
        artifact_fields=set(labeled),
        allow_download=True,
        max_workers=_pull_workers(),
    )
    rows = []
    for index, uri in enumerate(uris):
        path = resolved[f"uri_{index}"]
        rows.append(
            {
                "uri": uri,
                "path": str(path),
                "bytes": path.stat().st_size if path.exists() else 0,
            }
        )
    return rows


_APP_TAGS = {"project": "sdcpp-modal"}
storage_app = modal.App("sdcpp-storage", image=_cpu_image(), tags={**_APP_TAGS, "role": "storage"})
convert_app = modal.App("sdcpp-convert", image=_convert_image(), tags={**_APP_TAGS, "role": "convert"})
gpu_app = modal.App("sdcpp-cli", image=_cuda_image(), tags={**_APP_TAGS, "role": "gpu"})


@storage_app.function(
    timeout=2 * 60 * 60,
    volumes={str(MODEL_ROOT): volume},
    secrets=_secrets(),
    **_idle_kwargs(),
)
def pull(uris: list[str]) -> list[dict]:
    volume.reload()
    rows = _pull_uris(uris)
    volume.commit()
    return rows


@storage_app.function(
    timeout=2 * 60 * 60,
    volumes={str(MODEL_ROOT): volume},
    secrets=_secrets(),
    **_idle_kwargs(),
)
def ensure_artifacts(payload: dict) -> list[dict]:
    request = GenerateRequest.from_dict(payload)
    request.validate()
    uris = collect_fetchable_uris(request.to_dict(), request.extra_cli)
    if not uris:
        return []
    volume.reload()
    rows = _pull_uris(uris)
    volume.commit()
    return rows


@storage_app.function(volumes={str(MODEL_ROOT): volume}, **_idle_kwargs())
def list_storage() -> list[dict]:
    volume.reload()
    return list_cached_artifacts(MODEL_ROOT)


@storage_app.function(
    timeout=30 * 60,
    volumes={str(MODEL_ROOT): volume},
    **_idle_kwargs(),
)
def put_files(files: list[dict]) -> list[dict]:
    volume.reload()
    rows = []
    for item in files:
        rel = Path(item["path"])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"invalid remote path {item['path']!r}")
        dest = MODEL_ROOT / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(base64.b64decode(item["data"]))
        rows.append(
            {
                "path": dest.relative_to(MODEL_ROOT).as_posix(),
                "bytes": dest.stat().st_size,
            }
        )
    volume.commit()
    return rows


@convert_app.function(
    timeout=2 * 60 * 60,
    memory=32768,
    volumes={str(MODEL_ROOT): volume},
    **_idle_kwargs(),
)
def convert_fp8_to_bf16(src: str, dest: str) -> dict:
    volume.reload()
    source = Path(src)
    output = Path(dest)
    if not source.exists() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing fp8 weights {src}; pull --recipe ideogram4 first")
    if output.exists() and output.stat().st_size > 0:
        return {"path": dest, "bytes": output.stat().st_size, "skipped": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "/opt/convert_fp8_scale_to_bf16.py",
            "--input",
            src,
            "--output",
            dest,
            "--overwrite",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or f"fp8 to bf16 failed for {src}")
    volume.commit()
    return {"path": dest, "bytes": output.stat().st_size, "skipped": False}


@gpu_app.function(
    gpu=GPU,
    timeout=2 * 60 * 60,
    volumes={str(MODEL_ROOT): volume},
    **_idle_kwargs(),
)
def convert_to_gguf(src: str, dest: str, rules: str) -> dict:
    volume.reload()
    source = Path(src)
    output = Path(dest)
    if not source.exists() or source.stat().st_size <= 0:
        raise FileNotFoundError(f"missing bf16 weights {src}")
    if output.exists() and output.stat().st_size > 0:
        return {"path": dest, "bytes": output.stat().st_size, "skipped": True}
    help_text, binary = probe_cli_help()
    output.parent.mkdir(parents=True, exist_ok=True)
    argv = [binary, "-M", "convert", "-m", src, "-o", dest, "--tensor-type-rules", rules, "-v"]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if completed.stdout:
        print(completed.stdout, flush=True)
    if completed.stderr:
        print(completed.stderr, flush=True)
    if completed.returncode != 0 or not output.exists():
        raise RuntimeError(completed.stderr.strip() or completed.stdout.strip() or "sd-cli convert failed")
    volume.commit()
    return {"path": dest, "bytes": output.stat().st_size, "skipped": False}


def ideogram4_convert_plan(quant: str = "q8_0") -> list[dict[str, str]]:
    return convert_jobs("ideogram4", MODEL_ROOT, quant=quant)


@gpu_app.function(**_idle_kwargs())
def probe() -> dict:
    help_text, binary = probe_cli_help()
    engine = use_engine(help_text=help_text, binary=binary)
    return {
        "binary": binary,
        "flags": sorted(name for name in engine.flags if name.startswith("--")),
    }


@gpu_app.cls(
    gpu=GPU,
    timeout=2 * 60 * 60,
    volumes={str(MODEL_ROOT): volume},
    **_idle_kwargs(),
)
class SDEngine:
    @modal.enter()
    def start(self) -> None:
        self._meter = ContainerMeter.start("SDEngine", gpu=GPU)
        help_text, binary = probe_cli_help()
        self.engine = use_engine(help_text=help_text, binary=binary)
        self.binary = binary

    @modal.exit()
    def stop(self) -> None:
        meter = getattr(self, "_meter", None)
        if meter is not None:
            meter.stop()

    @modal.method()
    def generate(self, payload: dict) -> dict:
        request = GenerateRequest.from_dict(payload)
        request.validate()
        volume.reload()
        models = use_models(request, cache_dir=MODEL_ROOT, allow_download=False)
        output_path = Path("/tmp/sdcpp-output.png")
        try:
            result = generate(
                request,
                engine=self.engine,
                models=models,
                run=run_cli,
                output_path=output_path,
            )
        except EngineError as exc:
            raise RuntimeError(str(exc)) from None
        host = dict(result.host or collect_run_environment(
            help_text=self.engine.raw_help,
            binary=self.binary,
        ))
        host.setdefault("modal_gpu", GPU)
        host.setdefault("sdcpp_image", IMAGE_TAG)
        return {
            "images": [base64.b64encode(image).decode("ascii") for image in result.images],
            "argv": result.argv,
            "dropped_fields": result.dropped_fields,
            "engine_id": result.engine_id,
            "seed": result.seed,
            "duration_ms": result.duration_ms,
            "host": host,
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "cfg_scale": request.cfg_scale,
            "model": request.model or request.diffusion_model,
        }
