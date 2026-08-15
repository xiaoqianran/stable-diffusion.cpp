"""Modal workers for the standalone sdcpp CLI.

pull/ls use a slim CPU image and write to volume sdcpp-models.
probe/generate use the official CUDA image; only generate requests a GPU.
"""

from __future__ import annotations

import base64
import os
import subprocess
from pathlib import Path

import modal

from sdcpp_hooks.artifacts import list_cached_artifacts, resolve_artifacts
from sdcpp_hooks.contract import GenerateRequest
from sdcpp_hooks.hardware import collect_run_environment
from sdcpp_hooks.hooks import generate, use_engine, use_models
from sdcpp_hooks.probe import probe_cli_help
from sdcpp_hooks.runner import run_cli


HERE = Path(__file__).resolve().parent
IMAGE_TAG = os.environ.get(
    "SDCPP_IMAGE",
    "ghcr.io/leejet/stable-diffusion.cpp:master-cuda",
)
GPU = os.environ.get("SDCPP_GPU", "L4")
MODEL_ROOT = Path(os.environ.get("SDCPP_MODEL_ROOT", "/models"))

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
    return _with_hooks(modal.Image.debian_slim(python_version="3.12"))


def _cuda_image() -> modal.Image:
    return _with_hooks(
        modal.Image.from_registry(IMAGE_TAG, add_python="3.12")
        .entrypoint([])
        .env({"SDCPP_GPU": GPU, "SDCPP_IMAGE": IMAGE_TAG})
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


storage_app = modal.App("sdcpp-storage", image=_cpu_image())
gpu_app = modal.App("sdcpp-cli", image=_cuda_image())


@storage_app.function(
    timeout=2 * 60 * 60,
    volumes={str(MODEL_ROOT): volume},
    secrets=_secrets(),
)
def pull(uris: list[str]) -> list[dict]:
    volume.reload()
    labeled = {f"uri_{index}": uri for index, uri in enumerate(uris)}
    resolved = resolve_artifacts(
        labeled,
        cache_dir=MODEL_ROOT,
        artifact_fields=set(labeled),
    )
    volume.commit()
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


@storage_app.function(volumes={str(MODEL_ROOT): volume})
def list_storage() -> list[dict]:
    volume.reload()
    return list_cached_artifacts(MODEL_ROOT)


@storage_app.function(
    timeout=30 * 60,
    volumes={str(MODEL_ROOT): volume},
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


@gpu_app.function()
def probe() -> dict:
    help_text, binary = probe_cli_help()
    engine = use_engine(help_text=help_text, binary=binary)
    return {
        "binary": binary,
        "flags": sorted(name for name in engine.flags if name.startswith("--")),
    }


@gpu_app.cls(
    gpu=GPU,
    timeout=60 * 60,
    scaledown_window=120,
    volumes={str(MODEL_ROOT): volume},
    secrets=_secrets(),
)
class SDEngine:
    @modal.enter()
    def start(self) -> None:
        help_text, binary = probe_cli_help()
        self.engine = use_engine(help_text=help_text, binary=binary)
        self.binary = binary

    @modal.method()
    def generate(self, payload: dict) -> dict:
        request = GenerateRequest.from_dict(payload)
        request.validate()
        volume.reload()
        models = use_models(request, cache_dir=MODEL_ROOT)
        volume.commit()
        output_path = Path("/tmp/sdcpp-output.png")
        result = generate(
            request,
            engine=self.engine,
            models=models,
            run=run_cli,
            output_path=output_path,
        )
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
