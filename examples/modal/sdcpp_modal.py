#!/usr/bin/env python3
"""Standalone CLI: stage weights on CPU, then call persistent Modal workers."""

from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

from sdcpp_hooks.cli import parse_argv
from sdcpp_hooks.contract import ValidationError
from sdcpp_hooks.deployed import (
    engine,
    ensure_deployed,
    gpu_function,
    storage_function,
)
from sdcpp_hooks.gpu import default_gpu_for_recipe, normalize_gpu
from sdcpp_hooks.hardware import format_host_summary
from sdcpp_hooks.hf_dataset import publish_image, trigger_pages_rebuild
from sdcpp_hooks.modal_meter import billed_remote, billed_service, cost_command, print_last_cost
from sdcpp_hooks.recipes import recipe_volume_status


MAX_PUT_BYTES = 64 * 1024 * 1024


def _runtime_gpu(recipe: str) -> str:
    raw = os.environ.get("SDCPP_GPU")
    if raw:
        return normalize_gpu(raw)
    return default_gpu_for_recipe(recipe)


def _print_prefetch_status(rows: list[dict]) -> None:
    paths = {str(row.get("path") or "") for row in rows}
    print("volume sdcpp-models")
    for item in recipe_volume_status(paths):
        mark = "complete" if item["complete"] else "missing"
        print(f"{item['recipe']}\t{mark}\t{item['have']}/{item['need']}")
        for uri in item["missing"]:
            print(f"  missing {uri}")


def _print_storage(rows: list[dict]) -> None:
    if not rows:
        print("volume sdcpp-models is empty")
        return
    for row in rows:
        print(f"{row['path']}\t{row['bytes']}")


def _put_payload(paths: list[str]) -> list[dict]:
    payload = []
    for raw in paths:
        path = Path(raw)
        data = path.read_bytes()
        if len(data) > MAX_PUT_BYTES:
            raise ValueError(
                f"{path} is too large for put; use pull or: "
                f"modal volume put sdcpp-models {path} uploads/{path.name}"
            )
        payload.append(
            {
                "path": f"uploads/{path.name}",
                "data": base64.b64encode(data).decode("ascii"),
            }
        )
    return payload


def main(argv: list[str] | None = None) -> int:
    command = parse_argv(sys.argv[1:] if argv is None else argv)
    if command.action == "cost":
        return cost_command(official=command.official)

    if command.action == "publish":
        return _publish(command)

    # Persistent deployments are the default execution model. This lookup is
    # cheap when they already exist; the first run deploys them once. Deployment
    # itself does not keep a GPU container alive because SDEngine has
    # min_containers=0.
    if not (command.action == "web" and command.dry_run):
        ensure_deployed()

    if command.action == "web":
        return _run_web(command)

    if command.action in {"pull", "prefetch", "ls", "put"}:
        with billed_service("storage"):
            if command.action in {"pull", "prefetch"}:
                if command.status:
                    _print_prefetch_status(
                        billed_remote(storage_function("list_storage"), name="ls")
                    )
                else:
                    print(f"cpu prefetch {len(command.uris)} file(s) onto volume sdcpp-models")
                    for row in billed_remote(
                        storage_function("pull"),
                        command.uris,
                        name="prefetch",
                    ):
                        print(f"{row['uri']} -> {row['path']} ({row['bytes']} bytes)")
            elif command.action == "put":
                try:
                    files = _put_payload(command.files)
                except (OSError, ValueError) as exc:
                    print(exc, file=sys.stderr)
                    return 2
                for row in billed_remote(
                    storage_function("put_files"),
                    files,
                    name="put",
                ):
                    print(f"{row['path']} ({row['bytes']} bytes)")
            else:
                _print_storage(
                    billed_remote(storage_function("list_storage"), name="ls")
                )
            print_last_cost()
        print_last_cost()
        return 0

    if command.action == "probe":
        with billed_service("probe"):
            caps = billed_remote(gpu_function("probe"), name="probe")
            print(caps["binary"])
            print(" ".join(caps["flags"]))
            print_last_cost()
        print_last_cost()
        return 0

    try:
        payload = command.to_payload()
    except (KeyError, ValidationError) as exc:
        print(exc, file=sys.stderr)
        return 2

    # CPU phase: ensure every required model is present and committed to the
    # shared Volume. This call is synchronous; GPU is not touched before it
    # returns successfully.
    with billed_service("storage"):
        for row in billed_remote(
            storage_function("ensure_artifacts"),
            payload,
            name="ensure_artifacts",
        ):
            print(f"cpu storage {row['uri']} -> {row['path']} ({row['bytes']} bytes)")
        print_last_cost()
    print_last_cost()

    # GPU phase starts only after the CPU staging phase completed. The deployed
    # Cls is persistent, while its containers still autoscale to zero when idle.
    gpu = _runtime_gpu(command.recipe)
    print(f"gpu {gpu}")
    with billed_service("gpu"):
        remote_engine = engine(gpu=gpu)
        result = billed_remote(
            remote_engine.generate,
            payload,
            name="generate",
            gpu=True,
            gpu_name=gpu,
        )
        host = result.get("host")
        if isinstance(host, dict):
            host["modal_gpu"] = gpu
        print_last_cost()
    print_last_cost()
    if not result["images"]:
        print(f"no images returned; dropped={result['dropped_fields']}", file=sys.stderr)
        return 1
    dest = Path(command.output)
    dest.write_bytes(base64.b64decode(result["images"][0]))
    print(f"wrote {dest} via {result['engine_id']}")
    summary = format_host_summary(result.get("host"), result.get("duration_ms"))
    if summary:
        print(summary)
    if result["dropped_fields"]:
        print("dropped_fields:", ", ".join(result["dropped_fields"]))
    if command.publish:
        return _publish(command, image_path=dest, payload=result)
    return 0


def _resolved_generate_fields(command) -> dict:
    try:
        return command.to_payload()
    except (KeyError, ValidationError):
        return {}


def _publish(command, image_path: Path | None = None, payload: dict | None = None) -> int:
    path = Path(image_path or command.image)
    payload = payload or {}
    resolved = _resolved_generate_fields(command) if command.action == "generate" else {}
    host = payload.get("host") if isinstance(payload.get("host"), dict) else {}
    extra = {
        "source": "sdcpp-modal",
        "recipe": command.recipe,
        "model_uri": payload.get("model") or resolved.get("model") or command.model,
        "sd_cli_argv": payload.get("argv"),
        "duration_ms": payload.get("duration_ms"),
        "gpu_name": host.get("gpu_name"),
        "cuda_version": host.get("cuda_version"),
        "torch_version": host.get("torch_version"),
        "driver_version": host.get("driver_version"),
        "nvidia_driver": host.get("driver_version"),
        "sd_cli_version": host.get("sd_cli_version"),
        "sd_cli_binary": host.get("sd_cli_binary"),
        "python_version": host.get("python_version"),
        "platform": host.get("platform"),
        "modal_gpu": host.get("modal_gpu"),
        "sdcpp_image": host.get("sdcpp_image"),
        "gpu_memory_mib": host.get("gpu_memory_mib"),
        "torch_cuda": host.get("torch_cuda"),
        "runtime": host.get("runtime"),
        "cost": payload.get("cost"),
    }
    extra.update({key: value for key, value in host.items() if extra.get(key) in (None, "")})
    extra = {key: value for key, value in extra.items() if value not in (None, "")}
    record = publish_image(
        path,
        model=command.model_id or command.recipe or "other",
        prompt=command.prompt,
        negative_prompt=command.negative_prompt,
        seed=int(payload.get("seed") or command.seed),
        steps=int(payload.get("steps") or resolved.get("steps") or command.steps or 0) or None,
        width=int(payload.get("width") or resolved.get("width") or command.width or 0) or None,
        height=int(payload.get("height") or resolved.get("height") or command.height or 0) or None,
        cfg_scale=float(payload.get("cfg_scale") or resolved.get("cfg_scale") or command.cfg_scale or 0) or None,
        extra=extra,
        duration_ms=payload.get("duration_ms"),
        gpu_name=str(host.get("gpu_name") or ""),
        cuda_version=str(host.get("cuda_version") or ""),
        torch_version=str(host.get("torch_version") or ""),
    )
    print(f"published {record.path} as {record.model}/{record.id}")
    summary = format_host_summary(host, record.duration_ms)
    if summary:
        print(summary)
    if trigger_pages_rebuild():
        print("triggered gallery Pages rebuild")
    return 0


def _run_web(command) -> int:
    import webbrowser

    import uvicorn

    host = str(command.host).strip()
    loopback = host.lower() in {"127.0.0.1", "localhost", "::1"}
    if not loopback and not os.environ.get("SDCPP_WEB_TOKEN"):
        print("Refusing non-loopback Web binding without SDCPP_WEB_TOKEN authentication.", file=sys.stderr)
        print("Set SDCPP_WEB_TOKEN to protect the Web UI and API.", file=sys.stderr)
        return 2
    if command.dry_run:
        os.environ["SDCPP_WEB_DRY_RUN"] = "1"
    url = f"http://{host}:{command.port}"
    print(f"sdcpp-modal web → {url}")
    if command.dry_run:
        print("Local FastAPI dry-run; Modal deployment is not required.")
    else:
        print("Local FastAPI. CPU/GPU jobs call persistent deployed Modal apps.")
    if command.open_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    uvicorn.run("web.server:app", host=host, port=command.port, reload=False, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
