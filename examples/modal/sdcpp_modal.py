#!/usr/bin/env python3
"""Standalone CLI: pull weights onto Modal storage, then run sd-cli on GPU."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from app import (
    GPU,
    SDEngine,
    ensure_artifacts,
    gpu_app,
    list_storage,
    probe,
    pull,
    put_files,
    storage_app,
)
from sdcpp_hooks.cli import parse_argv
from sdcpp_hooks.contract import ValidationError
from sdcpp_hooks.hardware import format_host_summary
from sdcpp_hooks.hf_dataset import publish_image, trigger_pages_rebuild
from sdcpp_hooks.modal_meter import billed_app, billed_remote, cost_command, print_last_cost


MAX_PUT_BYTES = 64 * 1024 * 1024


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

    if command.action in {"pull", "ls", "put"}:
        with billed_app(storage_app, "storage"):
            if command.action == "pull":
                for row in billed_remote(pull, command.uris, name="pull"):
                    print(f"{row['uri']} -> {row['path']} ({row['bytes']} bytes)")
            elif command.action == "put":
                try:
                    files = _put_payload(command.files)
                except (OSError, ValueError) as exc:
                    print(exc, file=sys.stderr)
                    return 2
                for row in billed_remote(put_files, files, name="put"):
                    print(f"{row['path']} ({row['bytes']} bytes)")
            else:
                _print_storage(billed_remote(list_storage, name="ls"))
            print_last_cost()
        print_last_cost()
        return 0

    if command.action == "publish":
        return _publish(command)

    if command.action == "probe":
        with billed_app(gpu_app, "probe"):
            caps = billed_remote(probe, name="probe")
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

    with billed_app(storage_app, "storage"):
        for row in billed_remote(ensure_artifacts, payload, name="ensure_artifacts"):
            print(f"cpu storage {row['uri']} -> {row['path']} ({row['bytes']} bytes)")
        print_last_cost()
    print_last_cost()

    print(f"gpu {GPU}")
    with billed_app(gpu_app, "gpu"):
        result = billed_remote(SDEngine().generate, payload, name="generate", gpu=True)
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


if __name__ == "__main__":
    raise SystemExit(main())
