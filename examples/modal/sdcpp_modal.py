#!/usr/bin/env python3
"""Standalone CLI: pull weights onto Modal storage, then run sd-cli on GPU."""

from __future__ import annotations

import base64
import sys
from pathlib import Path

from app import SDEngine, gpu_app, list_storage, probe, pull, put_files, storage_app
from sdcpp_hooks.cli import parse_argv
from sdcpp_hooks.contract import ValidationError


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
    if command.action in {"pull", "ls", "put"}:
        with storage_app.run():
            if command.action == "pull":
                for row in pull.remote(command.uris):
                    print(f"{row['uri']} -> {row['path']} ({row['bytes']} bytes)")
                return 0
            if command.action == "put":
                try:
                    files = _put_payload(command.files)
                except (OSError, ValueError) as exc:
                    print(exc, file=sys.stderr)
                    return 2
                for row in put_files.remote(files):
                    print(f"{row['path']} ({row['bytes']} bytes)")
                return 0
            _print_storage(list_storage.remote())
            return 0

    if command.action == "probe":
        with gpu_app.run():
            caps = probe.remote()
            print(caps["binary"])
            print(" ".join(caps["flags"]))
            return 0

    try:
        payload = command.to_payload()
    except (KeyError, ValidationError) as exc:
        print(exc, file=sys.stderr)
        return 2

    with gpu_app.run():
        result = SDEngine().generate.remote(payload)
        if not result["images"]:
            print(f"no images returned; dropped={result['dropped_fields']}", file=sys.stderr)
            return 1
        dest = Path(command.output)
        dest.write_bytes(base64.b64decode(result["images"][0]))
        print(f"wrote {dest} via {result['engine_id']}")
        if result["dropped_fields"]:
            print("dropped_fields:", ", ".join(result["dropped_fields"]))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
