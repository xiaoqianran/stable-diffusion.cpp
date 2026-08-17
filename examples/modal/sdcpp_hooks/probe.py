from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_BINARIES = (
    "/sd-cli",
    "/sd.cpp/bin/sd-cli",
    "sd-cli",
)
DEFAULT_SERVER_BINARIES = (
    "/sd-server",
    "/sd.cpp/bin/sd-server",
    "sd-server",
)


def _probe_help(
    binaries: tuple[str, ...],
    *,
    required_any: tuple[str, ...],
    label: str,
) -> tuple[str, str]:
    errors: list[str] = []
    for binary in binaries:
        if binary.startswith("/") and not Path(binary).exists():
            errors.append(f"{binary}: missing")
            continue
        try:
            completed = subprocess.run(
                [binary, "-h"],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (FileNotFoundError, OSError) as exc:
            errors.append(f"{binary}: {exc}")
            continue
        text = completed.stdout or completed.stderr
        if any(marker in text for marker in required_any):
            return text, binary
        errors.append(f"{binary}: help text missing expected flags")
    raise FileNotFoundError(f"{label} not found: " + "; ".join(errors))


def probe_cli_help(binaries: tuple[str, ...] = DEFAULT_BINARIES) -> tuple[str, str]:
    return _probe_help(
        binaries,
        required_any=("--prompt", "--model"),
        label="sd-cli",
    )


def probe_server_help(
    binaries: tuple[str, ...] = DEFAULT_SERVER_BINARIES,
) -> tuple[str, str]:
    return _probe_help(
        binaries,
        required_any=("--listen-port", "--diffusion-model", "--model"),
        label="sd-server",
    )
