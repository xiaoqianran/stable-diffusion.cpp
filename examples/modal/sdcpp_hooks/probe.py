from __future__ import annotations

import subprocess
from pathlib import Path


DEFAULT_BINARIES = (
    "/sd-cli",
    "/sd.cpp/bin/sd-cli",
    "sd-cli",
)


def probe_cli_help(binaries: tuple[str, ...] = DEFAULT_BINARIES) -> tuple[str, str]:
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
        if "--prompt" in text or "--model" in text:
            return text, binary
        errors.append(f"{binary}: help text had no --prompt/--model")
    raise FileNotFoundError("sd-cli not found: " + "; ".join(errors))
