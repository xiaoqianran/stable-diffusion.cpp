from __future__ import annotations

import os
import platform
import subprocess
import sys


def _run(argv: list[str]) -> str:
    try:
        completed = subprocess.run(argv, capture_output=True, text=True, check=False, timeout=20)
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return ""
    return (completed.stdout or completed.stderr).strip()


def _nvidia_field(query: str) -> str:
    text = _run(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"])
    return text.splitlines()[0].strip() if text else ""


def _cuda_from_smi() -> str:
    text = _run(["nvidia-smi"])
    for line in text.splitlines():
        if "CUDA Version" in line:
            return line.split("CUDA Version:")[-1].split()[0]
    return _run(["nvcc", "--version"])


def _torch_info() -> dict[str, str]:
    try:
        import torch
    except Exception:
        return {
            "torch_version": "not installed (sd-cli uses ggml, not PyTorch)",
            "torch_cuda": "",
        }
    return {
        "torch_version": getattr(torch, "__version__", "unknown"),
        "torch_cuda": getattr(getattr(torch, "version", None), "cuda", "") or "",
    }


def _sd_cli_version(help_text: str = "") -> str:
    for line in help_text.splitlines():
        if "stable-diffusion.cpp" in line.lower() or line.lower().startswith("version"):
            return line.strip()
    return ""


def collect_run_environment(help_text: str = "", binary: str = "") -> dict[str, str | int | None]:
    torch_info = _torch_info()
    return {
        "gpu_name": _nvidia_field("name"),
        "gpu_memory_mib": _nvidia_field("memory.total"),
        "driver_version": _nvidia_field("driver_version"),
        "cuda_version": _cuda_from_smi(),
        "torch_version": torch_info["torch_version"],
        "torch_cuda": torch_info["torch_cuda"],
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "sd_cli_binary": binary,
        "sd_cli_version": _sd_cli_version(help_text),
        "modal_gpu": os.environ.get("SDCPP_GPU", ""),
        "sdcpp_image": os.environ.get("SDCPP_IMAGE", ""),
        "runtime": "stable-diffusion.cpp / ggml",
    }


def format_host_summary(host: dict | None, duration_ms: int | None = None) -> str:
    host = host or {}
    parts: list[str] = []
    if duration_ms:
        seconds = duration_ms / 1000.0
        parts.append(f"{seconds:.1f}s" if seconds >= 10 else f"{seconds:.2f}s")
    for key, label in (
        ("gpu_name", "gpu"),
        ("cuda_version", "cuda"),
        ("torch_version", "torch"),
        ("driver_version", "driver"),
        ("modal_gpu", "modal"),
    ):
        value = host.get(key)
        if value not in (None, ""):
            parts.append(f"{label} {value}")
    return " · ".join(parts)
