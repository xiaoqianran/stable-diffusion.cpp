from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPOSITORY_IMAGE = "ghcr.io/xiaoqianran/stable-diffusion.cpp"


def local_git_sha(root: Path | None = None) -> str:
    explicit = os.environ.get("SDCPP_DEPLOY_SHA") or os.environ.get("GITHUB_SHA")
    if explicit:
        return explicit.strip()
    cwd = root or Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def default_image_tag() -> str:
    explicit = os.environ.get("SDCPP_IMAGE")
    if explicit:
        return explicit
    sha = os.environ.get("SDCPP_IMAGE_SHA") or os.environ.get("GITHUB_SHA")
    if sha:
        return f"{REPOSITORY_IMAGE}:sha-{sha.strip()}-cuda"
    return f"{REPOSITORY_IMAGE}:master-cuda"


def deployment_identity(*, image: str | None = None, role: str = "") -> dict[str, str]:
    return {
        "deploy_sha": os.environ.get("SDCPP_DEPLOY_SHA") or local_git_sha(),
        "image": image or os.environ.get("SDCPP_IMAGE") or default_image_tag(),
        "role": role,
    }
