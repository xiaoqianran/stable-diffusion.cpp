from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .gallery import ImageRecord, image_id, record_paths, slugify_model, utc_now, write_sidecar


DEFAULT_DATASET = os.environ.get("SDCPP_GALLERY_DATASET", "seachen/stable-diffusion-cpp-gallery")
DEFAULT_GITHUB_REPO = os.environ.get("SDCPP_GITHUB_REPO", "xiaoqianran/stable-diffusion.cpp")
WORKFLOW_FILE = "gallery-pages.yml"


DATASET_CARD = """---
license: mit
task_categories:
  - text-to-image
tags:
  - stable-diffusion.cpp
  - sd-cli
  - gallery
pretty_name: stable-diffusion.cpp gallery
---

# stable-diffusion.cpp gallery

Public dataset of images generated with [stable-diffusion.cpp](https://github.com/xiaoqianran/stable-diffusion.cpp).

Each model family has its own folder. Unknown future models just get a new slug under `images/<model-id>/`.

```
images/
  sd15/<id>.png
  sd15/<id>.json
  flux/<id>.png
  wan/<id>.png
```

Sidecar JSON keeps prompt, seed, steps, size, run duration, GPU name, CUDA version, and torch version. GitHub Pages rebuilds a paginated gallery from these files.
"""


def _api(token: str | None = None):
    from huggingface_hub import HfApi

    return HfApi(token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))


def ensure_dataset(repo_id: str = DEFAULT_DATASET, token: str | None = None) -> str:
    api = _api(token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=DATASET_CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="docs: dataset card for sdcpp gallery",
    )
    models = Path(__file__).resolve().parent.parent / "gallery" / "models.json"
    api.upload_file(
        path_or_fileobj=models.read_bytes(),
        path_in_repo="models.json",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="chore: known model families",
    )
    return repo_id


def publish_image(
    image_path: Path,
    *,
    model: str,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int | None = None,
    steps: int | None = None,
    width: int | None = None,
    height: int | None = None,
    cfg_scale: float | None = None,
    extra: dict[str, Any] | None = None,
    repo_id: str = DEFAULT_DATASET,
    token: str | None = None,
    created_at: datetime | None = None,
    duration_ms: int | None = None,
    gpu_name: str = "",
    cuda_version: str = "",
    torch_version: str = "",
) -> ImageRecord:
    data = Path(image_path).read_bytes()
    model_id = slugify_model(model)
    when = created_at or utc_now()
    record_id = image_id(model_id, seed if seed is not None else -1, data, when)
    png_rel, json_rel = record_paths(model_id, record_id)
    extra = dict(extra or {})
    record = ImageRecord(
        id=record_id,
        model=model_id,
        path=png_rel,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        width=width,
        height=height,
        cfg_scale=cfg_scale,
        created_at=when.astimezone(timezone.utc).isoformat(),
        extra=extra,
        duration_ms=duration_ms if duration_ms is not None else extra.get("duration_ms"),
        gpu_name=gpu_name or str(extra.get("gpu_name") or ""),
        cuda_version=cuda_version or str(extra.get("cuda_version") or ""),
        torch_version=torch_version or str(extra.get("torch_version") or ""),
    )
    api = _api(token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    api.upload_file(
        path_or_fileobj=data,
        path_in_repo=png_rel,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"feat: add {model_id} image {record_id}",
    )
    api.upload_file(
        path_or_fileobj=(json.dumps(record.to_dict(), ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
        path_in_repo=json_rel,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"feat: add {model_id} sidecar {record_id}",
    )
    return record


def publish_local_bundle(root: Path, record: ImageRecord, image_bytes: bytes, repo_id: str = DEFAULT_DATASET) -> ImageRecord:
    write_sidecar(root, record, image_bytes)
    return record


def trigger_pages_rebuild(
    repo: str = DEFAULT_GITHUB_REPO,
    token: str | None = None,
    ref: str = "master",
) -> bool:
    github_token = token or os.environ.get("GITHUB_TOKEN")
    if not github_token:
        return False
    payload = json.dumps({"ref": ref}).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.github.com/repos/{repo}/actions/workflows/{WORKFLOW_FILE}/dispatches",
        data=payload,
        method="POST",
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {github_token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sdcpp-modal-cli",
        },
    )
    try:
        with urllib.request.urlopen(request) as response:
            return 200 <= response.status < 300
    except urllib.error.HTTPError:
        return False
