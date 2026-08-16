from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urlparse

from .fast_fetch import fast_fetch


ARTIFACT_FIELDS = {
    "model",
    "diffusion_model",
    "uncond_diffusion_model",
    "vae",
    "clip_l",
    "clip_g",
    "clip_vision",
    "t5xxl",
    "llm",
    "llm_vision",
    "lora_dir",
    "lora",
    "init_image",
    "control_net",
    "taesd",
    "upscale_model",
}

FetchFn = Callable[[str, Path, dict[str, str]], Path]
TokenFn = Callable[[str], str | None]


@dataclass
class ArtifactRef:
    scheme: str
    raw: str
    repo_id: str | None = None
    revision: str | None = None
    path: str | None = None
    version_id: str | None = None
    local_path: Path | None = None
    url: str | None = None

    @classmethod
    def parse(cls, value: str) -> ArtifactRef:
        text = value.strip()
        if text.startswith("hf://"):
            return cls._parse_hf(text)
        if text.startswith("civitai://"):
            version_id = text[len("civitai://") :].strip("/")
            if version_id.startswith("model/"):
                version_id = version_id.split("/", 1)[1]
            return cls(scheme="civitai", raw=text, version_id=version_id)
        if text.startswith("https://") or text.startswith("http://"):
            return cls(
                scheme="https" if text.startswith("https://") else "http",
                raw=text,
                url=text,
            )
        return cls(scheme="file", raw=text, local_path=Path(text))

    @classmethod
    def _parse_hf(cls, text: str) -> ArtifactRef:
        rest = text[len("hf://") :].lstrip("/")
        parts = rest.split("/")
        if len(parts) < 3:
            raise ValueError(f"hf URI must be hf://org/repo/path, got {text!r}")
        org, repo_part, *path_parts = parts
        revision = "main"
        if "@" in repo_part:
            repo_part, revision = repo_part.split("@", 1)
        return cls(
            scheme="hf",
            raw=text,
            repo_id=f"{org}/{repo_part}",
            revision=revision,
            path="/".join(path_parts),
        )

    def download_url(self, hf_endpoint: str = "https://huggingface.co") -> str:
        if self.scheme == "hf":
            base = hf_endpoint.rstrip("/")
            return f"{base}/{self.repo_id}/resolve/{self.revision}/{self.path}"
        if self.scheme == "civitai":
            return f"https://civitai.com/api/download/models/{self.version_id}"
        if self.scheme in {"http", "https"} and self.url:
            return self.url
        raise ValueError(f"{self.raw} is a local path")

    def cache_path(self, cache_dir: Path) -> Path:
        if self.scheme == "hf":
            rel = Path(self.path or "model.bin")
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe hf path {self.path!r}")
            return cache_dir.joinpath(
                "hf",
                *self.repo_id.split("/"),
                self.revision or "main",
                *rel.parts,
            )
        if self.scheme == "civitai":
            return cache_dir / "civitai" / str(self.version_id) / "model.bin"
        if self.scheme in {"http", "https"} and self.url:
            parsed = urlparse(self.url)
            filename = Path(parsed.path).name or "download.bin"
            return cache_dir / "url" / filename
        if self.local_path is not None:
            return self.local_path
        raise ValueError(f"cannot place {self.raw}")


class ArtifactMissingError(FileNotFoundError):
    pass


def is_fetchable(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip()
    return text.startswith(("hf://", "civitai://", "https://", "http://"))


def collect_fetchable_uris(
    values: Mapping[str, object],
    extra_cli: Mapping[str, object] | None = None,
) -> list[str]:
    uris: list[str] = []
    seen: set[str] = set()
    for key, raw in values.items():
        if key not in ARTIFACT_FIELDS or not is_fetchable(raw):
            continue
        text = str(raw).strip()
        if text not in seen:
            seen.add(text)
            uris.append(text)
    for raw in (extra_cli or {}).values():
        if not is_fetchable(raw):
            continue
        text = str(raw).strip()
        if text not in seen:
            seen.add(text)
            uris.append(text)
    return uris


def default_token_for_url(url: str) -> str | None:
    if "huggingface.co" in url or "hf-mirror.com" in url:
        return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if "civitai.com" in url:
        return os.environ.get("CIVITAI_TOKEN")
    return None


def default_fetch(url: str, dest: Path, headers: Mapping[str, str]) -> Path:
    return fast_fetch(url, dest, headers)


def cache_is_complete(dest: Path) -> bool:
    if not dest.exists() or dest.stat().st_size <= 0:
        return False
    return not dest.with_name(dest.name + ".aria2").exists()


def resolve_artifacts(
    values: Mapping[str, object],
    cache_dir: Path,
    fetch: FetchFn | None = None,
    token_for_url: TokenFn | None = None,
    hf_endpoint: str | None = None,
    artifact_fields: Iterable[str] = ARTIFACT_FIELDS,
    allow_download: bool = True,
    max_workers: int = 1,
) -> dict[str, Path]:
    cache_dir = Path(cache_dir)
    fetch = fetch or default_fetch
    token_for_url = token_for_url or default_token_for_url
    endpoint = hf_endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    resolved: dict[str, Path] = {}
    pending: list[tuple[str, Path, str, dict[str, str]]] = []

    for key, raw in values.items():
        if key not in artifact_fields or raw in (None, ""):
            continue
        ref = ArtifactRef.parse(str(raw))
        if ref.scheme == "file":
            resolved[key] = Path(ref.local_path or raw)
            continue

        dest = ref.cache_path(cache_dir)
        if cache_is_complete(dest):
            resolved[key] = dest
            continue

        if not allow_download:
            raise ArtifactMissingError(
                f"{raw} is not on volume storage; pull it on CPU first"
            )

        url = ref.download_url(hf_endpoint=endpoint)
        headers = {"User-Agent": "sdcpp-modal-cli/0.1"}
        token = token_for_url(url)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        pending.append((key, dest, url, headers))

    def _download(item: tuple[str, Path, str, dict[str, str]]) -> tuple[str, Path]:
        key, dest, url, headers = item
        return key, fetch(url, dest, headers)

    if pending:
        workers = max(1, int(max_workers))
        if workers == 1 or len(pending) == 1:
            for item in pending:
                key, path = _download(item)
                resolved[key] = path
        else:
            with ThreadPoolExecutor(max_workers=min(workers, len(pending))) as pool:
                for key, path in pool.map(_download, pending):
                    resolved[key] = path

    return resolved


def list_cached_artifacts(cache_dir: Path) -> list[dict[str, int | str]]:
    root = Path(cache_dir)
    if not root.exists():
        return []
    listed: list[dict[str, int | str]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            listed.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                }
            )
    return listed
