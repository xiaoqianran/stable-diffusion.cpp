from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping
from urllib.parse import urlparse

from .fast_fetch import fast_fetch


ARTIFACT_FIELDS = {
    "model", "diffusion_model", "uncond_diffusion_model", "vae", "clip_l", "clip_g",
    "clip_vision", "t5xxl", "llm", "llm_vision", "lora_dir", "lora", "init_image",
    "control_net", "taesd", "upscale_model",
}
FetchFn = Callable[[str, Path, dict[str, str]], Path]
TokenFn = Callable[[str], str | None]


@dataclass(frozen=True)
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
    def parse(cls, value: str) -> "ArtifactRef":
        text = value.strip()
        if text.startswith("hf://"):
            return cls._parse_hf(text)
        if text.startswith("civitai://"):
            version_id = text[len("civitai://") :].strip("/")
            if version_id.startswith("model/"):
                version_id = version_id.split("/", 1)[1]
            if not version_id:
                raise ValueError(f"invalid civitai URI {text!r}")
            return cls(scheme="civitai", raw=text, version_id=version_id)
        if text.startswith(("https://", "http://")):
            return cls(scheme="https" if text.startswith("https://") else "http", raw=text, url=text)
        return cls(scheme="file", raw=text, local_path=Path(text))

    @classmethod
    def _parse_hf(cls, text: str) -> "ArtifactRef":
        rest = text[len("hf://") :].lstrip("/")
        parts = rest.split("/")
        if len(parts) < 3:
            raise ValueError(f"hf URI must be hf://org/repo/path, got {text!r}")
        org, repo_part, *path_parts = parts
        revision = "main"
        if "@" in repo_part:
            repo_part, revision = repo_part.split("@", 1)
        if not repo_part or not revision:
            raise ValueError(f"invalid hf URI {text!r}")
        rel = Path(*path_parts)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe hf path {'/'.join(path_parts)!r}")
        return cls(scheme="hf", raw=text, repo_id=f"{org}/{repo_part}", revision=revision, path=rel.as_posix())

    def with_revision(self, revision: str) -> "ArtifactRef":
        return replace(self, revision=revision) if self.scheme == "hf" else self

    def download_url(self, hf_endpoint: str = "https://huggingface.co") -> str:
        if self.scheme == "hf":
            return f"{hf_endpoint.rstrip('/')}/{self.repo_id}/resolve/{self.revision}/{self.path}"
        if self.scheme == "civitai":
            return f"https://civitai.com/api/download/models/{self.version_id}"
        if self.scheme in {"http", "https"} and self.url:
            return self.url
        raise ValueError(f"{self.raw} is a local path")

    def cache_path(self, cache_dir: Path) -> Path:
        cache_dir = Path(cache_dir)
        if self.scheme == "hf":
            rel = Path(self.path or "model.bin")
            if rel.is_absolute() or ".." in rel.parts:
                raise ValueError(f"unsafe hf path {self.path!r}")
            return cache_dir.joinpath("hf", *str(self.repo_id).split("/"), self.revision or "main", *rel.parts)
        if self.scheme == "civitai":
            return cache_dir / "civitai" / str(self.version_id) / "model.bin"
        if self.scheme in {"http", "https"} and self.url:
            parsed = urlparse(self.url)
            filename = Path(parsed.path).name or "download.bin"
            digest = hashlib.sha256(self.url.encode("utf-8")).hexdigest()[:16]
            host = (parsed.hostname or "remote").replace(":", "_")
            return cache_dir / "url" / host / f"{digest}-{filename}"
        if self.local_path is not None:
            return self.local_path
        raise ValueError(f"cannot place {self.raw}")


class ArtifactMissingError(FileNotFoundError):
    pass


class ArtifactIntegrityError(IOError):
    pass


def is_fetchable(value: object) -> bool:
    return isinstance(value, str) and value.strip().startswith(("hf://", "civitai://", "https://", "http://"))


def collect_fetchable_uris(values: Mapping[str, object], extra_cli: Mapping[str, object] | None = None) -> list[str]:
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
        if is_fetchable(raw):
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


def _manifest_path(dest: Path) -> Path:
    return dest.with_name(dest.name + ".sdcpp.json")


def _index_path(cache_dir: Path, source: str) -> Path:
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    return Path(cache_dir) / ".sdcpp-index" / f"{digest}.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cache_is_complete(dest: Path) -> bool:
    """Low-level legacy completeness check; verified remote cache also needs manifest."""
    return dest.is_file() and dest.stat().st_size > 0 and not dest.with_name(dest.name + ".aria2").exists()


def _cache_matches(dest: Path, *, source: str) -> bool:
    if not cache_is_complete(dest):
        return False
    manifest_path = _manifest_path(dest)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        size = int(manifest["bytes"])
        expected_hash = str(manifest["sha256"])
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False
    if manifest.get("source") != source or size != dest.stat().st_size or not expected_hash:
        return False
    # Hashing multi-GB model weights on every request defeats warm-container
    # reuse. The hash is computed once before the manifest is atomically
    # published; subsequent reads trust the write-once manifest + exact size.
    # Operators can opt into a full on-read audit for diagnostics.
    if os.environ.get("SDCPP_VERIFY_ARTIFACT_HASH") == "1":
        return _sha256(dest) == expected_hash
    return True


def _write_manifest(dest: Path, *, source: str, resolved_source: str) -> dict[str, object]:
    if not cache_is_complete(dest):
        raise ArtifactIntegrityError(f"artifact is empty or partial: {dest}")
    manifest = {
        "version": 1,
        "source": source,
        "resolved_source": resolved_source,
        "bytes": dest.stat().st_size,
        "sha256": _sha256(dest),
        "created_at": int(time.time()),
    }
    target = _manifest_path(dest)
    tmp = target.with_name(f".{target.name}.partial-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)
    return manifest


def _read_index(cache_dir: Path, *, source: str) -> Path | None:
    path = _index_path(cache_dir, source)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = Path(str(payload["path"]))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        return None
    if not target.is_absolute():
        target = Path(cache_dir) / target
    return target if _cache_matches(target, source=source) else None


def _write_index(cache_dir: Path, *, source: str, dest: Path) -> None:
    root = Path(cache_dir)
    target = _index_path(root, source)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        stored = dest.relative_to(root).as_posix()
    except ValueError:
        stored = str(dest)
    payload = {"version": 1, "source": source, "path": stored, "updated_at": int(time.time())}
    tmp = target.with_name(f".{target.name}.partial-{os.getpid()}-{time.time_ns()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    os.replace(tmp, target)


def _pin_hf_revision(ref: ArtifactRef, *, endpoint: str, token: str | None) -> ArtifactRef:
    if ref.scheme != "hf" or ref.revision != "main":
        return ref
    if os.environ.get("SDCPP_ALLOW_MOVING_HF_REVISION") == "1":
        return ref
    try:
        from huggingface_hub import HfApi
        info = HfApi(endpoint=endpoint, token=token).model_info(str(ref.repo_id), revision="main")
        sha = str(info.sha or "").strip()
    except Exception as exc:
        raise ArtifactIntegrityError(
            f"cannot resolve immutable Hugging Face revision for {ref.repo_id}: {exc}; "
            "use an explicit @revision or set SDCPP_ALLOW_MOVING_HF_REVISION=1"
        ) from exc
    if not sha:
        raise ArtifactIntegrityError(f"Hugging Face returned no commit SHA for {ref.repo_id}")
    return ref.with_revision(sha)


def resolve_artifacts(
    values: Mapping[str, object],
    cache_dir: Path,
    fetch: FetchFn | None = None,
    token_for_url: TokenFn | None = None,
    hf_endpoint: str | None = None,
    artifact_fields: Iterable[str] = ARTIFACT_FIELDS,
    allow_download: bool = True,
    max_workers: int = 1,
    pin_hf_revisions: bool | None = None,
) -> dict[str, Path]:
    """Resolve remote refs to verified immutable cache files.

    CPU download paths resolve moving HF `main` once, write a content manifest,
    and persist source->immutable-path index metadata on the shared Volume. GPU
    read-only paths consume that index and never need Hugging Face network access.
    """
    cache_dir = Path(cache_dir)
    custom_fetch = fetch is not None
    fetch = fetch or default_fetch
    token_for_url = token_for_url or default_token_for_url
    endpoint = hf_endpoint or os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    if pin_hf_revisions is None:
        pin_hf_revisions = not custom_fetch
    resolved: dict[str, Path] = {}
    pending: list[tuple[str, ArtifactRef, Path, str, str, dict[str, str]]] = []

    for key, raw in values.items():
        if key not in artifact_fields or raw in (None, ""):
            continue
        original = ArtifactRef.parse(str(raw))
        if original.scheme == "file":
            resolved[key] = Path(original.local_path or str(raw))
            continue

        source_key = original.raw
        refresh_moving = original.scheme == "hf" and original.revision == "main" and os.environ.get("SDCPP_REFRESH_MOVING_HF") == "1"
        indexed = None if refresh_moving else _read_index(cache_dir, source=source_key)
        if indexed is not None:
            resolved[key] = indexed
            continue

        legacy_dest = original.cache_path(cache_dir)
        if custom_fetch and cache_is_complete(legacy_dest):
            resolved[key] = legacy_dest
            continue

        if not allow_download:
            raise ArtifactMissingError(f"{raw} is not on verified volume storage; pull it on CPU first")

        ref = original
        if pin_hf_revisions and ref.scheme == "hf":
            ref = _pin_hf_revision(
                ref,
                endpoint=endpoint,
                token=os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
            )
        dest = ref.cache_path(cache_dir)
        if _cache_matches(dest, source=source_key):
            _write_index(cache_dir, source=source_key, dest=dest)
            resolved[key] = dest
            continue
        url = ref.download_url(hf_endpoint=endpoint)
        headers = {"User-Agent": "sdcpp-modal-cli/0.3"}
        token = token_for_url(url)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        pending.append((key, ref, dest, source_key, url, headers))

    def _download(item: tuple[str, ArtifactRef, Path, str, str, dict[str, str]]) -> tuple[str, Path]:
        key, _ref, dest, source_key, url, headers = item
        path = Path(fetch(url, dest, headers))
        if path != dest:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.replace(path, dest)
        _write_manifest(dest, source=source_key, resolved_source=url)
        _write_index(cache_dir, source=source_key, dest=dest)
        return key, dest

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
        if (
            path.is_file()
            and ".sdcpp-index" not in path.parts
            and not path.name.endswith(".sdcpp.json")
            and ".partial-" not in path.name
        ):
            listed.append({"path": path.relative_to(root).as_posix(), "bytes": path.stat().st_size})
    return listed
