from pathlib import Path

import pytest

from sdcpp_hooks.artifacts import (
    ArtifactMissingError,
    ArtifactRef,
    cache_is_complete,
    collect_fetchable_uris,
    is_fetchable,
    resolve_artifacts,
)


def test_is_fetchable_accepts_remote_uris_only():
    assert is_fetchable("hf://org/repo/a.safetensors")
    assert is_fetchable("civitai://128713")
    assert is_fetchable("https://example.com/a.safetensors")
    assert not is_fetchable("/models/local.safetensors")
    assert not is_fetchable(True)


def test_parse_huggingface_uri_with_optional_revision():
    ref = ArtifactRef.parse(
        "hf://stable-diffusion-v1-5/stable-diffusion-v1-5@main/v1-5-pruned-emaonly.safetensors"
    )

    assert ref.scheme == "hf"
    assert ref.repo_id == "stable-diffusion-v1-5/stable-diffusion-v1-5"
    assert ref.revision == "main"
    assert ref.path == "v1-5-pruned-emaonly.safetensors"


def test_parse_civitai_version_uri():
    ref = ArtifactRef.parse("civitai://128713")

    assert ref.scheme == "civitai"
    assert ref.version_id == "128713"


def test_parse_https_and_local_paths(tmp_path):
    local = tmp_path / "model.safetensors"
    local.write_bytes(b"weights")

    assert ArtifactRef.parse("https://example.com/a.safetensors").scheme == "https"
    assert ArtifactRef.parse(str(local)).scheme == "file"
    assert ArtifactRef.parse(str(local)).local_path == local


def test_resolve_artifacts_skips_existing_cache_and_downloads_missing(tmp_path):
    cache = tmp_path / "cache"
    existing = (
        cache
        / "hf"
        / "org"
        / "repo"
        / "main"
        / "already.safetensors"
    )
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"cached")

    fetched = {}

    def fetch(url, dest, headers):
        fetched[url] = (dest, headers)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"downloaded")
        return dest

    resolved = resolve_artifacts(
        {
            "model": "hf://org/repo/already.safetensors",
            "vae": "https://example.com/vae.safetensors",
            "prompt": "ignored",
        },
        cache_dir=cache,
        fetch=fetch,
        token_for_url=lambda url: "secret" if "example.com" in url else None,
    )

    assert resolved["model"] == existing
    assert resolved["vae"].read_bytes() == b"downloaded"
    assert "prompt" not in resolved
    assert list(fetched) == ["https://example.com/vae.safetensors"]
    assert fetched["https://example.com/vae.safetensors"][1]["Authorization"] == "Bearer secret"


def test_resolve_artifacts_builds_hf_and_civitai_urls(tmp_path):
    seen = []

    def fetch(url, dest, headers):
        seen.append((url, headers))
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ok")
        return dest

    resolved = resolve_artifacts(
        {
            "model": "hf://org/repo@dev/weights.gguf",
            "lora": "civitai://128713",
        },
        cache_dir=tmp_path,
        fetch=fetch,
        token_for_url=lambda url: "hf" if "huggingface.co" in url else "civitai",
    )

    hf_url, hf_headers = seen[0]
    civitai_url, civitai_headers = seen[1]
    assert hf_url == "https://huggingface.co/org/repo/resolve/dev/weights.gguf"
    assert hf_headers["Authorization"] == "Bearer hf"
    assert civitai_url == "https://civitai.com/api/download/models/128713"
    assert civitai_headers["Authorization"] == "Bearer civitai"
    assert resolved["model"].name == "weights.gguf"
    assert resolved["lora"].parent.name == "128713"


def test_resolve_artifacts_can_pull_arbitrary_uri_keys(tmp_path):
    def fetch(url, dest, headers):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"weights")
        return dest

    resolved = resolve_artifacts(
        {"uri_0": "hf://org/repo/a.safetensors"},
        cache_dir=tmp_path,
        fetch=fetch,
        artifact_fields={"uri_0"},
    )

    assert resolved["uri_0"].read_bytes() == b"weights"


def test_collect_fetchable_uris_from_request_fields_and_extra_cli():
    uris = collect_fetchable_uris(
        {
            "model": "hf://org/repo/model.safetensors",
            "prompt": "a cat",
            "vae": "https://example.com/vae.safetensors",
        },
        extra_cli={"--taesd": "hf://org/repo/tae.safetensors", "--verbose": True},
    )

    assert uris == [
        "hf://org/repo/model.safetensors",
        "https://example.com/vae.safetensors",
        "hf://org/repo/tae.safetensors",
    ]


def test_resolve_artifacts_refuses_download_when_cache_missing(tmp_path):
    def fetch(url, dest, headers):
        raise AssertionError("GPU path must not download")

    with pytest.raises(ArtifactMissingError, match="pull it on CPU"):
        resolve_artifacts(
            {"model": "hf://org/repo/missing.safetensors"},
            cache_dir=tmp_path,
            fetch=fetch,
            allow_download=False,
        )


def test_resolve_artifacts_can_load_cache_without_downloading(tmp_path):
    dest = tmp_path / "hf" / "org" / "repo" / "main" / "cached.safetensors"
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"cached")

    resolved = resolve_artifacts(
        {"model": "hf://org/repo/cached.safetensors"},
        cache_dir=tmp_path,
        fetch=lambda *args: (_ for _ in ()).throw(AssertionError("download")),
        allow_download=False,
    )

    assert resolved["model"] == dest


def test_hf_cache_path_keeps_nested_repo_paths(tmp_path):
    cond = ArtifactRef.parse(
        "hf://ideogram-ai/ideogram-4-fp8/transformer/diffusion_pytorch_model.safetensors"
    )
    uncond = ArtifactRef.parse(
        "hf://ideogram-ai/ideogram-4-fp8/unconditional_transformer/diffusion_pytorch_model.safetensors"
    )

    assert cond.cache_path(tmp_path) != uncond.cache_path(tmp_path)
    assert cond.cache_path(tmp_path).as_posix().endswith(
        "transformer/diffusion_pytorch_model.safetensors"
    )


def test_cache_is_complete_treats_aria2_control_file_as_partial(tmp_path):
    dest = tmp_path / "model.safetensors"
    dest.write_bytes(b"partial")
    assert cache_is_complete(dest)
    dest.with_name(dest.name + ".aria2").write_text("in-progress")
    assert not cache_is_complete(dest)


def test_resolve_artifacts_can_download_in_parallel(tmp_path):
    seen = []

    def fetch(url, dest, headers):
        seen.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(url.encode())
        return dest

    resolved = resolve_artifacts(
        {
            "model": "hf://org/repo/a.safetensors",
            "vae": "hf://org/repo/vae.safetensors",
        },
        cache_dir=tmp_path,
        fetch=fetch,
        max_workers=2,
    )

    assert resolved["model"].read_bytes() == b"https://huggingface.co/org/repo/resolve/main/a.safetensors"
    assert resolved["vae"].read_bytes() == b"https://huggingface.co/org/repo/resolve/main/vae.safetensors"
    assert len(seen) == 2


def test_collect_fetchable_uris_includes_uncond_diffusion_model():
    uris = collect_fetchable_uris(
        {
            "diffusion_model": "hf://org/repo/cond.safetensors",
            "uncond_diffusion_model": "hf://org/repo/uncond.safetensors",
            "llm": "hf://org/repo/llm.gguf",
        }
    )

    assert uris == [
        "hf://org/repo/cond.safetensors",
        "hf://org/repo/uncond.safetensors",
        "hf://org/repo/llm.gguf",
    ]
