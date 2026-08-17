from __future__ import annotations

import json
import sqlite3
import sys
import types
from pathlib import Path

import pytest

from sdcpp_hooks.artifacts import ArtifactMissingError, ArtifactRef, resolve_artifacts
from sdcpp_hooks.fast_fetch import fast_fetch
from sdcpp_hooks.web_events import Event, EventBus
from sdcpp_hooks.web_jobs import JobService


def test_remote_url_cache_uses_full_url_identity(tmp_path: Path):
    first = ArtifactRef.parse("https://a.example/models/model.bin?rev=1").cache_path(tmp_path)
    second = ArtifactRef.parse("https://b.example/other/model.bin?rev=1").cache_path(tmp_path)
    third = ArtifactRef.parse("https://a.example/models/model.bin?rev=2").cache_path(tmp_path)
    assert first != second
    assert first != third
    assert first.name.endswith("-model.bin")


def test_unverified_nonempty_remote_file_is_not_trusted_in_production(tmp_path: Path):
    ref = ArtifactRef.parse("https://example.com/model.bin")
    dest = ref.cache_path(tmp_path)
    dest.parent.mkdir(parents=True)
    dest.write_bytes(b"truncated-but-nonempty")
    with pytest.raises(ArtifactMissingError, match="verified volume"):
        resolve_artifacts({"model": ref.raw}, cache_dir=tmp_path, allow_download=False, pin_hf_revisions=False)


def test_fast_fetch_failure_never_replaces_good_destination(tmp_path: Path, monkeypatch):
    import sdcpp_hooks.fast_fetch as module

    dest = tmp_path / "model.bin"
    dest.write_bytes(b"known-good")
    monkeypatch.setattr(module, "download_with_aria2c", lambda *args, **kwargs: False)
    monkeypatch.setattr(module, "download_with_hf_cli", lambda *args, **kwargs: False)

    def broken(_url, partial, _headers):
        partial.write_bytes(b"half")
        raise OSError("network interrupted")

    monkeypatch.setattr(module, "download_with_urllib", broken)
    with pytest.raises(OSError, match="interrupted"):
        fast_fetch("https://example.com/model.bin", dest, {})
    assert dest.read_bytes() == b"known-good"
    assert not list(tmp_path.glob("*.partial-*"))
    assert not list(tmp_path.glob(".*.partial-*"))


def test_job_database_uses_wal_indexes_and_persists_modal_calls(tmp_path: Path):
    service = JobService(tmp_path, auto_recover=False)
    job = service.create_job([{"prompt": "cat"}], {"recipe": "sd15", "dry_run": True})
    image = service.get_job_detail(job["id"])["images"][0]
    service._store_call(job["id"], image["id"], "fc-123")
    service.close()

    conn = sqlite3.connect(tmp_path / "sdcpp-web.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    indexes = {row[1] for row in conn.execute("PRAGMA index_list(images)")}
    assert "idx_images_job_created" in indexes
    row = conn.execute("SELECT call_id,status FROM modal_calls WHERE image_id=?", (image["id"],)).fetchone()
    assert row == ("fc-123", "running")


def test_cancel_propagates_to_persisted_modal_function_call(tmp_path: Path, monkeypatch):
    service = JobService(tmp_path, auto_recover=False)
    job = service.create_job([{"prompt": "cat"}], {"recipe": "sd15"})
    image = service.get_job_detail(job["id"])["images"][0]
    service._store_call(job["id"], image["id"], "fc-cancel")
    cancelled: list[str] = []

    class FakeCall:
        def __init__(self, call_id: str): self.call_id = call_id
        def cancel(self): cancelled.append(self.call_id)

    class FakeFunctionCall:
        @staticmethod
        def from_id(call_id: str): return FakeCall(call_id)

    monkeypatch.setitem(sys.modules, "modal", types.SimpleNamespace(FunctionCall=FakeFunctionCall))
    result = service.cancel(job["id"])
    assert cancelled == ["fc-cancel"]
    assert result["status"] == "cancelled"
    assert service.get_image(image["id"])["status"] == "cancelled"


def test_event_history_is_bounded_and_prunable(monkeypatch):
    monkeypatch.setenv("SDCPP_EVENT_HISTORY", "16")
    bus = EventBus()
    for index in range(100):
        bus.publish(Event("tick", "job-1", {"index": index}))
    history = bus.history("job-1")
    assert len(history) == 16
    assert history[0].payload["index"] == 84
    bus.prune("job-1")
    assert bus.history("job-1") == []


def test_api_model_alias_and_jsonl_are_real_inputs(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.api import configure
    from web.server import app

    configure(tmp_path)
    client = TestClient(app)
    aliased = client.post("/api/jobs", json={"prompt": "cat", "model": "sd15", "dry_run": True})
    assert aliased.status_code == 200
    assert aliased.json()["recipe"] == "sd15"

    payload = "\n".join([
        json.dumps({"prompt": "cat", "seed": 42}),
        json.dumps({"prompt": "dog", "seed": 99, "count": 2}),
    ])
    created = client.post(
        "/api/jobs/from-file?model=sd15&dry_run=true",
        files={"file": ("prompts.jsonl", payload, "application/jsonl")},
    )
    assert created.status_code == 200
    detail = client.get(f"/api/jobs/{created.json()['id']}").json()
    assert detail["job"]["total_images"] == 3
    seeds = sorted(item["seed"] for item in detail["images"])
    assert seeds == [42, 99, 100]
    assert {item["prompt"] for item in detail["images"]} == {"cat", "dog"}


def test_api_rejects_excessive_dimensions(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.api import MAX_DIMENSION, configure
    from web.server import app

    configure(tmp_path)
    client = TestClient(app)
    response = client.post("/api/jobs", json={"prompt": "cat", "width": MAX_DIMENSION + 1, "dry_run": True})
    assert response.status_code == 422


def test_cpu_artifact_index_makes_gpu_resolution_offline(tmp_path: Path):
    calls: list[str] = []

    def fetch(url, dest, _headers):
        calls.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"weights")
        return dest

    source = "hf://org/repo/model.safetensors"
    staged = resolve_artifacts(
        {"model": source}, cache_dir=tmp_path, fetch=fetch, pin_hf_revisions=False
    )["model"]
    assert staged.is_file()
    # GPU/server style read-only resolution must use the persisted index and
    # never invoke a downloader or Hugging Face revision lookup.
    resolved = resolve_artifacts(
        {"model": source},
        cache_dir=tmp_path,
        fetch=None,
        allow_download=False,
        pin_hf_revisions=True,
    )["model"]
    assert resolved == staged
    assert len(calls) == 1


def test_default_image_comes_from_this_fork(monkeypatch):
    from sdcpp_hooks.runtime_identity import default_image_tag

    monkeypatch.delenv("SDCPP_IMAGE", raising=False)
    monkeypatch.delenv("SDCPP_IMAGE_SHA", raising=False)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    assert default_image_tag() == "ghcr.io/xiaoqianran/stable-diffusion.cpp:master-cuda"


def test_transient_identity_failure_does_not_trigger_redeploy(monkeypatch):
    import sdcpp_hooks.deployed as deployed

    deployed._READY.clear()
    deployed._STATUS.clear()
    monkeypatch.setenv("SDCPP_DEPLOY_SHA", "abc123")
    monkeypatch.setattr(deployed, "_is_deployed", lambda *args, **kwargs: True)

    def unavailable(*args, **kwargs):
        raise RuntimeError("control plane unavailable")

    monkeypatch.setattr(deployed, "_remote_identity", unavailable)
    with pytest.raises(RuntimeError, match="control plane"):
        deployed.ensure_deployed(force=False)


def test_non_loopback_web_requires_token(monkeypatch):
    from types import SimpleNamespace
    from sdcpp_modal import _run_web

    monkeypatch.delenv("SDCPP_WEB_TOKEN", raising=False)
    command = SimpleNamespace(host="0.0.0.0", port=7863, dry_run=True, open_browser=False)
    assert _run_web(command) == 2


def test_web_token_protects_ui_and_api(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.api import configure
    from web.server import app

    configure(tmp_path)
    monkeypatch.setenv("SDCPP_WEB_TOKEN", "secret-token")
    client = TestClient(app)
    assert client.get("/").status_code == 401
    assert client.get("/api/meta").status_code == 401
    response = client.get("/api/meta", headers={"Authorization": "Bearer secret-token"})
    assert response.status_code == 200


def test_ideogram_plain_text_is_adapted_for_user(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.api import configure
    from web.server import app

    configure(tmp_path)
    client = TestClient(app)
    created = client.post(
        "/api/jobs", json={"prompt": "a fluffy orange cat", "recipe": "ideogram4", "dry_run": True}
    )
    assert created.status_code == 200
    detail = client.get(f"/api/jobs/{created.json()['id']}").json()
    prompt = json.loads(detail["images"][0]["prompt"])
    assert prompt == {"high_level_description": "a fluffy orange cat"}


def test_required_server_model_flag_cannot_be_silently_dropped(tmp_path: Path, monkeypatch):
    from sdcpp_hooks import server_runtime
    from sdcpp_hooks.contract import GenerateRequest

    request = GenerateRequest(prompt="x", model="/models/m.gguf")
    monkeypatch.setattr(server_runtime, "apply_recipe", lambda *args, **kwargs: request)
    monkeypatch.setattr(server_runtime, "_resolve_request", lambda req, cache: req)

    class Engine:
        def has_flag(self, _name): return False

    monkeypatch.setattr(server_runtime, "probe_server_help", lambda: ("", "/sd-server"))
    monkeypatch.setattr(server_runtime, "use_engine", lambda **kwargs: Engine())
    with pytest.raises(RuntimeError, match="required model flag"):
        server_runtime.recipe_server_argv("sd15", tmp_path)


def test_direct_uvicorn_non_loopback_is_denied_without_token(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from web.api import configure
    from web.server import app

    configure(tmp_path)
    monkeypatch.delenv("SDCPP_WEB_TOKEN", raising=False)
    client = TestClient(app)
    response = client.get("/api/meta", headers={"Host": "192.168.1.50"})
    assert response.status_code == 403
    assert "SDCPP_WEB_TOKEN" in response.json()["detail"]


def test_verified_deployment_is_cached_in_process(monkeypatch):
    import sdcpp_hooks.deployed as deployed

    deployed._READY.clear()
    deployed._STATUS.clear()
    deployed._READY.update({deployed.STORAGE_APP_NAME, deployed.GPU_APP_NAME})
    monkeypatch.setattr(deployed, "_is_deployed", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected RPC")))
    deployed.ensure_deployed(force=False)


def test_runtime_identity_prefers_explicit_deploy_sha(monkeypatch):
    from sdcpp_hooks.runtime_identity import deployment_identity

    monkeypatch.setenv("SDCPP_DEPLOY_SHA", "deadbeef")
    assert deployment_identity(role="storage")["deploy_sha"] == "deadbeef"
