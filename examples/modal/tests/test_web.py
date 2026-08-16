import time
from decimal import Decimal

import pytest

from sdcpp_hooks.gpu import PRO6000, default_gpu_for_recipe, normalize_gpu
from sdcpp_hooks.web_catalog import default_recipe, list_models
from sdcpp_hooks.web_jobs import JobService


def test_default_recipe_is_z_image_turbo():
    assert default_recipe() == "z-image-turbo"
    assert {item["id"] for item in list_models()} == {
        "ideogram4",
        "flux2-klein",
        "flux2-dev",
        "z-image-turbo",
        "sdxl-turbo",
        "sd2",
        "sd15",
    }


def test_normalize_gpu_blocks_a100():
    with pytest.raises(ValueError, match="blocked"):
        normalize_gpu("A100")
    assert normalize_gpu("rtx-pro-6000") == PRO6000
    assert normalize_gpu("RTX6000") == PRO6000


def test_runtime_gpu_uses_env_then_recipe(monkeypatch):
    from sdcpp_modal import _runtime_gpu

    monkeypatch.delenv("SDCPP_GPU", raising=False)
    assert _runtime_gpu("ideogram4") == PRO6000
    assert _runtime_gpu("z-image-turbo") == "L40S"
    monkeypatch.setenv("SDCPP_GPU", "L40S")
    assert _runtime_gpu("ideogram4") == "L40S"


def test_heavy_recipes_default_to_pro_6000():
    assert default_gpu_for_recipe("ideogram4") == PRO6000
    assert default_gpu_for_recipe("flux2-dev") == PRO6000
    assert default_gpu_for_recipe("z-image-turbo") == "L40S"
    assert default_gpu_for_recipe("flux2-klein") == "L40S"
    models = {item["id"]: item for item in list_models()}
    assert models["ideogram4"]["default_gpu"] == PRO6000
    assert models["flux2-dev"]["default_gpu"] == PRO6000


def test_create_job_omitted_gpu_follows_recipe(tmp_path):
    service = JobService(tmp_path)
    heavy = service.create_job(
        [{"prompt": "a paper boat", "count": 1, "seed": 1}],
        {"recipe": "flux2-dev", "dry_run": True},
    )
    light = service.create_job(
        [{"prompt": "a paper boat", "count": 1, "seed": 1}],
        {"recipe": "z-image-turbo", "dry_run": True},
    )
    forced = service.create_job(
        [{"prompt": "a paper boat", "count": 1, "seed": 1}],
        {"recipe": "ideogram4", "gpu": "L40S", "dry_run": True},
    )
    assert heavy["gpu"] == PRO6000
    assert light["gpu"] == "L40S"
    assert forced["gpu"] == "L40S"


def test_dry_run_job_writes_a_png(tmp_path):
    service = JobService(tmp_path)
    job = service.create_job(
        [{"prompt": "a rainy city at night", "count": 2, "seed": 7}],
        {"recipe": "z-image-turbo", "gpu": "L40S", "dry_run": True},
    )
    service.start(job["id"])
    deadline = time.time() + 10
    while time.time() < deadline:
        current = service.get_job(job["id"])
        if current["status"] in {"completed", "failed"}:
            break
        time.sleep(0.05)
    detail = service.get_job_detail(job["id"])
    assert detail["job"]["status"] == "completed"
    assert detail["job"]["completed_images"] == 2
    assert Decimal(detail["job"]["cost_usd"]) == 0
    assert detail["job"]["cost_events"] >= 1
    assert detail["job"]["cost_chain"][0]["phase"] == "local"
    assert detail["job"]["cost_chain"][0]["job_id"] == job["id"]
    assert all(item["status"] == "completed" for item in detail["images"])
    gallery = service.list_images(job_id=job["id"])
    assert gallery["total"] == 2
    first = tmp_path / "outputs" / job["id"] / f"{detail['images'][0]['id']}.png"
    assert first.is_file()
    assert first.stat().st_size > 100


def test_fastapi_create_job_dry_run(tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from web.api import configure
    from web.server import app

    configure(tmp_path)
    client = TestClient(app)
    meta = client.get("/api/meta").json()
    assert meta["defaults"]["recipe"] == "z-image-turbo"
    assert meta["defaults"]["gpu"] == "L40S"
    models = {item["id"]: item for item in meta["models"]}
    assert models["ideogram4"]["default_gpu"] == "RTX-PRO-6000"
    assert models["flux2-dev"]["default_gpu"] == "RTX-PRO-6000"
    gpu_ids = {item["id"] for item in meta["gpus"]}
    assert "RTX-PRO-6000" in gpu_ids
    heavy = client.post(
        "/api/jobs",
        json={"prompt": "a paper boat", "recipe": "ideogram4", "dry_run": True, "count": 1, "seed": 3},
    )
    assert heavy.status_code == 200
    assert heavy.json()["gpu"] == "RTX-PRO-6000"
    created = client.post(
        "/api/jobs",
        json={"prompt": "a paper boat", "recipe": "z-image-turbo", "dry_run": True, "count": 1, "seed": 3},
    )
    assert created.status_code == 200
    job_id = created.json()["id"]
    deadline = time.time() + 10
    while time.time() < deadline:
        detail = client.get(f"/api/jobs/{job_id}").json()
        if detail["job"]["status"] == "completed":
            break
        time.sleep(0.05)
    assert detail["job"]["completed_images"] == 1
    image_id = detail["images"][0]["id"]
    png = client.get(f"/api/images/{image_id}/file")
    assert png.status_code == 200
    assert png.headers["content-type"].startswith("image/")
    gallery = client.get("/api/gallery").json()
    assert gallery["total"] >= 1
    home = client.get("/")
    assert home.status_code == 200
    assert "sdcpp-modal" in home.text
    assert 'lang="zh-CN"' in home.text
    assert "七方工作台" in home.text
    assert "成本" in home.text
    assert meta["defaults"]["cost_log"]
    cost = client.get("/api/cost").json()
    assert "traces" in cost
    assert "billed" in cost
    assert "cards" in cost["rates"]
    assert any(card["id"] == "RTX-PRO-6000" for card in cost["rates"]["cards"])
    filtered = client.get(f"/api/cost?job_id={job_id}").json()
    assert filtered["job_id"] == job_id
    assert filtered["event_count"] >= 1
    assert Decimal(filtered["billed"]["usd"]) == 0
    listed = client.get("/api/jobs").json()
    match = next(item for item in listed if item["id"] == job_id)
    assert Decimal(match["cost_usd"]) == 0
    assert match["cost_events"] >= 1
