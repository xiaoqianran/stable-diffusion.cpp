import time

import pytest

from sdcpp_hooks.web_catalog import default_recipe, list_models, normalize_gpu
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
    assert normalize_gpu("rtx-pro-6000") == "RTX6000"


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
