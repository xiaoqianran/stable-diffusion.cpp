from sdcpp_hooks.gallery import (
    ImageRecord,
    build_site,
    image_id,
    paginate,
    record_paths,
    scan_records,
    slugify_model,
    write_sidecar,
)


def test_slugify_model_keeps_known_and_future_ids():
    assert slugify_model("sd15") == "sd15"
    assert slugify_model("Qwen Image Edit") == "qwen-image-edit"
    assert slugify_model("My_New.Model!!") == "my-new-model"


def test_record_paths_are_per_model():
    png, sidecar = record_paths("flux2", "abc")
    assert png == "images/flux2/abc.png"
    assert sidecar == "images/flux2/abc.json"


def test_scan_and_paginate_records(tmp_path):
    first = ImageRecord(
        id="a",
        model="sd15",
        path="images/sd15/a.png",
        prompt="cat",
        created_at="2026-08-15T16:00:00+00:00",
    )
    second = ImageRecord(
        id="b",
        model="flux",
        path="images/flux/b.png",
        prompt="dog",
        created_at="2026-08-15T17:00:00+00:00",
    )
    write_sidecar(tmp_path, first, b"png-a")
    write_sidecar(tmp_path, second, b"png-b")

    records = scan_records(tmp_path)
    assert [item.id for item in records] == ["b", "a"]
    pages = paginate(records, per_page=1)
    assert len(pages) == 2
    assert pages[0][0].model == "flux"


def test_build_site_writes_model_pages_and_pagination(tmp_path):
    for index in range(3):
        record = ImageRecord(
            id=f"img-{index}",
            model="sd15",
            path=f"images/sd15/img-{index}.png",
            prompt=f"prompt {index}",
            seed=index,
            created_at=f"2026-08-15T18:0{index}:00+00:00",
        )
        write_sidecar(tmp_path, record, b"png")

    out = tmp_path / "site"
    stats = build_site(tmp_path, out, per_page=2)
    assert stats["images"] == 3
    assert (out / "index.html").exists()
    assert (out / "page" / "2.html").exists()
    assert (out / "model" / "sd15" / "index.html").exists()
    assert (out / "model" / "sd15" / "page" / "2.html").exists()
    assert (out / "model" / "flux" / "index.html").exists()
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "prompt 2" in html
    assert "Page 1 / 2" in html
    assert image_id("sd15", 1, b"png").endswith("-sd15-1-" + __import__("hashlib").sha256(b"png").hexdigest()[:8])
