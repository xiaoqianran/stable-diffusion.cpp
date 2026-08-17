from pathlib import Path


STATIC = Path(__file__).parents[1] / "web" / "static"


def test_index_uses_user_facing_information_architecture():
    html = (STATIC / "index.html").read_text(encoding="utf-8")

    assert '#/create' in html
    assert '#/runs' in html
    assert '#/gallery' in html
    assert '#/generate' not in html
    assert '#/batch' not in html
    assert '#/cost' not in html
    assert '#/settings' not in html
    assert '/static/ux.css' in html
    assert '/static/ux-main.js' in html


def test_redesigned_web_modules_exist():
    for name in (
        "ux-core.js",
        "ux-create.js",
        "ux-runs.js",
        "ux-gallery.js",
        "ux-system.js",
        "ux-main.js",
        "ux.css",
    ):
        assert (STATIC / name).is_file(), name


def test_legacy_web_assets_are_removed():
    for name in ("app.js", "batch_controls.js", "queue.js", "styles.css", "queue.css"):
        assert not (STATIC / name).exists(), name


def test_runs_ui_labels_local_cost_as_estimate():
    text = (STATIC / "ux-runs.js").read_text(encoding="utf-8")
    assert "估算成本" in text
    assert "当前成本" not in text
    assert "new EventSource" in text
