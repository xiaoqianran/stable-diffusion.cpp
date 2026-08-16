from pathlib import Path


README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_has_cost_tutorial():
    text = README.read_text(encoding="utf-8")
    assert "## 成本教程" in text
    assert "http://127.0.0.1:7863" in text
    assert "python3 sdcpp_modal.py web --dry-run" in text
    assert "python3 sdcpp_modal.py cost" in text
    assert "GET /api/cost" in text
    assert "session:storage" in text
    assert "remote:ensure_artifacts" in text
    assert "remote:generate" in text
    assert "$0.000545531/s" in text
    assert "$0.000845531/s" in text
    assert "job_…" in text or "job_" in text
    assert "SDCPP_COST_LOG" in text
    assert "不能把 session 和 remote 加在一起" in text
