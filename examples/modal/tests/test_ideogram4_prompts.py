import json
from pathlib import Path


PROMPTS = Path(__file__).resolve().parent.parent / "gallery" / "ideogram4_prompts.json"


def test_ideogram4_gallery_has_five_structured_prompts():
    rows = json.loads(PROMPTS.read_text(encoding="utf-8"))

    assert len(rows) == 5
    ids = [row["id"] for row in rows]
    assert len(set(ids)) == 5
    for row in rows:
        prompt = row["prompt"]
        assert prompt["high_level_description"]
        assert prompt["style_description"]["aesthetics"]
        assert prompt["compositional_deconstruction"]["elements"]
        assert isinstance(row["seed"], int)
