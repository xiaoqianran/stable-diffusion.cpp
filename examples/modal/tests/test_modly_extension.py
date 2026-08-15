from pathlib import Path

from modly_extension.generator import SDCppGenerator, build_payload


def test_build_payload_accepts_text_alias_and_recipe_defaults():
    payload = build_payload({"text": "a red fox", "recipe": "sd15", "seed": 9})

    assert payload["prompt"] == "a red fox"
    assert payload["seed"] == 9
    assert payload["model"].startswith("hf://")
    assert payload["width"] == 512


def test_generator_writes_remote_png_without_importing_sdcpp_binaries(tmp_path):
    class FakeMethod:
        def remote(self, payload):
            assert payload["prompt"] == "a cat"
            return {"images": ["aW1hZ2U="], "dropped_fields": []}

    class FakeInstance:
        generate = FakeMethod()

    class FakeCls:
        def __call__(self):
            return FakeInstance()

    gen = SDCppGenerator(tmp_path / "models", tmp_path / "out")
    gen._model = FakeCls()
    path = gen.generate(b"", {"prompt": "a cat", "recipe": "sd15"})

    assert path.suffix == ".png"
    assert path.read_bytes() == b"image"
    assert Path(path).parent == tmp_path / "out"
