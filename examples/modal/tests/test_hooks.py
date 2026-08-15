from pathlib import Path

from sdcpp_hooks.contract import GenerateRequest
from sdcpp_hooks.hooks import generate, use_engine, use_models, use_sdcpp


def test_use_engine_probes_help_text_once():
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return "Usage:\n  -p, --prompt <string>  the prompt\n  -m, --model <string>  model\n"

    engine = use_engine(probe=probe, binary="/sd-cli")

    assert calls["n"] == 1
    assert engine.has_flag("--prompt")
    assert use_engine(probe=probe, binary="/sd-cli").has_flag("--model")
    assert calls["n"] == 2


def test_use_models_only_resolves_artifact_fields(tmp_path):
    request = GenerateRequest(
        prompt="a cat",
        model="hf://org/repo/model.safetensors",
        width=512,
    )

    def fetch(url, dest, headers):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"weights")
        return dest

    models = use_models(request, cache_dir=tmp_path, fetch=fetch)

    assert set(models) == {"model"}
    assert models["model"].read_bytes() == b"weights"


def test_generate_hook_runs_adapted_argv_and_collects_images(tmp_path, current_help_text):
    output_png = tmp_path / "out.png"
    seen = {}

    def run(argv, workdir):
        seen["argv"] = argv
        seen["workdir"] = workdir
        output_png.write_bytes(b"fake-png")
        return [output_png]

    request = GenerateRequest(
        prompt="a lovely cat",
        model=str(tmp_path / "local.safetensors"),
        steps=12,
        seed=3,
    )
    (tmp_path / "local.safetensors").write_bytes(b"weights")

    result = generate(
        request,
        engine=use_engine(help_text=current_help_text, binary="/sd-cli"),
        models=use_models(request, cache_dir=tmp_path / "cache"),
        run=run,
        output_path=output_png,
    )

    assert result.images == [b"fake-png"]
    assert result.argv[result.argv.index("--prompt") + 1] == "a lovely cat"
    assert result.argv[result.argv.index("--steps") + 1] == "12"
    assert result.dropped_fields == []
    assert seen["argv"][0] == "/sd-cli"


def test_use_sdcpp_composes_the_three_hooks(tmp_path, current_help_text):
    local_model = tmp_path / "sd.safetensors"
    local_model.write_bytes(b"weights")
    written = tmp_path / "output.png"

    def run(argv, workdir):
        dest = Path(argv[argv.index("--output") + 1])
        dest.write_bytes(b"png")
        return [dest]

    sd = use_sdcpp(
        probe=lambda: current_help_text,
        binary="/sd-cli",
        cache_dir=tmp_path / "cache",
        run=run,
        output_path=written,
    )
    result = sd(
        GenerateRequest(
            prompt="red apple",
            model=str(local_model),
            extra_cli={"--verbose": True},
        )
    )

    assert result.images == [b"png"]
    assert "--verbose" in result.argv
