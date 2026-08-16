import pytest

from sdcpp_hooks.cli import parse_argv


def test_parse_pull_requires_uris_or_recipe():
    with pytest.raises(SystemExit):
        parse_argv(["pull"])


def test_parse_pull_can_use_a_recipe():
    command = parse_argv(["pull", "--recipe", "sd15"])

    assert command.action == "pull"
    assert command.uris[0].startswith("hf://")


def test_parse_pull_collects_one_or_more_uris():
    command = parse_argv(
        [
            "pull",
            "hf://org/repo/model.safetensors",
            "civitai://128713",
        ]
    )

    assert command.action == "pull"
    assert command.uris == [
        "hf://org/repo/model.safetensors",
        "civitai://128713",
    ]


def test_parse_ls_and_probe_have_no_uris():
    assert parse_argv(["ls"]).action == "ls"
    assert parse_argv(["probe"]).action == "probe"
    assert parse_argv(["cost"]).action == "cost"


def test_parse_generate_requires_a_prompt():
    with pytest.raises(SystemExit):
        parse_argv(["generate", "--model", "hf://org/repo/model.safetensors"])


def test_parse_generate_maps_cli_flags_to_a_request_payload():
    command = parse_argv(
        [
            "generate",
            "--prompt",
            "a lovely cat",
            "--model",
            "hf://org/repo/model.safetensors",
            "--width",
            "768",
            "--steps",
            "12",
            "--output",
            "cat.png",
        ]
    )

    assert command.action == "generate"
    assert command.output == "cat.png"
    payload = command.to_payload()
    assert payload["prompt"] == "a lovely cat"
    assert payload["model"] == "hf://org/repo/model.safetensors"
    assert payload["width"] == 768
    assert payload["steps"] == 12


def test_parse_generate_can_use_a_recipe_without_an_explicit_model():
    command = parse_argv(["generate", "-p", "a cat", "--recipe", "sd15"])

    assert command.to_payload()["model"].startswith("hf://")
    assert command.to_payload()["width"] == 512


def test_parse_generate_accepts_ideogram4_recipe():
    command = parse_argv(["generate", "-p", '{"high_level_description":"a cat"}', "--recipe", "ideogram4"])
    payload = command.to_payload()

    assert payload["model"] is None
    assert "leejet/ideogram-4-GGUF/ideogram4-Q4_0.gguf" in payload["diffusion_model"]
    assert payload["uncond_diffusion_model"].endswith("ideogram4_uncond-Q4_0.gguf")
    assert payload["llm"].endswith("Qwen3-VL-8B-Instruct-Q4_K_M.gguf")
    assert payload["vae"].endswith("ae.safetensors")
    assert payload["width"] == 1024
    assert payload["extra_cli"]["--diffusion-fa"] is True
    assert payload["extra_cli"]["--offload-to-cpu"] is True


def test_parse_convert_ideogram4_defaults_to_q8():
    command = parse_argv(["convert", "--recipe", "ideogram4"])

    assert command.action == "convert"
    assert command.recipe == "ideogram4"
    assert command.quant == "q8_0"


def test_ideogram4_convert_jobs_keep_cond_and_uncond_apart(tmp_path):
    from sdcpp_hooks.recipes import convert_jobs

    jobs = convert_jobs("ideogram4", tmp_path, quant="q8_0")
    assert len(jobs) == 2
    assert jobs[0]["gguf"].endswith("ideogram4-Q8_0.gguf")
    assert jobs[1]["gguf"].endswith("ideogram4_uncond-Q8_0.gguf")
    assert jobs[0]["src"] != jobs[1]["src"]


def test_parse_pull_recipe_includes_ideogram4_uncond():
    command = parse_argv(["pull", "--recipe", "ideogram4"])

    assert any("ideogram4_uncond-Q4_0.gguf" in uri for uri in command.uris)
    assert any("Qwen3-VL-8B-Instruct-Q4_K_M.gguf" in uri for uri in command.uris)
    assert len(command.uris) == 4


def test_parse_generate_accepts_the_five_new_recipes():
    recipes = {
        "sd2": "Manojb/stable-diffusion-2-1-base",
        "sd-turbo": "stabilityai/sd-turbo",
        "sdxl-turbo": "stabilityai/sdxl-turbo",
        "ssd-1b": "segmind/SSD-1B",
        "dreamlike-photoreal": "dreamlike-art/dreamlike-photoreal-2.0",
    }
    for name, marker in recipes.items():
        command = parse_argv(["generate", "-p", "a test image", "--recipe", name])
        payload = command.to_payload()
        assert marker in payload["model"]
        assert payload["width"] == 512
        assert payload["steps"]


def test_recipe_extra_cli_merges_with_user_flags():
    from sdcpp_hooks.recipes import RECIPES, apply_recipe

    RECIPES["sd2"]["extra_cli"] = {"--prediction": "v"}
    try:
        request = apply_recipe("sd2", prompt="a fox", extra_cli={"--type": "f16"})
        assert request.extra_cli["--prediction"] == "v"
        assert request.extra_cli["--type"] == "f16"
    finally:
        RECIPES["sd2"].pop("extra_cli", None)


def test_parse_generate_forwards_sd_cli_artifact_and_extra_flags():
    command = parse_argv(
        [
            "generate",
            "-p",
            "a cat",
            "--vae",
            "hf://org/repo/vae.safetensors",
            "--uncond-diffusion-model",
            "hf://org/repo/uncond.safetensors",
            "--control-net",
            "hf://org/repo/control.safetensors",
            "--offload-to-cpu",
            "--type",
            "f16",
        ]
    )

    payload = command.to_payload()
    assert payload["vae"] == "hf://org/repo/vae.safetensors"
    assert payload["uncond_diffusion_model"] == "hf://org/repo/uncond.safetensors"
    assert payload["control_net"] == "hf://org/repo/control.safetensors"
    assert payload["extra_cli"]["--offload-to-cpu"] is True
    assert payload["extra_cli"]["--type"] == "f16"


def test_parse_publish_and_generate_publish_flag():
    publish = parse_argv(["publish", "cat.png", "--recipe", "flux", "-p", "a cat"])
    assert publish.action == "publish"
    assert publish.image == "cat.png"
    assert publish.recipe == "flux"
    assert publish.prompt == "a cat"

    generate = parse_argv(["generate", "-p", "a cat", "--recipe", "sd15", "--publish"])
    assert generate.publish is True
    assert generate.model_id == ""


def test_parse_put_collects_local_files():
    command = parse_argv(["put", "cat.png", "mask.png"])

    assert command.action == "put"
    assert command.files == ["cat.png", "mask.png"]
