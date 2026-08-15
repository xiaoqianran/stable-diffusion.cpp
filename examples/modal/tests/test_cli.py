import pytest

from sdcpp_hooks.cli import parse_argv


def test_parse_pull_requires_uris():
    with pytest.raises(SystemExit):
        parse_argv(["pull"])


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
