from sdcpp_hooks.adapt import adapt_request
from sdcpp_hooks.contract import GenerateRequest
from sdcpp_hooks.discover import discover_engine


def test_adapt_request_maps_stable_fields_to_current_cli_flags(current_help_text):
    engine = discover_engine(current_help_text, binary="/sd-cli")
    request = GenerateRequest(
        prompt="a lovely cat",
        negative_prompt="blurry",
        width=768,
        height=512,
        steps=16,
        cfg_scale=4.5,
        seed=11,
        model="/models/sd15.safetensors",
        extra_cli={"--offload-to-cpu": True, "--unknown-flag": "nope"},
    )

    planned = adapt_request(request, engine, output_path="/tmp/out.png")

    assert planned.argv[0] == "/sd-cli"
    assert planned.argv[planned.argv.index("--prompt") + 1] == "a lovely cat"
    assert planned.argv[planned.argv.index("--width") + 1] == "768"
    assert planned.argv[planned.argv.index("--cfg-scale") + 1] == "4.5"
    assert planned.argv[planned.argv.index("--output") + 1] == "/tmp/out.png"
    assert "--offload-to-cpu" in planned.argv
    assert "--unknown-flag" not in planned.argv
    assert "extra_cli.--unknown-flag" in planned.dropped_fields
    assert "steps" not in planned.dropped_fields


def test_adapt_request_uses_aliases_when_upstream_renames_common_flags(renamed_help_text):
    engine = discover_engine(renamed_help_text, binary="/sd-cli")
    request = GenerateRequest(
        prompt="a cat",
        model="/models/sd15.safetensors",
        steps=20,
        cfg_scale=7.0,
        sampling_method="euler",
        extra_cli={"--new-turbo-flag": True},
    )

    planned = adapt_request(request, engine, output_path="/tmp/out.png")

    assert "--steps" not in planned.argv
    assert "--cfg-scale" not in planned.argv
    assert planned.argv[planned.argv.index("--sample-steps") + 1] == "20"
    assert planned.argv[planned.argv.index("--txt-cfg") + 1] == "7.0"
    assert "steps" not in planned.dropped_fields
    assert "cfg_scale" not in planned.dropped_fields
    assert "sampling_method" in planned.dropped_fields
    assert "--new-turbo-flag" in planned.argv


def test_adapt_request_does_not_emit_false_boolean_flags(current_help_text):
    engine = discover_engine(current_help_text, binary="/sd-cli")
    request = GenerateRequest(
        prompt="a cat",
        model="/models/sd15.safetensors",
        extra_cli={"--verbose": False, "--offload-to-cpu": True},
    )

    planned = adapt_request(request, engine, output_path="/tmp/out.png")

    assert "--verbose" not in planned.argv
    assert "--offload-to-cpu" in planned.argv
