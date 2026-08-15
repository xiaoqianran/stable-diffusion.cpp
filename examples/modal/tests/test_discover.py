from sdcpp_hooks.discover import discover_engine


def test_discover_engine_parses_short_and_long_flags(current_help_text):
    engine = discover_engine(current_help_text, binary="/sd-cli")

    assert engine.binary == "/sd-cli"
    assert engine.has_flag("--prompt")
    assert engine.has_flag("-p")
    assert engine.flag("--model").short == "-m"
    assert engine.flag("--cfg-scale").value_hint == "float"
    assert engine.flag("--verbose").value_hint is None
    assert engine.flag("--seed").name == "--seed"


def test_discover_engine_ignores_wrapped_description_lines(current_help_text):
    engine = discover_engine(current_help_text)

    assert not engine.has_flag("diffusion")
    assert engine.has_flag("--diffusion-model")


def test_discover_engine_sees_future_renames_without_code_changes(renamed_help_text):
    engine = discover_engine(renamed_help_text)

    assert engine.has_flag("--sample-steps")
    assert not engine.has_flag("--steps")
    assert engine.has_flag("--txt-cfg")
    assert engine.has_flag("--new-turbo-flag")
