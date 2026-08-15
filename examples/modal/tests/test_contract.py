import pytest

from sdcpp_hooks.contract import GenerateRequest, GenerateResult, ValidationError


def test_generate_request_requires_a_prompt():
    with pytest.raises(ValidationError, match="prompt"):
        GenerateRequest(prompt="   ", model="/tmp/model.safetensors").validate()


def test_generate_request_requires_a_model_or_diffusion_model():
    with pytest.raises(ValidationError, match="model"):
        GenerateRequest(prompt="a cat").validate()


def test_generate_request_accepts_diffusion_model_without_full_model():
    request = GenerateRequest(
        prompt="a cat",
        diffusion_model="hf://org/repo/unet.safetensors",
        vae="hf://org/repo/vae.safetensors",
    )

    request.validate()


def test_generate_request_round_trips_through_dict():
    request = GenerateRequest(
        prompt="a lovely cat",
        negative_prompt="blurry",
        width=768,
        height=512,
        steps=16,
        cfg_scale=5.5,
        seed=7,
        model="hf://stable-diffusion-v1-5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors",
        extra_cli={"--offload-to-cpu": True},
    )

    restored = GenerateRequest.from_dict(request.to_dict())

    assert restored.prompt == "a lovely cat"
    assert restored.width == 768
    assert restored.cfg_scale == 5.5
    assert restored.extra_cli["--offload-to-cpu"] is True


def test_generate_result_keeps_image_bytes_and_dropped_fields():
    result = GenerateResult(
        images=[b"png-bytes"],
        argv=["/sd-cli", "--prompt", "a cat"],
        dropped_fields=["steps"],
        engine_id="sd-cli",
        seed=42,
    )

    assert result.images == [b"png-bytes"]
    assert result.dropped_fields == ["steps"]
    assert result.engine_id == "sd-cli"
