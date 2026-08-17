from pathlib import Path

from sdcpp_hooks.contract import GenerateRequest
from sdcpp_hooks import server_runtime


SERVER_HELP = """
stable-diffusion.cpp server test
Usage: /sd-server [options]
  -m, --model <string>                          path to full model
  --diffusion-model <string>                    standalone diffusion model
  --uncond-diffusion-model <string>             unconditional diffusion model
  --vae <string>                                vae
  --llm <string>                                llm
  --diffusion-fa                                flash attention
  --offload-to-cpu                              offload
  --listen-ip <string>                          listen ip
  --listen-port <int>                           listen port
"""


def _fake_models(request, **_kwargs):
    resolved = {}
    for field in server_runtime.SERVER_MODEL_FIELDS:
        if getattr(request, field, None):
            resolved[field] = Path(f"/models/{field}.bin")
    return resolved


def test_recipe_server_argv_contains_models_not_per_image_flags(monkeypatch, tmp_path):
    monkeypatch.setattr(server_runtime, "use_models", _fake_models)
    monkeypatch.setattr(
        server_runtime,
        "probe_server_help",
        lambda: (SERVER_HELP, "/sd-server"),
    )

    argv, dropped = server_runtime.recipe_server_argv("z-image-turbo", tmp_path)

    assert argv[0] == "/sd-server"
    assert "--diffusion-model" in argv
    assert "--vae" in argv
    assert "--llm" in argv
    assert "--diffusion-fa" in argv
    assert "--offload-to-cpu" in argv
    assert "--listen-port" in argv
    assert "--prompt" not in argv
    assert "--seed" not in argv
    assert dropped == []


def test_server_generate_maps_request_to_sdapi(monkeypatch):
    captured = {}

    def fake_request(method, url, payload=None, *, timeout):
        captured.update({"method": method, "url": url, "payload": payload, "timeout": timeout})
        return {"images": ["aGVsbG8="]}

    monkeypatch.setattr(server_runtime, "_request_json", fake_request)
    request = GenerateRequest(
        prompt="a cat",
        model="/models/sd15.bin",
        width=512,
        height=512,
        steps=20,
        cfg_scale=7.0,
        seed=42,
        sampling_method="euler",
        scheduler="discrete",
    )

    result = server_runtime.server_generate(request)

    assert captured["method"] == "POST"
    assert captured["url"].endswith("/sdapi/v1/txt2img")
    assert captured["payload"]["prompt"] == "a cat"
    assert captured["payload"]["batch_size"] == 1
    assert captured["payload"]["sampler_name"] == "euler"
    assert captured["payload"]["scheduler"] == "discrete"
    assert result["images"] == [b"hello"]
