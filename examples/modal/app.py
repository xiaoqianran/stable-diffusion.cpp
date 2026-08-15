"""Modal entrypoints. This file only wires hooks to GPU + volumes.

The generation contract lives in sdcpp_hooks and does not import Modal or
stable-diffusion.cpp internals.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

import modal

from sdcpp_hooks.contract import GenerateRequest
from sdcpp_hooks.hooks import generate, use_engine, use_models
from sdcpp_hooks.probe import probe_cli_help
from sdcpp_hooks.recipes import apply_recipe
from sdcpp_hooks.runner import run_cli


HERE = Path(__file__).resolve().parent
IMAGE_TAG = os.environ.get(
    "SDCPP_IMAGE",
    "ghcr.io/leejet/stable-diffusion.cpp:master-cuda",
)
GPU = os.environ.get("SDCPP_GPU", "L4")
MODEL_ROOT = Path(os.environ.get("SDCPP_MODEL_ROOT", "/models"))

volume = modal.Volume.from_name("sdcpp-models", create_if_missing=True)


def _image() -> modal.Image:
    image = (
        modal.Image.from_registry(IMAGE_TAG, add_python="3.12")
        .entrypoint([])
        .pip_install("fastapi[standard]>=0.115")
    )
    if hasattr(image, "add_local_python_source"):
        return image.add_local_python_source("sdcpp_hooks")
    return image.add_local_dir(str(HERE / "sdcpp_hooks"), remote_path="/pkg/sdcpp_hooks").env(
        {"PYTHONPATH": "/pkg"}
    )


def _secrets() -> list[modal.Secret]:
    local = {
        key: os.environ[key]
        for key in ("HF_TOKEN", "CIVITAI_TOKEN")
        if os.environ.get(key)
    }
    if local:
        return [modal.Secret.from_dict(local)]
    return [modal.Secret.from_name("sdcpp-tokens")]


app = modal.App("sdcpp-hooks", image=_image())


@app.cls(
    gpu=GPU,
    timeout=60 * 60,
    scaledown_window=120,
    volumes={str(MODEL_ROOT): volume},
    secrets=_secrets(),
)
class SDEngine:
    @modal.enter()
    def start(self) -> None:
        help_text, binary = probe_cli_help()
        self.engine = use_engine(help_text=help_text, binary=binary)
        self.binary = binary

    @modal.method()
    def capabilities(self) -> dict:
        return {
            "binary": self.binary,
            "flags": sorted(
                name for name in self.engine.flags if name.startswith("--")
            ),
        }

    def _generate_payload(self, payload: dict) -> dict:
        request = GenerateRequest.from_dict(payload)
        request.validate()
        models = use_models(request, cache_dir=MODEL_ROOT)
        volume.commit()
        output_path = Path("/tmp/sdcpp-output.png")
        result = generate(
            request,
            engine=self.engine,
            models=models,
            run=run_cli,
            output_path=output_path,
        )
        return {
            "images": [base64.b64encode(image).decode("ascii") for image in result.images],
            "argv": result.argv,
            "dropped_fields": result.dropped_fields,
            "engine_id": result.engine_id,
            "seed": result.seed,
        }

    @modal.method()
    def generate(self, payload: dict) -> dict:
        return self._generate_payload(payload)

    @modal.fastapi_endpoint(method="POST")
    def api_generate(self, payload: dict) -> dict:
        return self._generate_payload(payload)


@app.local_entrypoint()
def main(
    prompt: str = "",
    recipe: str = "sd15",
    model: str = "",
    output: str = "output.png",
    width: int = 0,
    height: int = 0,
    steps: int = 0,
    seed: int = 42,
    probe_only: bool = False,
) -> None:
    if probe_only:
        caps = SDEngine().capabilities.remote()
        print(caps["binary"])
        print(" ".join(caps["flags"]))
        return
    if not prompt:
        raise SystemExit("--prompt is required unless --probe-only is set")
    request = apply_recipe(
        recipe,
        prompt=prompt,
        model=model or None,
        width=width or None,
        height=height or None,
        steps=steps or None,
        seed=seed,
    )
    result = SDEngine().generate.remote(request.to_dict())
    if not result["images"]:
        raise SystemExit(f"no images returned; dropped={result['dropped_fields']}")
    dest = Path(output)
    dest.write_bytes(base64.b64decode(result["images"][0]))
    print(f"wrote {dest} via {result['engine_id']}")
    if result["dropped_fields"]:
        print("dropped_fields:", ", ".join(result["dropped_fields"]))
