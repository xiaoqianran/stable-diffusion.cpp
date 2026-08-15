# sdcpp-hooks

Decoupled image-generation hooks for [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).

This package treats `sd-cli` as a black box. It never imports C++ headers, never
binds the public C API, and never copies server JSON field names into the
stable contract. Upstream can keep moving; this side keeps generating.

## Why this survives upstream updates

1. **Stable contract.** Callers only send `GenerateRequest` (`prompt`, size,
   seed, model URIs, plus an escape hatch).
2. **Runtime discovery.** Each container start probes `sd-cli -h` and records
   the flags that actually exist in that binary.
3. **Alias table + drop, don't crash.** Common renames (`--steps` /
   `--sample-steps`, `--cfg-scale` / `--txt-cfg`) are mapped automatically.
   Unknown extra flags are dropped and reported instead of failing the run.
4. **Pinned engine image.** Modal pulls
   `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` by default. Override with
   `SDCPP_IMAGE` when you want a digest or your own build.
5. **Model URIs, not repo paths.** Weights come from `hf://`, `civitai://`,
   `https://`, or a local/volume path. Download and cache live on a Modal
   Volume, outside the C++ tree.

```text
caller  ->  GenerateRequest  ->  use_models()  ->  use_engine()  ->  adapt  ->  sd-cli
                (stable)         hf/civitai         --help probe     drop
```

## Hooks

```python
from sdcpp_hooks import GenerateRequest, use_sdcpp

sd = use_sdcpp(probe=..., binary="/sd-cli", cache_dir=..., run=..., output_path=...)
result = sd(GenerateRequest(prompt="a lovely cat", model="hf://org/repo/model.safetensors"))
```

Or compose the pieces:

- `use_engine(probe=...)` — parse `--help`
- `use_models(request, cache_dir=...)` — resolve URIs onto disk
- `generate(request, engine=..., models=..., run=...)` — adapt + execute

## Model URIs

| URI | Meaning |
| --- | --- |
| `hf://org/repo/file.safetensors` | Hugging Face file at revision `main` |
| `hf://org/repo@rev/file.gguf` | Hugging Face file at `rev` |
| `civitai://128713` | Civitai **model version** id |
| `https://...` | Direct download |
| `/models/foo.safetensors` | Already on the volume or local disk |

Set `HF_TOKEN` and `CIVITAI_TOKEN` in the environment (or the Modal secret
`sdcpp-tokens`). Do not put tokens in git. If a token was pasted into chat,
rotate it.

`HF_ENDPOINT` can point at a Hugging Face mirror.

## Local tests

The hook layer is stdlib-only. From this directory:

```bash
python3 -m pip install pytest
python3 -m pytest
```

These tests do not download weights and do not need a GPU.

## Modal

```bash
python3 -m pip install 'modal>=0.64'
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
modal secret create sdcpp-tokens HF_TOKEN="$HF_TOKEN" CIVITAI_TOKEN="$CIVITAI_TOKEN"
```

Probe the remote `sd-cli` flags without downloading a checkpoint:

```bash
modal run app.py --probe-only
```

Generate with the default SD 1.5 recipe (first run downloads the checkpoint
onto volume `sdcpp-models`):

```bash
modal run app.py --prompt "a lovely cat" --recipe sd15 --output /tmp/cat.png
```

Optional environment:

| Variable | Default |
| --- | --- |
| `SDCPP_IMAGE` | `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` |
| `SDCPP_GPU` | `L4` |
| `HF_ENDPOINT` | `https://huggingface.co` |

`POST` the same `GenerateRequest` JSON to the `api_generate` endpoint after
`modal deploy app.py`.

## What this package will not do

- Track every new sampler, DiT family, or server route in `examples/server`
- Rebuild `stable-diffusion.cpp` from this repository on each request
- Store tokens, weights, or generated images in git
