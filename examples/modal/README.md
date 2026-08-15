# Modal CLI

Run the official `sd-cli` on [Modal](https://modal.com) when the local machine cannot host the model.

This example does not change the C API, local CLI, or server. It treats `sd-cli` as a black box: each GPU container probes `--help` and only forwards flags that exist on that binary.

Weights live on Modal Volume `sdcpp-models`. The default image is `ghcr.io/leejet/stable-diffusion.cpp:master-cuda`.

## Setup

```bash
cd examples/modal
python3 -m pip install 'modal>=0.64' pytest
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
```

Public Hugging Face files work without a token. For gated models or Civitai:

```bash
export HF_TOKEN=...
export CIVITAI_TOKEN=...
# optional persistent secret
modal secret create sdcpp-tokens HF_TOKEN="$HF_TOKEN" CIVITAI_TOKEN="$CIVITAI_TOKEN"
```

Do not put tokens in git.

## CLI

```bash
python3 sdcpp_modal.py pull --recipe sd15
python3 sdcpp_modal.py ls
python3 sdcpp_modal.py probe
python3 sdcpp_modal.py generate -p "a lovely cat" --recipe sd15 -o cat.png
```

| Command | Where it runs | What it does |
| --- | --- | --- |
| `pull` | CPU | download URIs (or a `--recipe`) onto volume `sdcpp-models` |
| `put` | CPU | upload a small local file (init image, mask) to `uploads/` |
| `ls` | CPU | list files already on that volume |
| `probe` | CUDA image, no GPU | print remote `sd-cli` flags |
| `generate` | GPU (`SDCPP_GPU`, default `L4`) | run `sd-cli` and write a local PNG |

`generate` downloads a missing URI onto the same volume before it runs `sd-cli`.

Common `sd-cli` flags are first-class (`--vae`, `--diffusion-model`, `--init-img`, `--control-net`, `--taesd`, `--sampling-method`, ...). Any other remote flag can be appended and is forwarded if `probe` sees it:

```bash
python3 sdcpp_modal.py generate -p "a cat" --recipe sd15 --offload-to-cpu --type f16
```

Unknown flags are dropped and printed as `dropped_fields` instead of failing the run.

### Model URIs

| URI | Meaning |
| --- | --- |
| `hf://org/repo/file.safetensors` | Hugging Face file at revision `main` |
| `hf://org/repo@rev/file.gguf` | Hugging Face file at `rev` |
| `civitai://128713` | Civitai **model version** id |
| `https://...` | Direct download |

`HF_ENDPOINT` can point at a Hugging Face mirror.

For a multi-gigabyte local checkpoint, use `modal volume put` instead of `put`:

```bash
modal volume put sdcpp-models ./v1-5-pruned-emaonly.safetensors hf/local/v1-5-pruned-emaonly.safetensors
```

Then pass `--model /models/hf/local/v1-5-pruned-emaonly.safetensors`.

### Environment

| Variable | Default |
| --- | --- |
| `SDCPP_IMAGE` | `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` |
| `SDCPP_GPU` | `L4` |
| `SDCPP_SECRET` | `sdcpp-tokens` (used only if that Modal secret exists) |
| `HF_ENDPOINT` | `https://huggingface.co` |

## Limits

- First-class path is `img_gen`. `vid_gen`, `adetailer`, `convert`, and `metadata` are not wrapped; some of their flags can still be forwarded if the remote binary accepts them.
- `put` is for small files (64 MiB). Weights should use `pull` or `modal volume put`.
- Size defaults match local `sd-cli` (512x512, 20 steps, cfg 7.0) unless you override them or use a recipe.

## Tests

```bash
python3 -m pytest
```

These tests do not download weights and do not need a GPU.
