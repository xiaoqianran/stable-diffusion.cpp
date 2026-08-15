# sdcpp Modal CLI

Standalone CLI for this repo. It is not a plugin and is not wired into any other app.

It does two things:

1. **`pull`** — download checkpoints onto Modal Volume `sdcpp-models`
2. **`generate`** — run remote `sd-cli` on a GPU and write a PNG locally

`sd-cli` is treated as a black box. Each GPU container probes `--help` and only uses flags that exist on that binary.

## Setup

```bash
cd examples/modal
python3 -m pip install 'modal>=0.64' pytest
modal token set --token-id "$MODAL_TOKEN_ID" --token-secret "$MODAL_TOKEN_SECRET"
modal secret create sdcpp-tokens HF_TOKEN="$HF_TOKEN" CIVITAI_TOKEN="$CIVITAI_TOKEN"
```

Do not put tokens in git. Use env vars or the `sdcpp-tokens` secret.

## CLI

```bash
python3 sdcpp_modal.py pull hf://stable-diffusion-v1-5/stable-diffusion-v1-5/v1-5-pruned-emaonly.safetensors
python3 sdcpp_modal.py ls
python3 sdcpp_modal.py probe
python3 sdcpp_modal.py generate -p "a lovely cat" --recipe sd15 -o cat.png
```

| Command | Where it runs | What it does |
| --- | --- | --- |
| `pull` | CPU | download URIs onto volume `sdcpp-models` |
| `ls` | CPU | list files already on that volume |
| `probe` | CUDA image, no GPU | print remote `sd-cli` flags |
| `generate` | GPU (`SDCPP_GPU`, default `L4`) | run txt2img and write a local PNG |

`generate` will download a missing URI onto the same volume before it runs `sd-cli`.

### Model URIs

| URI | Meaning |
| --- | --- |
| `hf://org/repo/file.safetensors` | Hugging Face file at revision `main` |
| `hf://org/repo@rev/file.gguf` | Hugging Face file at `rev` |
| `civitai://128713` | Civitai **model version** id |
| `https://...` | Direct download |

`HF_ENDPOINT` can point at a Hugging Face mirror.

### Environment

| Variable | Default |
| --- | --- |
| `SDCPP_IMAGE` | `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` |
| `SDCPP_GPU` | `L4` |
| `HF_ENDPOINT` | `https://huggingface.co` |

## Tests

```bash
python3 -m pytest
```

These tests do not download weights and do not need a GPU.
