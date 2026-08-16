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
python3 sdcpp_modal.py pull --all
python3 sdcpp_modal.py ls
python3 sdcpp_modal.py probe
python3 sdcpp_modal.py generate -p "a lovely cat" --recipe sd15 -o cat.png --publish
python3 sdcpp_modal.py publish cat.png --recipe sd15 -p "a lovely cat"
python3 sdcpp_modal.py cost
python3 sdcpp_modal.py cost --official
```

| Command | Where it runs | What it does |
| --- | --- | --- |
| `pull` | CPU | download URIs, a `--recipe`, or `--all` recipes onto volume `sdcpp-models` |
| `put` | CPU | upload a small local file (init image, mask) to `uploads/` |
| `ls` | CPU | list files already on that volume |
| `probe` | CUDA image, no GPU | print remote `sd-cli` flags |
| `generate` | CPU, then GPU (`SDCPP_GPU`, default `L40S`) | CPU pulls missing weights; GPU only loads them and runs `sd-cli` |
| `publish` | local + Hugging Face | upload a PNG into the multi-model gallery dataset |
| `cost` | local (+ optional Modal API) | print the local billed-session ledger |

`generate` first ensures missing URIs are on the volume **from a CPU container**. The GPU container only reloads those files and runs `sd-cli`. It does not download weights.

CPU pulls use `aria2c` (`-x 16 -s 16 -c -k 1M`) when it is on the image, then the Hugging Face CLI, then urllib. Several missing files download in parallel (`SDCPP_PULL_WORKERS`, default 4). Tokens stay in headers or `CIVITAI_TOKEN` query params and are redacted in logs.

```bash
python3 sdcpp_modal.py pull --all
python3 sdcpp_modal.py generate -p '{"high_level_description":"A fluffy orange cat"}' --recipe ideogram4 -o ideogram4.png --publish
```

Do not convert Ideogram4 weights yourself. `pull --recipe ideogram4` downloads the prebuilt GGUF pair from [`leejet/ideogram-4-GGUF`](https://huggingface.co/leejet/ideogram-4-GGUF) (`ideogram4-Q4_0.gguf` and `ideogram4_uncond-Q4_0.gguf`), plus the FLUX.2 VAE and Qwen3-VL GGUF. There is no `convert` command in this CLI. The FLUX.2 VAE is gated, so set `HF_TOKEN`. Ideogram4 prompts must be JSON.

The default GPU is `L40S`. A 24 GB `L4` can OOM on the diffusion compute buffer. `RTX6000` also works; A10 and A100 are blocked.

Idle CPU and GPU containers scale to zero after `SDCPP_IDLE_SECONDS` (default **10**). `min_containers=0`, so nothing stays warm when there are no requests.

Cost tracking lives in `sdcpp_hooks/cost.py` and `sdcpp_hooks/modal_meter.py`, not in `sd-cli`. Every `app.run()` and `.remote()` is a billed span. GPU containers also record enter→exit lifetime, including idle until scale-to-zero. Estimates use `modal.Workspace.billing.rates()` when available, otherwise a snapshot of those rates. `cost --official` prints the workspace invoice summary. Session and remote windows overlap, so the ledger does not add them together.

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

## Bundled models

This CLI ships **seven** recipes and nothing else. `pull --all` downloads every file they need onto volume `sdcpp-models`. Other Modal recipes (SD-Turbo, SSD-1B, Dreamlike, FLUX.1, and same-family klein/base variants) were removed.

Artificial Analysis text-to-image Elo is from the [open-weights leaderboard](https://artificialanalysis.ai/image/leaderboard/text-to-image/open-weights) on 2026-08-16. SDXL-Turbo / SD 2.1 / SD 1.5 are older checkpoints and are not ranked there the same way.

| Recipe | Model | AA Elo | Defaults | Volume files under `/models/` |
| --- | --- | --- | --- | --- |
| `ideogram4` | Ideogram 4.0 Q4_0 | 1217 | 1024², 20 steps, cfg 4.0 | `hf/leejet/ideogram-4-GGUF/main/ideogram4-Q4_0.gguf`, `ideogram4_uncond-Q4_0.gguf`; shared FLUX.2 VAE; `hf/unsloth/Qwen3-VL-8B-Instruct-GGUF/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf` |
| `flux2-klein` | FLUX.2 [klein] 9B Q4_0 | 1149 | 1024², 4 steps, cfg 1.0 | `hf/leejet/FLUX.2-klein-9B-GGUF/main/flux-2-klein-9b-Q4_0.gguf`; shared FLUX.2 VAE; `hf/unsloth/Qwen3-8B-GGUF/main/Qwen3-8B-Q4_K_M.gguf` |
| `flux2-dev` | FLUX.2 [dev] Q4_K_S | 1200 | 1024², 20 steps, cfg 1.0, euler | `hf/city96/FLUX.2-dev-gguf/main/flux2-dev-Q4_K_S.gguf`; shared FLUX.2 VAE; `hf/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/main/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf` |
| `z-image-turbo` | Z-Image Turbo Q3_K | 1131 | 512×1024, 8 steps, cfg 1.0 | `hf/leejet/Z-Image-Turbo-GGUF/main/z_image_turbo-Q3_K.gguf`; `hf/black-forest-labs/FLUX.1-schnell/main/ae.safetensors`; `hf/unsloth/Qwen3-4B-Instruct-2507-GGUF/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` |
| `sdxl-turbo` | SDXL-Turbo fp16 | — | 512², 4 steps, cfg 1.0, euler | `hf/stabilityai/sdxl-turbo/main/sd_xl_turbo_1.0_fp16.safetensors` |
| `sd2` | Stable Diffusion 2.1 | — | 512², 20 steps, cfg 7.0 | `hf/Manojb/stable-diffusion-2-1-base/main/v2-1_512-ema-pruned.safetensors` |
| `sd15` | Stable Diffusion 1.5 | — | 512², 20 steps, cfg 7.0 | `hf/stable-diffusion-v1-5/stable-diffusion-v1-5/main/v1-5-pruned-emaonly.safetensors` |

Shared FLUX.2 VAE (gated): `hf/black-forest-labs/FLUX.2-dev/main/ae.safetensors`. Used by Ideogram 4.0, FLUX.2 [klein], and FLUX.2 [dev]. Z-Image Turbo uses the Flux.1 schnell VAE, not the Flux.2 VAE. Qwen3-8B (klein) is not Qwen3-VL-8B (Ideogram4).

GGUF diffusion files still need their VAE and text-encoder companions. `pull --all` is the CPU path that fetches the full set.

### Volume inventory (2026-08-16)

CPU `pull --all` on volume `sdcpp-models` left only these weight files (77.60 GiB). Leftover FLUX.1-dev and the old extra recipes were deleted.

| Path under `/models/` | Bytes | Size |
| --- | ---: | ---: |
| `hf/leejet/ideogram-4-GGUF/main/ideogram4-Q4_0.gguf` | 5643820832 | 5.26 GiB |
| `hf/leejet/ideogram-4-GGUF/main/ideogram4_uncond-Q4_0.gguf` | 5643820832 | 5.26 GiB |
| `hf/unsloth/Qwen3-VL-8B-Instruct-GGUF/main/Qwen3-VL-8B-Instruct-Q4_K_M.gguf` | 5027785568 | 4.68 GiB |
| `hf/black-forest-labs/FLUX.2-dev/main/ae.safetensors` | 336211292 | 0.31 GiB |
| `hf/leejet/FLUX.2-klein-9B-GGUF/main/flux-2-klein-9b-Q4_0.gguf` | 5616208032 | 5.23 GiB |
| `hf/unsloth/Qwen3-8B-GGUF/main/Qwen3-8B-Q4_K_M.gguf` | 5027784512 | 4.68 GiB |
| `hf/city96/FLUX.2-dev-gguf/main/flux2-dev-Q4_K_S.gguf` | 19299128288 | 17.97 GiB |
| `hf/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF/main/Mistral-Small-3.2-24B-Instruct-2506-Q4_K_M.gguf` | 14333922848 | 13.35 GiB |
| `hf/leejet/Z-Image-Turbo-GGUF/main/z_image_turbo-Q3_K.gguf` | 3143559104 | 2.93 GiB |
| `hf/black-forest-labs/FLUX.1-schnell/main/ae.safetensors` | 335304388 | 0.31 GiB |
| `hf/unsloth/Qwen3-4B-Instruct-2507-GGUF/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf` | 2497281120 | 2.33 GiB |
| `hf/stabilityai/sdxl-turbo/main/sd_xl_turbo_1.0_fp16.safetensors` | 6938081905 | 6.46 GiB |
| `hf/Manojb/stable-diffusion-2-1-base/main/v2-1_512-ema-pruned.safetensors` | 5214604494 | 4.86 GiB |
| `hf/stable-diffusion-v1-5/stable-diffusion-v1-5/main/v1-5-pruned-emaonly.safetensors` | 4265146304 | 3.97 GiB |

```bash
python3 sdcpp_modal.py pull --all
python3 sdcpp_modal.py generate -p "a lovely cat" --recipe flux2-klein -o klein.png
python3 sdcpp_modal.py generate -p "a lovely cat" --recipe flux2-dev -o flux2.png
python3 sdcpp_modal.py generate -p "a rainy city at night" --recipe z-image-turbo -o zimage.png
```

`ideogram4`, `flux2-klein`, `flux2-dev`, and `z-image-turbo` also pass `--diffusion-fa` and `--offload-to-cpu`.

### Environment

| Variable | Default |
| --- | --- |
| `SDCPP_IMAGE` | `ghcr.io/leejet/stable-diffusion.cpp:master-cuda` |
| `SDCPP_GPU` | `L40S` (also `L4` or `RTX6000` / RTX PRO 6000; A10 and A100 are blocked) |
| `SDCPP_IDLE_SECONDS` | `10` (CPU and GPU scale to zero after this idle window) |
| `SDCPP_SECRET` | `sdcpp-tokens` (used only if that Modal secret exists) |
| `SDCPP_PULL_WORKERS` | `4` (parallel CPU downloads) |
| `HF_ENDPOINT` | `https://huggingface.co` |
| `SDCPP_GALLERY_DATASET` | `seachen/stable-diffusion-cpp-gallery` |
| `SDCPP_GITHUB_REPO` | `xiaoqianran/stable-diffusion.cpp` |
| `SDCPP_COST_LOG` | `~/.cache/sdcpp-modal/cost.jsonl` |

## Gallery dataset and Pages

Generated images are stored on the public Hugging Face dataset
[`seachen/stable-diffusion-cpp-gallery`](https://huggingface.co/datasets/seachen/stable-diffusion-cpp-gallery),
one folder per bundled recipe (`images/ideogram4`, `images/flux2-klein`,
`images/sd15`, ...). A new `--model-id` creates a new folder, so later models
do not need a schema change.

```bash
python3 -m pip install 'huggingface_hub>=0.26'
export HF_TOKEN=...
python3 sdcpp_modal.py publish cat.png --model-id sd15 -p "a lovely cat" --seed 42
```

`generate --publish` writes the prompt plus run facts into the sidecar: duration,
GPU name, CUDA version, torch version (or a note that sd-cli uses ggml), NVIDIA
driver, sd-cli version, Python, Modal GPU type, and the container image. Pages
cards show those fields under the prompt.

GitHub Actions workflow `.github/workflows/gallery-pages.yml` downloads that
dataset and deploys a paginated gallery to GitHub Pages (12 images per page,
filters for current and future model families):
https://xiaoqianran.github.io/stable-diffusion.cpp/

## Limits

- First-class path is `img_gen`. `vid_gen`, `adetailer`, `convert`, and `metadata` are not wrapped; some of their flags can still be forwarded if the remote binary accepts them.
- `put` is for small files (64 MiB). Weights should use `pull` or `modal volume put`.
- Size defaults match local `sd-cli` (512x512, 20 steps, cfg 7.0) unless you override them or use a recipe.

## Tests

```bash
python3 -m pytest
```

These tests do not download weights and do not need a GPU.
